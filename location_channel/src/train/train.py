from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.event_dataset import EventDataset, collate_fn
from src.models.frame_mlp import FrameMLP
from src.models.protodist import ProtoDist
from src.train.losses import compute_class_weights
from src.train.metrics import compute_metrics
from src.utils.io import load_yaml_with_base, save_yaml, ensure_dir, read_json
from src.utils.seed import set_seed
from src.utils.logging import TBLogger, JsonlLogger


def build_model(cfg: Dict[str, Any], num_classes: int, feat_dim: int):
    model_type = cfg['model']['name']
    pool_mode = cfg['training'].get('pool', 'logsumexp')
    if model_type == 'frame_mlp':
        return FrameMLP(feat_dim, num_classes,
                        hidden_dim=cfg['model']['hidden_dim'],
                        emb_dim=cfg['model']['emb_dim'],
                        dropout=cfg['model']['dropout'],
                        pool_mode=pool_mode)
    elif model_type == 'protodist':
        labels_cfg = read_json(Path(cfg['paths']['location_labels']))
        codes = labels_cfg['codes']
        return ProtoDist(codes=codes,
                         joint_names=cfg['pose']['joint_names'],
                         anchor_defs=cfg['model']['anchors'],
                         hand_rep=cfg['model'].get('hand_rep', 'center'),
                         tau=cfg['model'].get('tau', 0.5),
                         pool_mode=pool_mode)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--labels', type=str, required=True)
    ap.add_argument('--features', type=str, required=True)
    ap.add_argument('--splits', type=str, required=True)
    ap.add_argument('--run_name', type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml_with_base(Path(args.config))
    cfg['paths']['labels_csv'] = args.labels
    cfg['paths']['features_dir'] = args.features
    cfg['paths']['splits_dir'] = args.splits

    set_seed(cfg['seed'])

    run_dir = ensure_dir(Path('runs') / args.run_name)
    save_yaml(run_dir / 'config_resolved.yaml', cfg)
    # save seed and split manifest for audit
    (run_dir / 'seed.txt').write_text(str(cfg['seed']), encoding='utf-8')
    manifest_src = Path(args.splits) / 'split_manifest.json'
    if manifest_src.exists():
        (run_dir / 'split_manifest.json').write_text(manifest_src.read_text(encoding='utf-8'), encoding='utf-8')
    # git commit (best-effort)
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2]).decode().strip()
    except Exception:
        commit = 'unknown'
    (run_dir / 'git_commit.txt').write_text(commit, encoding='utf-8')

    # datasets
    labels_cfg = cfg['paths']['location_labels']
    ds_train = EventDataset(args.labels, args.features, labels_cfg, split_txt=Path(args.splits) / 'train.txt', cfg=cfg)
    ds_val = EventDataset(args.labels, args.features, labels_cfg, split_txt=Path(args.splits) / 'val.txt', cfg=cfg)

    if len(ds_train) == 0:
        raise RuntimeError("Empty training set.")

    dl_train = DataLoader(ds_train, batch_size=cfg['training']['batch_size'], shuffle=True,
                          num_workers=cfg['training'].get('num_workers', 0), collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=cfg['training']['batch_size'], shuffle=False,
                        num_workers=cfg['training'].get('num_workers', 0), collate_fn=collate_fn)

    # feat dim
    feat_dim = ds_train[0].feat.shape[-1]

    model = build_model(cfg, num_classes=len(ds_train.labels), feat_dim=feat_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # loss
    labels_train = [int(ds_train.code2id[r['label_loc']]) for r in ds_train.rows if r['label_loc'] in ds_train.code2id]
    weights = compute_class_weights(labels_train, num_classes=len(ds_train.labels),
                                    smoothing=cfg['training']['weight_smooth']).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg['training']['epochs']))

    tb = TBLogger(run_dir / 'tb')
    jl = JsonlLogger(run_dir / 'train_log.jsonl')

    best = -1.0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        losses = []
        for batch in dl_train:
            feat = batch['feat'].to(device)
            mask = batch['mask'].to(device)
            hand_xy = batch['hand_xy'].to(device)
            pose_xy = batch['pose_xy'].to(device)
            y = batch['label'].to(device)

            if cfg['model']['name'] == 'frame_mlp':
                out = model(feat, mask)
            else:
                out = model(hand_xy, pose_xy, mask)
            logits = out['logits_clip']
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.item()))

        scheduler.step()

        # eval
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in dl_val:
                feat = batch['feat'].to(device)
                mask = batch['mask'].to(device)
                hand_xy = batch['hand_xy'].to(device)
                pose_xy = batch['pose_xy'].to(device)
                y = batch['label'].cpu().numpy()
                if cfg['model']['name'] == 'frame_mlp':
                    out = model(feat, mask)
                else:
                    out = model(hand_xy, pose_xy, mask)
                logits = out['logits_clip']
                prob = torch.softmax(logits, dim=-1).cpu().numpy()
                pred = prob.argmax(axis=-1)

                y_true.extend(y.tolist())
                y_pred.extend(pred.tolist())
                y_prob.append(prob)

        y_prob = np.concatenate(y_prob, axis=0) if y_prob else None
        metrics = compute_metrics(np.asarray(y_true), np.asarray(y_pred), y_prob=y_prob)

        tb.scalar('train/loss', float(np.mean(losses)), epoch)
        tb.scalar('val/acc', metrics['acc'], epoch)
        tb.scalar('val/macro_f1', metrics['macro_f1'], epoch)
        tb.scalar('val/acc@3', metrics['acc@3'], epoch)

        jl.log({'epoch': epoch, 'train_loss': float(np.mean(losses)), **{k: metrics[k] for k in ['acc','macro_f1','macro_recall','acc@3']}})

        save_metric = cfg['training'].get('save_metric', 'macro_f1')
        current = metrics['macro_f1'] if save_metric == 'macro_f1' else metrics['acc']
        if current > best:
            best = current
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'args': cfg,
            }, run_dir / 'best.pt')

        print(f"Epoch {epoch:03d} | train_loss {np.mean(losses):.4f} | val acc {metrics['acc']:.3f} | val acc@3 {metrics['acc@3']:.3f} | val macro_f1 {metrics['macro_f1']:.3f}")

    tb.close()


if __name__ == '__main__':
    main()
