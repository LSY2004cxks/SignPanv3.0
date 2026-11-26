from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.event_dataset import EventDataset, collate_fn
from src.models.frame_mlp import FrameMLP
from src.models.protodist import ProtoDist
from src.train.metrics import compute_metrics
from src.utils.io import load_yaml_with_base, write_json, write_jsonl, read_json


def load_model(run_dir: Path, cfg: Dict[str, Any], num_classes: int, feat_dim: int):
    ckpt = torch.load(run_dir / 'best.pt', map_location='cpu')
    model_type = cfg['model']['name']
    pool_mode = cfg['training'].get('pool', 'logsumexp')
    if model_type == 'frame_mlp':
        model = FrameMLP(feat_dim, num_classes, hidden_dim=cfg['model']['hidden_dim'],
                         emb_dim=cfg['model']['emb_dim'], dropout=cfg['model']['dropout'], pool_mode=pool_mode)
        model.load_state_dict(ckpt['state_dict'])
        return model
    elif model_type == 'protodist':
        labels_cfg = read_json(Path(cfg['paths']['location_labels']))
        codes = labels_cfg['codes']
        model = ProtoDist(codes=codes,
                          joint_names=cfg['pose']['joint_names'],
                          anchor_defs=cfg['model']['anchors'],
                          hand_rep=cfg['model'].get('hand_rep', 'center'),
                          tau=cfg['model'].get('tau', 0.5),
                          pool_mode=pool_mode)
        model.load_state_dict(ckpt['state_dict'])
        return model
    else:
        raise ValueError(f"Unknown model: {model_type}")


def eval_split(split: str, run_dir: Path, cfg: Dict[str, Any]):
    labels_csv = cfg['paths']['labels_csv']
    features_dir = cfg['paths']['features_dir']
    splits_dir = cfg['paths']['splits_dir']
    labels_cfg = cfg['paths']['location_labels']

    ds = EventDataset(labels_csv, features_dir, labels_cfg, split_txt=Path(splits_dir)/f"{split}.txt", cfg=cfg)
    if len(ds) == 0:
        return None
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)

    # feat dim
    sample = ds[0]
    feat_dim = sample.feat.shape[-1]

    model = load_model(run_dir, cfg, num_classes=len(ds.labels), feat_dim=feat_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    y_prob = []
    preds_rows = []

    for batch in dl:
        feat = batch['feat'].to(device)
        mask = batch['mask'].to(device)
        hand_xy = batch['hand_xy'].to(device)
        pose_xy = batch['pose_xy'].to(device)
        label = batch['label'].cpu().numpy()

        with torch.no_grad():
            if cfg['model']['name'] == 'frame_mlp':
                out = model(feat, mask)
            else:
                out = model(hand_xy, pose_xy, mask)
            logits = out['logits_clip']
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = prob.argmax(axis=-1)

        y_true.extend(label.tolist())
        y_pred.extend(pred.tolist())
        y_prob.append(prob)

        # per-sample outputs
        for i in range(len(label)):
            top3 = np.argsort(-prob[i])[:3]
            top3_codes = [ds.labels[t] for t in top3]
            top3_prob = [float(prob[i][t]) for t in top3]
            margin = float(top3_prob[0] - top3_prob[1]) if len(top3_prob) > 1 else float(top3_prob[0])
            preds_rows.append({
                'event_id': batch['event_id'][i],
                'signer_id': batch['signer_id'][i],
                'sample_id': batch['sample_id'][i],
                'y_true': ds.labels[label[i]],
                'y_pred_top1': ds.labels[pred[i]],
                'top3': top3_codes,
                'prob_top3': top3_prob,
                'margin': margin,
            })

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.concatenate(y_prob, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob)
    write_json(run_dir / f"summary_{split}.json", {k: v for k, v in metrics.items() if k != 'confusion'})
    # confusion matrix
    import pandas as pd
    cm = metrics['confusion']
    pd.DataFrame(cm).to_csv(run_dir / f"confusion_{split}.csv", index=False)

    write_jsonl(run_dir / f"predictions_{split}.jsonl", preds_rows)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml_with_base(run_dir / 'config_resolved.yaml')

    for split in ['val', 'test']:
        res = eval_split(split, run_dir, cfg)
        if res is None:
            print(f"No samples for {split}")
        else:
            print(f"[{split}] acc={res['acc']:.4f} macro_f1={res['macro_f1']:.4f} acc@3={res['acc@3']:.4f}")


if __name__ == '__main__':
    main()
