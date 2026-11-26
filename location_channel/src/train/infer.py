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
from src.utils.io import load_yaml_with_base, write_jsonl, read_json


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, required=True)
    ap.add_argument('--labels', type=str, required=True)
    ap.add_argument('--features', type=str, required=True)
    ap.add_argument('--splits', type=str, required=False)
    ap.add_argument('--split', type=str, default=None, help='train/val/test or None to use all labels')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml_with_base(run_dir / 'config_resolved.yaml')
    cfg['paths']['labels_csv'] = args.labels
    cfg['paths']['features_dir'] = args.features
    if args.splits:
        cfg['paths']['splits_dir'] = args.splits

    labels_cfg = cfg['paths']['location_labels']
    split_txt = None
    if args.split and args.splits:
        split_txt = Path(args.splits) / f"{args.split}.txt"

    ds = EventDataset(args.labels, args.features, labels_cfg, split_txt=split_txt, cfg=cfg)
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)

    feat_dim = ds[0].feat.shape[-1]
    model = load_model(run_dir, cfg, num_classes=len(ds.labels), feat_dim=feat_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    rows = []
    for batch in dl:
        feat = batch['feat'].to(device)
        mask = batch['mask'].to(device)
        hand_xy = batch['hand_xy'].to(device)
        pose_xy = batch['pose_xy'].to(device)

        with torch.no_grad():
            if cfg['model']['name'] == 'frame_mlp':
                out = model(feat, mask)
            else:
                out = model(hand_xy, pose_xy, mask)
            logits = out['logits_clip']
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = prob.argmax(axis=-1)

        for i in range(len(pred)):
            top3 = np.argsort(-prob[i])[:3]
            top3_codes = [ds.labels[t] for t in top3]
            top3_prob = [float(prob[i][t]) for t in top3]
            margin = float(top3_prob[0] - top3_prob[1]) if len(top3_prob) > 1 else float(top3_prob[0])
            rows.append({
                'event_id': batch['event_id'][i],
                'signer_id': batch['signer_id'][i],
                'sample_id': batch['sample_id'][i],
                'y_true': batch['label_str'][i],
                'y_pred_top1': ds.labels[pred[i]],
                'top3': top3_codes,
                'prob_top3': top3_prob,
                'margin': margin,
            })

    write_jsonl(Path(args.out), rows)
    print(f"Wrote {len(rows)} predictions to {args.out}")


if __name__ == '__main__':
    main()
