import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.utils.io import load_yaml_with_inherit, write_jsonl
from src.data.dataset import OrientationDataset
from src.models.orientation_model import OrientationModel
from src.utils.calibration import apply_temperature


def load_labels(path: str):
    labels = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        labels.append(line)
    return labels


def collate_batch(batch):
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


def to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def build_model(cfg, labels):
    params = cfg['params']
    model_cfg = cfg['model']
    text_prototypes = None
    if model_cfg.get('use_text', False):
        proto_path = params.get('prototypes_path')
        if proto_path and Path(proto_path).exists():
            text_prototypes = torch.load(proto_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Missing prototypes: {proto_path}")

    model = OrientationModel(
        num_classes=len(labels),
        roi_backbone=model_cfg.get('roi_backbone', 'resnet18'),
        roi_feat_dim=model_cfg.get('roi_feat_dim', 512),
        geom_encoder_type=model_cfg.get('geom_encoder_type', 'stgcn'),
        geom_dim=model_cfg.get('geom_dim', 256),
        normal_dim=model_cfg.get('normal_dim', 128),
        fused_dim=model_cfg.get('fused_dim', 256),
        cross_attn_heads=model_cfg.get('cross_attn_heads', 4),
        use_roi=model_cfg.get('use_roi', True),
        use_geom=model_cfg.get('use_geom', True),
        use_normal=model_cfg.get('use_normal', True),
        use_text=model_cfg.get('use_text', True),
        text_prototypes=text_prototypes,
        use_text_as_kv=model_cfg.get('use_text_as_kv', False),
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--use_calibration', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml_with_inherit(args.config)
    output_root = Path(args.out_dir or cfg.get('output_root', 'outputs'))
    (output_root / 'results').mkdir(parents=True, exist_ok=True)

    params = cfg['params']
    labels = load_labels(params['labels_file'])

    ds = OrientationDataset(
        events_jsonl=str(Path(params['split_dir']) / f"{args.split}.jsonl"),
        labels=labels,
        precomputed_dir=params.get('precomputed_dir'),
        external_roi_dir=params.get('external_roi_dir'),
        external_skeleton_dir=params.get('external_skeleton_dir'),
        roi_size=params.get('roi_size', 224),
        T=params.get('T', 8),
        sampling=params.get('sampling', 'biased_start'),
        sampling_bias=params.get('sampling_bias', 0.7),
        use_precomputed=params.get('use_precomputed', True),
        use_roi=cfg['model'].get('use_roi', True),
        use_geom=cfg['model'].get('use_geom', True),
        use_normal=cfg['model'].get('use_normal', True),
        roi_input=params.get('roi_input', 'feat'),
    )

    batch_size = int(args.batch_size or cfg['train'].get('batch_size', 16))
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=cfg['train'].get('num_workers', 4),
                        collate_fn=collate_batch)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = build_model(cfg, labels).to(device)
    ckpt_path = args.ckpt or (output_root / 'checkpoints' / 'best.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    # load calibration if requested
    temperature = None
    if args.use_calibration:
        temp_path = output_root / 'calibration' / 'temperature.json'
        if temp_path.exists():
            temperature = json.loads(temp_path.read_text())['temperature']

    hitl = cfg.get('hitl', {})
    th_high = hitl.get('theta_high', 0.85)
    th_low = hitl.get('theta_low', 0.60)
    margin_min = hitl.get('margin_min', 0.12)
    low_quality_flags = set(hitl.get('low_quality_flags', ['FRAME_MISSING', 'NO_HAND', 'ROI_TOO_SMALL']))

    results = []
    use_amp = (device == 'cuda')
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with autocast(enabled=use_amp):
                logits = model(batch)
            if temperature is not None:
                logits = apply_temperature(logits, temperature)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            top3_idx = np.argsort(-probs, axis=1)[:, :3]
            top3_probs = np.take_along_axis(probs, top3_idx, axis=1)

            for i in range(len(batch['label'])):
                conf = float(top3_probs[i][0])
                margin = float(top3_probs[i][0] - top3_probs[i][1]) if top3_probs.shape[1] > 1 else conf
                quality_flags_i = batch['quality_flags'][i] if 'quality_flags' in batch else []
                has_low_quality = any(str(q) in low_quality_flags for q in quality_flags_i)

                if (not has_low_quality) and conf >= th_high and margin >= margin_min:
                    gate = 'auto_fill'
                elif conf >= th_low:
                    gate = 'recommend_top3'
                else:
                    gate = 'low_conf'
                res = {
                    'video_id': batch['video_id'][i],
                    'hand': batch['hand'][i],
                    'event_id': batch['event_id'][i],
                    'span': batch['span'][i],
                    'top3_labels': [labels[j] for j in top3_idx[i].tolist()],
                    'top3_probs': [float(x) for x in top3_probs[i].tolist()],
                    'conf': conf,
                    'margin': margin,
                    'hitl_gate': gate,
                }
                if 'quality_flags' in batch:
                    res['quality_flags'] = quality_flags_i
                    res['quality_downgraded'] = bool(has_low_quality)
                results.append(res)

    out_jsonl = output_root / 'results' / f"orientation_top3_{args.split}.jsonl"
    write_jsonl(str(out_jsonl), results)
    print('Saved', out_jsonl)


if __name__ == '__main__':
    main()
