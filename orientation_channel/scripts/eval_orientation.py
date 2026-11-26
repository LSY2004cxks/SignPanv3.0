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
from src.utils.metrics import topk_accuracy, mrr_score, per_class_accuracy, ece_score
from src.utils.calibration import temperature_scale, apply_temperature, save_temperature
from src.utils.plotting import reliability_diagram


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
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--use_calibration', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml_with_inherit(args.config)
    output_root = Path(args.out_dir or cfg.get('output_root', 'outputs'))
    (output_root / 'results').mkdir(parents=True, exist_ok=True)
    (output_root / 'calibration').mkdir(parents=True, exist_ok=True)

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

    all_logits = []
    all_labels = []
    meta = []
    use_amp = (device == 'cuda')

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with autocast(enabled=use_amp):
                logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch['label'].cpu())
            meta.append(batch)

    logits = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)

    metrics = topk_accuracy(logits, labels_t, ks=(1, 3))
    metrics['mrr'] = mrr_score(logits, labels_t)

    per_cls = per_class_accuracy(logits, labels_t, num_classes=len(labels))
    metrics['per_class_acc'] = {labels[k]: v for k, v in per_cls.items()}

    # calibration
    temperature = None
    cal_cfg = cfg.get('calibration', {})
    if args.calibrate and args.split == 'val' and cal_cfg.get('enable', True):
        temperature = temperature_scale(logits.to(device), labels_t.to(device), init_t=cal_cfg.get('temperature_init', 1.0))
        save_temperature(str(output_root / 'calibration' / 'temperature.json'), temperature)

    # use calibration if requested and exists
    if args.use_calibration:
        temp_path = output_root / 'calibration' / 'temperature.json'
        if temp_path.exists():
            temperature = json.loads(temp_path.read_text())['temperature']

    logits_for_prob = logits
    if temperature is not None:
        logits_for_prob = apply_temperature(logits, temperature)

    probs = torch.softmax(logits_for_prob, dim=1).numpy()
    ece = ece_score(probs, labels_t.numpy(), n_bins=cal_cfg.get('ece_bins', 15))
    metrics['ece'] = ece

    reliability_diagram(probs, labels_t.numpy(), str(output_root / 'calibration' / f"reliability_{args.split}.png"), n_bins=cal_cfg.get('ece_bins', 15))

    # export top3 results
    results = []
    idx = 0
    for batch in meta:
        B = len(batch['label'])
        p = probs[idx:idx+B]
        top3_idx = np.argsort(-p, axis=1)[:, :3]
        top3_probs = np.take_along_axis(p, top3_idx, axis=1)

        for i in range(B):
            res = {
                'video_id': batch['video_id'][i],
                'hand': batch['hand'][i],
                'event_id': batch['event_id'][i],
                'span': batch['span'][i],
                'top3_labels': [labels[j] for j in top3_idx[i].tolist()],
                'top3_probs': [float(x) for x in top3_probs[i].tolist()],
                'conf': float(top3_probs[i][0]),
            }
            if 'quality_flags' in batch:
                res['quality_flags'] = batch['quality_flags'][i]
            results.append(res)
        idx += B

    out_jsonl = output_root / 'results' / f"orientation_top3_{args.split}.jsonl"
    write_jsonl(str(out_jsonl), results)

    metrics_path = output_root / 'results' / f"metrics_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print('Metrics:', json.dumps(metrics, indent=2))
    print('Saved top3 ->', out_jsonl)


if __name__ == '__main__':
    main()
