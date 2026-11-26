import argparse
import csv
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.utils.io import load_yaml_with_inherit
from src.utils.seed import seed_everything
from src.data.dataset import OrientationDataset
from src.models.orientation_model import OrientationModel
from src.utils.metrics import topk_accuracy, mrr_score


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


def get_git_hash(repo_root: Path):
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'N/A'


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


def evaluate(model, loader, device, class_weights=None, focal_gamma=0.0):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total = 0
    use_amp = str(device).startswith('cuda')
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with autocast(enabled=use_amp):
                logits = model(batch)
                labels = batch['label']
                if focal_gamma and focal_gamma > 0:
                    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
                    pt = torch.exp(-ce)
                    loss = ((1 - pt) ** focal_gamma * ce).mean()
                else:
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = topk_accuracy(logits, labels, ks=(1, 3))
    metrics['mrr'] = mrr_score(logits, labels)
    metrics['loss'] = total_loss / max(total, 1)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    cfg = load_yaml_with_inherit(args.config)
    output_root = Path(args.out_dir or cfg.get('output_root', 'outputs'))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'checkpoints').mkdir(exist_ok=True)
    (output_root / 'logs').mkdir(exist_ok=True)

    seed_everything(cfg.get('seed', 42))

    params = cfg['params']
    labels = load_labels(params['labels_file'])

    train_ds = OrientationDataset(
        events_jsonl=str(Path(params['split_dir']) / 'train.jsonl'),
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
    val_ds = OrientationDataset(
        events_jsonl=str(Path(params['split_dir']) / 'val.jsonl'),
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

    train_cfg = cfg['train']
    device = train_cfg.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    train_loader = DataLoader(train_ds, batch_size=train_cfg.get('batch_size', 16),
                              shuffle=True, num_workers=train_cfg.get('num_workers', 4),
                              collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get('batch_size', 16),
                            shuffle=False, num_workers=train_cfg.get('num_workers', 4),
                            collate_fn=collate_batch)

    model = build_model(cfg, labels).to(device)

    # optional class balance
    class_weights = None
    if train_cfg.get('class_balance') == 'inverse':
        counts = torch.zeros(len(labels))
        for e in train_ds.events:
            counts[train_ds.label2id[e['label_orientation']]] += 1
        counts = torch.clamp(counts, min=1)
        class_weights = (counts.max() / counts).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get('lr', 1e-3),
                                  weight_decay=train_cfg.get('weight_decay', 1e-4))
    scaler = GradScaler(enabled=False)

    writer = SummaryWriter(log_dir=str(output_root / 'logs' / 'tb'))
    metrics_path = output_root / 'logs' / 'metrics.csv'
    if not metrics_path.exists():
        with metrics_path.open('w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['epoch', 'split', 'loss', 'acc@1', 'acc@3', 'mrr'])

    best_metric = -1.0
    monitor = train_cfg.get('monitor', 'acc@1')
    focal_gamma = train_cfg.get('focal_gamma', 0.0)

    # save meta
    meta = {
        'config': cfg,
        'git_hash': get_git_hash(Path(__file__).resolve().parents[1]),
    }
    (output_root / 'logs' / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    for epoch in range(1, train_cfg.get('epochs', 50) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        all_logits = []
        all_labels = []
        for batch in train_loader:
            batch = to_device(batch, device)
            with autocast(enabled=False):
                logits = model(batch)
                labels_b = batch['label']
                if focal_gamma and focal_gamma > 0:
                    ce = F.cross_entropy(logits, labels_b, weight=class_weights, reduction='none')
                    pt = torch.exp(-ce)
                    loss = ((1 - pt) ** focal_gamma * ce).mean()
                else:
                    loss = F.cross_entropy(logits, labels_b, weight=class_weights)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels_b.size(0)
            total += labels_b.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels_b.detach().cpu())

        train_logits = torch.cat(all_logits, dim=0)
        train_labels = torch.cat(all_labels, dim=0)
        train_metrics = topk_accuracy(train_logits, train_labels, ks=(1, 3))
        train_metrics['mrr'] = mrr_score(train_logits, train_labels)
        train_metrics['loss'] = total_loss / max(total, 1)

        val_metrics = evaluate(model, val_loader, device, class_weights, focal_gamma)

        # logging
        with metrics_path.open('a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, 'train', train_metrics['loss'], train_metrics['acc@1'], train_metrics['acc@3'], train_metrics['mrr']])
            writer_csv.writerow([epoch, 'val', val_metrics['loss'], val_metrics['acc@1'], val_metrics['acc@3'], val_metrics['mrr']])

        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/acc@1', train_metrics['acc@1'], epoch)
        writer.add_scalar('train/acc@3', train_metrics['acc@3'], epoch)
        writer.add_scalar('train/mrr', train_metrics['mrr'], epoch)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/acc@1', val_metrics['acc@1'], epoch)
        writer.add_scalar('val/acc@3', val_metrics['acc@3'], epoch)
        writer.add_scalar('val/mrr', val_metrics['mrr'], epoch)

        # save best
        score = val_metrics.get(monitor, val_metrics.get('acc@1', 0.0))
        if score > best_metric:
            best_metric = score
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'config': cfg,
                'labels': labels,
            }, output_root / 'checkpoints' / 'best.pt')

        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'config': cfg,
            'labels': labels,
        }, output_root / 'checkpoints' / 'last.pt')

        print(f"Epoch {epoch}: train acc@1={train_metrics['acc@1']:.4f} val acc@1={val_metrics['acc@1']:.4f}")

    writer.close()
    print('Done. Outputs in', output_root)


if __name__ == '__main__':
    main()
