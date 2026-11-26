import argparse
import json
import random
from pathlib import Path
import pandas as pd

from src.utils.io import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train_signers', type=int, default=18)
    ap.add_argument('--val_signers', type=int, default=5)
    ap.add_argument('--test_signers', type=int, default=5)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--remaining', choices=['drop', 'extra_test'], default='drop')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    signers_csv = data_root / 'metadata' / 'signers.csv'
    if not signers_csv.exists():
        raise FileNotFoundError(
            f"Missing {signers_csv}. Please run:\n"
            f"python -m scripts.prepare_data28_manifest --data_root {data_root} --label_root {data_root}"
        )

    signers = pd.read_csv(signers_csv)
    if 'signer_id' not in signers.columns:
        raise ValueError(f"Invalid signers.csv format: missing signer_id column in {signers_csv}")
    signer_ids = sorted(signers['signer_id'].unique().tolist())

    random.Random(args.seed).shuffle(signer_ids)
    train_ids = signer_ids[:args.train_signers]
    val_ids = signer_ids[args.train_signers:args.train_signers + args.val_signers]
    test_ids = signer_ids[args.train_signers + args.val_signers:args.train_signers + args.val_signers + args.test_signers]

    remaining = signer_ids[args.train_signers + args.val_signers + args.test_signers:]
    if remaining and args.remaining == 'extra_test':
        test_ids += remaining

    # load annotations
    ann_path = data_root / 'annotations' / 'mhsl_events.jsonl'
    if ann_path.exists():
        events = read_jsonl(str(ann_path))
    else:
        # fallback: read all json in annotations
        events = []
        for p in (data_root / 'annotations').glob('*.json'):
            events += json.load(p.open('r', encoding='utf-8'))

    # build optional mappings for backward compatibility
    uid2signer = {}
    if 'video_uid' in signers.columns:
        uid2signer = dict(zip(signers['video_uid'], signers['signer_id']))
    vid2signers = {}
    if 'video_id' in signers.columns:
        for _, row in signers[['video_id', 'signer_id']].iterrows():
            vid2signers.setdefault(row['video_id'], set()).add(row['signer_id'])

    splits = {'train': [], 'val': [], 'test': []}
    unresolved = 0
    for e in events:
        signer = e.get('signer_id')
        if signer is None and e.get('video_uid') is not None:
            signer = uid2signer.get(e['video_uid'])
        if signer is None:
            vid = e.get('video_id')
            candidates = sorted(vid2signers.get(vid, []))
            signer = candidates[0] if len(candidates) == 1 else None

        if signer is None:
            unresolved += 1
            continue
        if signer in train_ids:
            splits['train'].append(e)
        elif signer in val_ids:
            splits['val'].append(e)
        elif signer in test_ids:
            splits['test'].append(e)
        else:
            if args.remaining == 'extra_test':
                splits['test'].append(e)
            # drop otherwise

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, rows in splits.items():
        write_jsonl(str(out_dir / f"{k}.jsonl"), rows)
    print('Saved splits to', out_dir)
    print({k: len(v) for k, v in splits.items()})
    if unresolved > 0:
        print(f'Warning: skipped events with unresolved signer: {unresolved}')


if __name__ == '__main__':
    main()
