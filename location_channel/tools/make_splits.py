import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Allow running as standalone
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def read_labels(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Signer-disjoint split 18/5/5")
    ap.add_argument('--labels', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train_signers', type=int, default=18)
    ap.add_argument('--val_signers', type=int, default=5)
    ap.add_argument('--test_signers', type=int, default=5)
    args = ap.parse_args()

    labels = read_labels(Path(args.labels))
    signer2rows = defaultdict(list)
    for r in labels:
        signer2rows[r['signer_id']].append(r)

    signers = sorted(signer2rows.keys())
    if len(signers) < (args.train_signers + args.val_signers + args.test_signers):
        raise ValueError(f"Not enough signers: {len(signers)}")

    rnd = random.Random(args.seed)
    rnd.shuffle(signers)
    train_s = signers[:args.train_signers]
    val_s = signers[args.train_signers:args.train_signers + args.val_signers]
    test_s = signers[args.train_signers + args.val_signers:args.train_signers + args.val_signers + args.test_signers]

    def rows_for(signer_list):
        rows = []
        for s in signer_list:
            rows.extend(signer2rows[s])
        return rows

    train_rows = rows_for(train_s)
    val_rows = rows_for(val_s)
    test_rows = rows_for(test_s)

    out_dir = ensure_dir(Path(args.out))

    def write_split(name, rows):
        p = out_dir / f"{name}.txt"
        p.write_text("\n".join([r['sample_id'] for r in rows]) + "\n", encoding='utf-8')
        return p

    write_split('train', train_rows)
    write_split('val', val_rows)
    write_split('test', test_rows)

    def stats(rows):
        return {
            'samples': len(rows),
            'signers': len(set([r['signer_id'] for r in rows])),
            'label_top5': Counter([r['label_loc'] for r in rows]).most_common(5)
        }

    manifest = {
        'seed': args.seed,
        'train_signers': train_s,
        'val_signers': val_s,
        'test_signers': test_s,
        'train_stats': stats(train_rows),
        'val_stats': stats(val_rows),
        'test_stats': stats(test_rows)
    }

    (out_dir / 'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
