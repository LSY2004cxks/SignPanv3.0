import argparse
import csv
import json
from pathlib import Path


def read_best_val(metrics_csv: Path):
    if not metrics_csv.exists():
        return None
    best = None
    with metrics_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("split") != "val":
                continue
            try:
                acc1 = float(row.get("acc@1", 0.0))
                acc3 = float(row.get("acc@3", 0.0))
                mrr = float(row.get("mrr", 0.0))
                loss = float(row.get("loss", 0.0))
                epoch = int(row.get("epoch", 0))
            except Exception:
                continue
            item = {"epoch": epoch, "val_acc1": acc1, "val_acc3": acc3, "val_mrr": mrr, "val_loss": loss}
            if best is None or item["val_acc1"] > best["val_acc1"]:
                best = item
    return best


def read_test_metrics(metrics_json: Path):
    if not metrics_json.exists():
        return {}
    try:
        d = json.loads(metrics_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "test_acc1": float(d.get("acc@1", 0.0)),
        "test_acc3": float(d.get("acc@3", 0.0)),
        "test_mrr": float(d.get("mrr", 0.0)),
        "test_ece": float(d.get("ece", 0.0)) if "ece" in d else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="e.g. outputs_ablation/core_20260303_180000")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(run_root)

    rows = []
    for d in sorted([p for p in run_root.iterdir() if p.is_dir()]):
        best_val = read_best_val(d / "logs" / "metrics.csv")
        test_m = read_test_metrics(d / "results" / "metrics_test.json")
        if best_val is None:
            continue
        row = {"experiment": d.name}
        row.update(best_val)
        row.update(test_m)
        rows.append(row)

    if not rows:
        print("No valid runs found under", run_root)
        return

    rows.sort(key=lambda x: x.get("test_acc1", 0.0), reverse=True)

    print("experiment,val_epoch,val_acc1,val_acc3,val_mrr,val_loss,test_acc1,test_acc3,test_mrr,test_ece")
    for r in rows:
        print(
            f"{r['experiment']},"
            f"{r.get('epoch','')},"
            f"{r.get('val_acc1',0.0):.4f},"
            f"{r.get('val_acc3',0.0):.4f},"
            f"{r.get('val_mrr',0.0):.4f},"
            f"{r.get('val_loss',0.0):.4f},"
            f"{r.get('test_acc1',0.0):.4f},"
            f"{r.get('test_acc3',0.0):.4f},"
            f"{r.get('test_mrr',0.0):.4f},"
            f"{'' if r.get('test_ece') is None else format(r.get('test_ece'), '.4f')}"
        )

    out_json = run_root / "summary.json"
    out_md = run_root / "summary.md"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| experiment | best val epoch | val acc@1 | val acc@3 | val mrr | val loss | test acc@1 | test acc@3 | test mrr | test ece |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r.get('epoch','')} | "
            f"{r.get('val_acc1',0.0):.4f} | {r.get('val_acc3',0.0):.4f} | "
            f"{r.get('val_mrr',0.0):.4f} | {r.get('val_loss',0.0):.4f} | "
            f"{r.get('test_acc1',0.0):.4f} | {r.get('test_acc3',0.0):.4f} | "
            f"{r.get('test_mrr',0.0):.4f} | "
            f"{'' if r.get('test_ece') is None else format(r.get('test_ece'), '.4f')} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", out_json)
    print("Saved:", out_md)


if __name__ == "__main__":
    main()

