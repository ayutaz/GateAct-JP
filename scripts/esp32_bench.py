"""ESP32-S3(またはホストビルド)の GateAct ファームウェアを計測し、Python と照合する。

    # 実機(書き込み済み)。pyserial は device グループ
    uv run --group device python scripts/esp32_bench.py --port /dev/cu.usbmodem1101 \
        --checkpoint runs/<name>/<name>_s0.pt --out results/device/<name>_esp32s3.json
    # 同じ手順をホストビルドで(実機との比較用)
    uv run python scripts/esp32_bench.py --host-exe build/gateact_host --checkpoint ... --out ...

起動ログ(INFO / SELFTEST / BENCH / MEM)を集めたあと、評価セットの全文をシリアルで送り、
実機の出力スコアを IR simulate と照合する(完全一致率)。実機の予測で分類指標も出す。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gateact.benchmarks.device import HostProcess, SerialPort, Transport, query, read_boot
from gateact.data.labels import LabelSpec
from gateact.data.source import DEFAULT_DIR
from gateact.evaluation.metrics import classification_report
from gateact.evaluation.runner import load_splits
from gateact.export.pipeline import export_bundle, simulate_bundle
from gateact.features.hashing import extract_bits
from gateact.training.lut_trainer import load_checkpoint
from gateact.utils import git_sha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port")
    src.add_argument("--host-exe", type=Path)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--compress", default="fast")
    ap.add_argument("--max-texts", type=int, default=0, help="0 = 評価セット全文")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--labels", type=Path, default=Path("configs/labels_v1.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = LabelSpec.load(args.labels)
    splits, data_sha = load_splits(args.data_dir, spec)
    in_scope = [(d.example.text, d.label, d.example.category) for d in splits.test]
    oos = [(e.text, None, e.category) for e, _ in splits.test_out_of_scope]
    rows = in_scope + oos
    if args.max_texts:
        rows = rows[: args.max_texts]
    texts = [r[0] for r in rows]

    model, mcfg, fcfg, meta = load_checkpoint(args.checkpoint)
    bundle = export_bundle(model, input_width=fcfg.width, name="gateact", compress=args.compress)
    expected = simulate_bundle(bundle, extract_bits(texts, fcfg))

    t: Transport
    if args.port:
        t = SerialPort(args.port, args.baud, reset=not args.no_reset)
    else:
        t = HostProcess(args.host_exe)
    try:
        boot = read_boot(t)
        if not boot.ready:
            raise SystemExit("READY が来なかった(ログ: " + " | ".join(boot.raw[-5:]) + ")")
        t0 = time.monotonic()
        results = query(t, texts)
        wall = time.monotonic() - t0
    finally:
        t.close()

    mismatch = [
        i
        for i, (r, e) in enumerate(zip(results, expected, strict=True))
        if not r.get("ok") or r["scores"] != e.tolist()
    ]
    names = meta["labels"]
    labeled = [(r, row) for r, row in zip(results, rows, strict=True) if row[1] is not None]
    report = classification_report(
        [row[1] for _, row in labeled],
        [names[r["id"]] for r, _ in labeled],
        spec,
        categories=[row[2] for _, row in labeled],
    )
    report.pop("confusion", None)
    out = {
        "platform": "host" if args.host_exe else "esp32s3",
        "port": args.port,
        "checkpoint": str(args.checkpoint),
        "git_sha": git_sha(),
        "data_sha256": data_sha,
        "boot": boot.records,
        "query": {
            "n": len(results),
            "score_mismatch": len(mismatch),
            "exact_match_rate": 1 - len(mismatch) / max(len(results), 1),
            "first_mismatches": mismatch[:20],
            "wall_seconds_including_serial": wall,
            "per_text_us": {
                k: sorted(r["us"][k] for r in results if r.get("ok"))
                for k in ("features", "model", "total")
            },
        },
        "classification_in_scope": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    raw_log = args.out.with_suffix(".log")
    raw_log.write_text("\n".join(boot.raw) + "\n", encoding="utf-8")
    b = boot.records.get("BENCH", {})
    print(f"SELFTEST {boot.records.get('SELFTEST')}")
    print(f"BENCH total p50={b.get('total', {}).get('p50')} p95={b.get('total', {}).get('p95')} us")
    print(f"query n={len(results)} mismatch={len(mismatch)} exact={report['exact_accuracy']}")
    print(f"wrote {args.out}")
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
