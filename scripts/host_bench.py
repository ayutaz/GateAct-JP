"""ファームウェアをホスト(PC)向けにビルドし、起動時の SELFTEST / BENCH / MEM を記録する。

ESP32-S3 実機の値ではない(PC の参考値)。実機は ``scripts/esp32_bench.py --port``。

    uv run python scripts/host_bench.py runs/export/<name>/model --out results/host/<name>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from gateact.benchmarks.device import HostProcess, read_boot
from gateact.utils import git_sha

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "firmware" / "common"
FW = ROOT / "firmware" / "esp32s3" / "gateact" / "main"


def build(model_dir: Path, exe: Path, iters: int) -> list[str]:
    """``main.c`` をホスト向けにビルドする(LUT か baseline かは model_dir の中身で決める)。"""
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        raise SystemExit("C compiler not found")
    srcs = [FW / "main.c", COMMON / "gateact_features.c"]
    if (model_dir / "model.c").exists():
        srcs += [model_dir / "model.c", model_dir / "mdl_kernels.c"]
    else:
        srcs += [COMMON / "gateact_baselines.c"]
    cmd = [cc, "-std=c99", "-O2", "-DHOST_BUILD", "-DMDL_NO_DEFAULT_CTX", f"-DBENCH_ITERS={iters}",
           f"-I{COMMON}", f"-I{model_dir}", *map(str, srcs), "-o", str(exe)]  # fmt: skip
    subprocess.run(cmd, check=True)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=3, help="起動からの計測を繰り返す回数")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "gateact_host"
        cmd = build(args.model_dir.resolve(), exe, args.iters)
        runs = []
        for _ in range(args.repeat):
            t = HostProcess(exe)
            try:
                boot = read_boot(t, timeout=600)
            finally:
                t.close()
            if not boot.ready:
                raise SystemExit("READY not received:\n" + "\n".join(boot.raw[-20:]))
            runs.append(boot.records)
    out = {
        "environment": "host (NOT ESP32-S3)",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "compile": " ".join(Path(c).name if c.startswith("/") else c for c in cmd[:6]),
        "model_dir": str(args.model_dir),
        "git_sha": git_sha(),
        "iters": args.iters,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for r in runs:
        print(json.dumps(r.get("BENCH"), ensure_ascii=False))


if __name__ == "__main__":
    main()
