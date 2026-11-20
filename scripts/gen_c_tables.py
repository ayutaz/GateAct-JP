"""Python の正規化表から firmware/common/gateact_norm_tables.h を生成する。

uv run python scripts/gen_c_tables.py           # 書き出し
uv run python scripts/gen_c_tables.py --check   # 生成結果と一致しなければ exit 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gateact.features.normalize import HALFWIDTH_KANA_MAP, KANA_COMPOSE, NORM_VERSION

OUT = Path(__file__).resolve().parents[1] / "firmware" / "common" / "gateact_norm_tables.h"


def render() -> str:
    hw = ",\n".join(
        "  " + ", ".join(f"0x{v:04X}" for v in HALFWIDTH_KANA_MAP[i : i + 8])
        for i in range(0, len(HALFWIDTH_KANA_MAP), 8)
    )
    comp = ",\n".join(f"  {{0x{a:04X}, 0x{b:04X}, 0x{c:04X}}}" for a, b, c in KANA_COMPOSE)
    return f"""/* 自動生成 — scripts/gen_c_tables.py({NORM_VERSION})。編集しない。
 * 元: src/gateact/features/normalize.py
 * SPDX-License-Identifier: Apache-2.0 */
#ifndef GATEACT_NORM_TABLES_H
#define GATEACT_NORM_TABLES_H
#include <stdint.h>

#define GFEAT_NORM_VERSION "{NORM_VERSION}"

/* U+FF61..U+FF9F -> NFKC */
static const uint16_t GFEAT_HALFWIDTH_KANA[{len(HALFWIDTH_KANA_MAP)}] = {{
{hw}
}};

/* (base, combining mark, composed), base 昇順 */
#define GFEAT_N_COMPOSE {len(KANA_COMPOSE)}
static const uint16_t GFEAT_COMPOSE[GFEAT_N_COMPOSE][3] = {{
{comp}
}};

#endif /* GATEACT_NORM_TABLES_H */
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} is stale; run scripts/gen_c_tables.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
