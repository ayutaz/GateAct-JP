"""LUT6 層を、等価な LUT5 層 + LUT3 層(Shannon 展開)に書き換える IR 変換。

理由: Griffith(commit e0675f4)の C 生成(``deploy/plan.py``)は ``n_inputs > 5`` の lut_layer を
「真理値表が 32bit に収まらない」として拒否する。一方、学習は LUT6(``n_inputs=6``)が最も
精度が高い(docs/research.md)。そこで学習済みの IR を次のように書き換える::

    f(x0..x5) = x5 ? f1(x0..x4) : f0(x0..x4)

- 層 A(LUT5、出力 3W): ``f0_o``(表の下位 32bit)、``f1_o``(上位 32bit)、``x5`` のパススルー
- 層 B(LUT3、出力 W): ``mux(f0_o, f1_o, s_o)``、表 ``0xCA``(addr = f0 | f1<<1 | s<<2)

論理的に厳密に等価(テストで IR simulate を照合)。増えたパススルー・定数ゲートは
Griffith の回路圧縮(sweep)で畳まれる。この変換は Griffith に手を入れずに行う。
"""

from __future__ import annotations

import copy
from typing import Any

_MASK32 = 0xFFFFFFFF
_PASS0_5 = 0xAAAAAAAA  # 5 入力 LUT で入力 0 をそのまま出す表
_MUX3 = 0xCA  # addr = f0 | f1<<1 | s<<2 → s ? f1 : f0


def has_wide_luts(ir: dict[str, Any]) -> bool:
    """``n_inputs > 5`` の lut_layer があるか。"""
    return any(n["type"] == "lut_layer" and n["params"]["n_inputs"] > 5 for n in ir["nodes"])


def _tt_int(v: Any) -> int:
    return int(v, 16) if isinstance(v, str) else int(v)


def _split_node(node: dict[str, Any]) -> list[dict[str, Any]]:
    p = node["params"]
    if p["n_inputs"] != 6:
        raise ValueError(f"only n_inputs=6 can be split (got {p['n_inputs']})")
    w = int(p["out_width"])
    idx = [list(map(int, row)) for row in p["indices"]]  # [6][W]
    tts = [_tt_int(v) for v in p["truth_tables"]]
    a_idx: list[list[int]] = [[] for _ in range(5)]
    a_tt: list[str] = []
    for half in (0, 1):  # f0, f1
        for o in range(w):
            for j in range(5):
                a_idx[j].append(idx[j][o])
            a_tt.append(hex((tts[o] >> (32 * half)) & _MASK32))
    for o in range(w):  # x5 のパススルー(5 入力すべて x5 を見る)
        for j in range(5):
            a_idx[j].append(idx[5][o])
        a_tt.append(hex(_PASS0_5))
    a = {
        **copy.deepcopy({k: v for k, v in node.items() if k != "params"}),
        "id": f"{node['id']}_a",
        "params": {"in_width": p["in_width"], "out_width": 3 * w, "n_inputs": 5,
                   "indices": a_idx, "truth_tables": a_tt},
    }  # fmt: skip
    b = {
        **copy.deepcopy({k: v for k, v in node.items() if k != "params"}),
        "id": f"{node['id']}_b",
        "params": {"in_width": 3 * w, "out_width": w, "n_inputs": 3,
                   "indices": [list(range(w)), list(range(w, 2 * w)), list(range(2 * w, 3 * w))],
                   "truth_tables": [hex(_MUX3)] * w},
    }  # fmt: skip
    for n in (a, b):
        n.setdefault("meta", {})
        n["meta"] = {**(n["meta"] or {}), "gateact_split_from": node["id"]}
    return [a, b]


def split_wide_luts(ir: dict[str, Any]) -> dict[str, Any]:
    """LUT6 の lut_layer をすべて LUT5 + LUT3 に分ける(該当が無ければ入力をそのまま返す)。

    直列のグラフ(edges が x → n0 → n1 → … → y)だけを扱う。
    """
    if not has_wide_luts(ir):
        return ir
    out = copy.deepcopy(ir)
    nodes: list[dict[str, Any]] = []
    rename: dict[str, tuple[str, str]] = {}
    for n in out["nodes"]:
        if n["type"] == "lut_layer" and n["params"]["n_inputs"] > 5:
            a, b = _split_node(n)
            nodes += [a, b]
            rename[n["id"]] = (a["id"], b["id"])
        else:
            nodes.append(n)
    edges: list[dict[str, Any]] = []
    for e in out["edges"]:
        src, dst = e["src"], e["dst"]
        if src in rename:
            src = rename[src][1]
        if dst in rename:
            a_id, b_id = rename[dst]
            edges.append({**e, "src": src, "dst": a_id})
            edges.append({**e, "src": a_id, "dst": b_id})
            continue
        edges.append({**e, "src": src, "dst": dst})
    out["nodes"] = nodes
    out["edges"] = edges
    return out
