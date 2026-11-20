"""データの検査(重複・矛盾・分割間の漏れ)、学習/開発/評価の分割、manifest の作成。"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gateact.data.io import Example
from gateact.data.labels import LabelResult, LabelSpec
from gateact.data.schema import canonical_json
from gateact.features.normalize import normalize_text
from gateact.utils import sha256_file

DEV_SALT = "gateact-dev-v1"


def _key(text: str) -> str:
    return normalize_text(text)


def find_duplicates(rows: Sequence[Example]) -> list[list[str]]:
    """正規化後のテキストが同じサンプルの ID のグループ(2件以上)を返す。"""
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        groups[_key(r.text)].append(r.id)
    return [ids for ids in groups.values() if len(ids) > 1]


def conflicting_duplicates(rows: Sequence[Example]) -> list[list[str]]:
    """同じテキストなのに正解の call 列が違うグループを返す。"""
    groups: dict[str, list[Example]] = defaultdict(list)
    for r in rows:
        groups[_key(r.text)].append(r)
    return [
        [r.id for r in g] for g in groups.values() if len({canonical_json(r.calls) for r in g}) > 1
    ]


def split_overlap(a: Sequence[Example], b: Sequence[Example]) -> set[str]:
    """2つの分割の両方に現れる正規化テキストの集合。"""
    return {_key(r.text) for r in a} & {_key(r.text) for r in b}


def _unit(text: str, salt: str) -> float:
    h = hashlib.sha256((salt + "\x00" + _key(text)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def dev_split(
    rows: Sequence[Example], *, frac: float, salt: str = DEV_SALT
) -> tuple[list[Example], list[Example]]:
    """テキストのハッシュで決定的に開発用の分割を切り出す(同じテキストは同じ側)。"""
    train: list[Example] = []
    dev: list[Example] = []
    for r in rows:
        (dev if _unit(r.text, salt) < frac else train).append(r)
    return train, dev


@dataclass(frozen=True)
class Labeled:
    """ラベル付きのサンプル。"""

    example: Example
    result: LabelResult

    @property
    def label(self) -> str:
        """クラス名(対象内のサンプルだけを持つので None にならない)。"""
        assert self.result.label is not None
        return self.result.label


@dataclass
class Splits:
    """学習 / 開発 / 評価の分割と、除外したサンプルの集計。"""

    train: list[Labeled]
    dev: list[Labeled]
    test: list[Labeled]
    test_out_of_scope: list[tuple[Example, LabelResult]]
    removed_leakage: int
    out_of_scope: dict[str, dict[str, int]] = field(default_factory=dict)


def _label(
    rows: Sequence[Example], spec: LabelSpec
) -> tuple[list[Labeled], list[tuple[Example, LabelResult]]]:
    keep: list[Labeled] = []
    drop: list[tuple[Example, LabelResult]] = []
    for r in rows:
        res = spec.assign(r.calls)
        (keep.append(Labeled(r, res)) if res.in_scope else drop.append((r, res)))
    return keep, drop


def prepare_splits(
    train_rows: Sequence[Example],
    test_rows: Sequence[Example],
    spec: LabelSpec,
    *,
    dev_frac: float = 0.05,
    salt: str = DEV_SALT,
) -> Splits:
    """分割を作る。

    - 評価(``test_rows`` = HF の validation)と同じ正規化テキストを学習側から除く(漏れ防止)。
    - 学習側から ``dev_frac`` を開発用に切り出す(epoch 選択・ハイパーパラメータ用)。
    - 対象外(複数 Action など)は分類から除き、理由ごとに数える。評価側の対象外は
      ``test_out_of_scope`` に残す(別集計用)。
    """
    test_keys = {_key(r.text) for r in test_rows}
    clean = [r for r in train_rows if _key(r.text) not in test_keys]
    tr, dv = dev_split(clean, frac=dev_frac, salt=salt)
    splits_rows = {"train": tr, "dev": dv, "test": list(test_rows)}
    labeled: dict[str, list[Labeled]] = {}
    oos: dict[str, dict[str, int]] = {}
    test_oos: list[tuple[Example, LabelResult]] = []
    for name, rows in splits_rows.items():
        keep, drop = _label(rows, spec)
        labeled[name] = keep
        oos[name] = dict(Counter(str(res.reason) for _, res in drop))
        if name == "test":
            test_oos = drop
    return Splits(
        train=labeled["train"],
        dev=labeled["dev"],
        test=labeled["test"],
        test_out_of_scope=test_oos,
        removed_leakage=len(train_rows) - len(clean),
        out_of_scope=oos,
    )


def split_summary(rows: Sequence[Example], spec: LabelSpec) -> dict[str, Any]:
    """1分割の件数・カテゴリ・クラス分布・対象外の理由・カバー率・重複を集計する。"""
    labels: Counter[str] = Counter()
    oos: Counter[str] = Counter()
    arg_loss = 0
    for r in rows:
        res = spec.assign(r.calls)
        if res.in_scope:
            labels[str(res.label)] += 1
            arg_loss += int(res.arg_loss)
        else:
            oos[str(res.reason)] += 1
    n_in = sum(labels.values())
    dups = find_duplicates(rows)
    return {
        "n": len(rows),
        "categories": dict(sorted(Counter(r.category for r in rows).items())),
        "labels": {name: labels.get(name, 0) for name in spec.names},
        "out_of_scope": dict(sorted(oos.items())),
        "coverage": n_in / len(rows) if rows else 0.0,
        "arg_loss": arg_loss,
        "arg_loss_rate": arg_loss / n_in if n_in else 0.0,
        "duplicate_groups": len(dups),
        "duplicate_rows": sum(len(g) for g in dups),
        "conflicting_duplicate_groups": len(conflicting_duplicates(rows)),
    }


def build_manifest(
    files: Mapping[str, str | Path],
    rows: Mapping[str, Sequence[Example]],
    spec: LabelSpec,
    *,
    revision: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """再現用の manifest(ファイルの SHA-256、分割ごとの集計、分割間の重なり)を作る。"""
    splits: dict[str, Any] = {}
    for name, path in files.items():
        splits[name] = {"file": Path(path).name, "sha256": sha256_file(path)}
        splits[name].update(split_summary(rows[name], spec))
    names = list(rows)
    overlap = {
        f"{a}&{b}": len(split_overlap(rows[a], rows[b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    return {
        "source": source,
        "revision": revision,
        "label_spec": spec.version,
        "splits": splits,
        "overlap_normalized_text": overlap,
    }
