"""ベースライン A: キーワードによるルール分類器(GateAct-JP の16クラス向けに新規に書いたもの)。

方針は JapaneseTinyAgentLM ``jtalm.eval.rule_baseline`` の文書と同じ(否定の句は落とす、
「A ではなく B」は B を採る)が、規則とクラスはこのプロジェクトのもの。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from gateact.data.checks import Labeled
from gateact.data.labels import LabelSpec
from gateact.features.normalize import normalize_text

NEGATION = (
    r"ないで|なくていい|なくて(?:も)?(?:いい|良い|大丈夫)|しなくて|やめて|止めて"
    r"|だめ|ダメ|するな|いらない|不要|ません"
)
CORRECTION = r"ではなく|じゃなくて|じゃなく|でなく|じゃないよ|ではないよ"

# (クラス, パターン) — 上から順に評価する
RULES: list[tuple[str, str]] = [
    ("led_off", r"(?:ライト|led|ランプ|明かり|電気|光).*(?:消|オフ|off)|(?:消灯)"),
    ("led_on", r"(?:ライト|led|ランプ|光|点灯|ひか|光っ)"),
    ("volume_up", r"(?:音量|音|ボリューム|声).*(?:上げ|大きく|アップ|あげ)"),
    ("volume_down", r"(?:音量|音|ボリューム|声).*(?:下げ|小さく|ダウン|さげ|静か)"),
    ("nod", r"うなず|頷|首を縦|こくり|こくこく"),
    ("shake", r"首を横|横に振|首を振|いやいや|ぶんぶん"),
    ("bow", r"お辞儀|おじぎ|お礼|礼を|頭を下げ"),
    ("expr_happy", r"笑|にこ|ニコ|嬉|うれし|楽し|喜|スマイル"),
    ("expr_surprised", r"驚|びっくり|ビックリ|おどろ|仰天"),
    ("expr_sad", r"悲し|かなし|泣|しょんぼり|寂し|さみし|落ち込"),
]
LOOK_VERB = r"見|向|顔|視線|目|振り返|むい|みて"
LOOK_DIR: list[tuple[str, str]] = [
    ("look_center", r"正面|まっすぐ|真っ直ぐ|真ん中|前を"),
    ("look_up", r"上|見上げ|うえ"),
    ("look_down", r"下|うつむ|見下ろ|した[をに]"),
    ("look_left", r"左|ひだり"),
    ("look_right", r"右|みぎ"),
]


class RuleClassifier:
    """キーワードルールの分類器(学習しない)。"""

    name = "rules"

    def __init__(self, spec: LabelSpec) -> None:
        """クラス定義を受け取る(ルールのクラス名が定義にあることを確認する)。"""
        self.spec = spec
        for label, _ in RULES + LOOK_DIR:
            spec.index(label)

    def fit(self, train: Sequence[Labeled], dev: Sequence[Labeled]) -> dict[str, Any]:
        """学習しない。"""
        return {}

    def _one(self, text: str) -> str:
        t = normalize_text(text)
        m = re.search(CORRECTION, t)
        if m:
            t = t[m.end() :]
        if re.search(NEGATION, t):
            return self.spec.no_action
        for label, pat in RULES:
            if re.search(pat, t):
                return label
        if re.search(LOOK_VERB, t):
            for label, pat in LOOK_DIR:
                if re.search(pat, t):
                    return label
        return self.spec.no_action

    def predict(self, texts: Sequence[str]) -> list[str]:
        """クラス名を予測する。"""
        return [self._one(t) for t in texts]

    def describe(self) -> dict[str, Any]:
        """規則の数。"""
        return {"model": "keyword rules", "rules": len(RULES) + len(LOOK_DIR)}
