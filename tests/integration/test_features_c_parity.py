"""特徴量の C99 実装(firmware/common/gateact_features.c)が Python とビット単位で一致すること。"""

import random
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gateact.data.io import load_jsonl
from gateact.features.bits import pack_bits
from gateact.features.hashing import MAX_CODEPOINTS, FeatureConfig, extract_bits
from gateact.features.normalize import HALFWIDTH_KANA

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "firmware" / "common"
DATA_VAL = ROOT / "data" / "raw" / "action_synth" / "validation.jsonl"

pytestmark = pytest.mark.needs_cc

EDGE = [
    "", " ", "　", "右を見て", "右を見ないで", "ＡＢＣ　ｄｅｆ", "ｶﾞｷﾞｸﾞﾊﾟ", "ｳﾞｧｲｵﾘﾝ", "が",
    "ﾞ", "ﾞ先頭の濁点", "😀👍🏽", "1回うなずいて", "音量を５０％に", "  前後  空白  ", "a\tb\nc",
    "Ｈｅｌｌｏ　ＷＯＲＬＤ", "ーｰ～〜", "ヾゞ", "あ" * MAX_CODEPOINTS,
]  # fmt: skip
POOLS = [
    [chr(c) for c in range(0x3041, 0x3097)],
    [chr(c) for c in range(0x30A1, 0x30FB)],
    list(HALFWIDTH_KANA),
    [chr(c) for c in range(0xFF01, 0xFF5F)],
    [chr(c) for c in range(0x20, 0x7F)],
    list("右左上下見向首顔音量明光消点笑泣驚礼"),
    [" ", "\t", "　", "゙", "゚", "😀", "é", "​"],
]


def _random_texts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(0, 40)
        out.append("".join(rng.choice(rng.choice(POOLS)) for _ in range(k)))
    return out


def _cc() -> str:
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc
    return cc


@pytest.fixture(scope="module")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    exe = tmp_path_factory.mktemp("feat") / "feat_driver"
    subprocess.run(
        [_cc(), "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", f"-I{COMMON}",
         str(ROOT / "tests" / "c" / "feat_driver.c"), str(COMMON / "gateact_features.c"),
         "-o", str(exe)],
        check=True,
    )  # fmt: skip
    return exe


def _run_c(exe: Path, raw: list[bytes], cfg: FeatureConfig) -> tuple[list[int], np.ndarray]:
    mask = sum(1 << (n - 1) for n in cfg.ngram_orders)
    payload = struct.pack("<III", cfg.width, mask, cfg.num_hashes)
    payload += b"".join(struct.pack("<I", len(r)) + r for r in raw)
    out = subprocess.run([str(exe)], input=payload, capture_output=True, check=True).stdout
    nbytes = (cfg.width + 7) // 8
    rec = 4 + nbytes
    assert len(out) == rec * len(raw)
    status = [struct.unpack_from("<i", out, i * rec)[0] for i in range(len(raw))]
    packed = np.frombuffer(out, dtype=np.uint8).reshape(len(raw), rec)[:, 4:]
    return status, packed


CONFIGS = [
    FeatureConfig(width=512),
    FeatureConfig(width=256, ngram_orders=(2, 3)),
    FeatureConfig(width=1000, ngram_orders=(1, 2, 3, 4), num_hashes=2),
]


@pytest.mark.parametrize(
    "cfg", CONFIGS, ids=lambda c: f"w{c.width}-n{len(c.ngram_orders)}-h{c.num_hashes}"
)
def test_c_bits_equal_python(driver: Path, cfg: FeatureConfig) -> None:
    texts = EDGE + [e.text for e in load_jsonl(ROOT / "tests/fixtures/action_mini.jsonl")]
    texts += _random_texts(500)
    status, packed = _run_c(driver, [t.encode("utf-8") for t in texts], cfg)
    assert status == [0] * len(texts)
    assert np.array_equal(packed, pack_bits(extract_bits(texts, cfg)))


def test_c_rejects_too_long_like_python(driver: Path) -> None:
    status, _ = _run_c(driver, [("あ" * (MAX_CODEPOINTS + 1)).encode()], CONFIGS[0])
    assert status == [-2]


@pytest.mark.parametrize(
    "raw",
    [b"\xff", b"\xc0\xaf", b"\xe3\x81", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"ok\x80"],
    ids=["ff", "overlong", "truncated", "surrogate", "above-max", "stray-cont"],
)
def test_c_rejects_invalid_utf8_like_python_strict_decoder(driver: Path, raw: bytes) -> None:
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    status, _ = _run_c(driver, [raw], CONFIGS[0])
    assert status == [-1]


@pytest.mark.skipif(not DATA_VAL.exists(), reason="HF データ未取得(make data)")
def test_c_bits_equal_python_on_validation_set(driver: Path) -> None:
    texts = [e.text for e in load_jsonl(DATA_VAL)]
    cfg = CONFIGS[0]
    status, packed = _run_c(driver, [t.encode("utf-8") for t in texts], cfg)
    assert status == [0] * len(texts)
    assert np.array_equal(packed, pack_bits(extract_bits(texts, cfg)))
