import unicodedata

import numpy as np
import pytest

from gateact.features.hashing import FeatureConfig, extract_bits, fmix32, fnv1a32, ngrams
from gateact.features.normalize import HALFWIDTH_KANA, normalize_text

# --- 正規化 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("ＡＢＣ１２３！", "abc123!"),
        ("Hello World", "hello world"),
        ("ｱｲｳｴｵ", "アイウエオ"),
        ("ｶﾞｷﾞﾊﾟﾋﾟｳﾞ", "ガギパピヴ"),
        ("ひらがなカタカナ漢字", "ひらがなカタカナ漢字"),
        ("右を　見て", "右を 見て"),
        ("  a \t b\n", "a b"),
        ("", ""),
        ("右を見ないで", "右を見ないで"),
        ("😀絵文字", "😀絵文字"),
    ],
)
def test_normalize_examples(src: str, dst: str) -> None:
    assert normalize_text(src) == dst


def test_fullwidth_ascii_matches_nfkc_lower() -> None:
    for cp in range(0xFF01, 0xFF5F):
        c = chr(cp)
        assert normalize_text(c) == unicodedata.normalize("NFKC", c).lower(), hex(cp)


def test_halfwidth_kana_matches_nfkc_including_voicing_marks() -> None:
    for c in HALFWIDTH_KANA:
        assert normalize_text(c) == unicodedata.normalize("NFKC", c), hex(ord(c))
        for mark in ("ﾞ", "ﾟ"):
            s = c + mark
            assert normalize_text(s) == unicodedata.normalize("NFKC", s), repr(s)


def test_normalize_is_idempotent() -> None:
    for s in ["ｶﾞｷﾞ ＡＢ", "右を見て", "  x  y "]:
        assert normalize_text(normalize_text(s)) == normalize_text(s)


# --- ハッシュ ---------------------------------------------------------------------


def test_fnv1a32_known_vectors() -> None:
    # FNV-1a 32bit の公開テストベクタ
    assert fnv1a32(b"") == 0x811C9DC5
    assert fnv1a32(b"a") == 0xE40C292C
    assert fnv1a32(b"foobar") == 0xBF9CF968


def test_fmix32_known_vectors() -> None:
    # MurmurHash3 fmix32
    assert fmix32(0) == 0
    assert fmix32(1) == 0x514E28B7


def test_ngrams_include_boundary_marks() -> None:
    grams = ngrams("ab", (1, 2, 3))
    assert grams[:4] == ["\x02", "a", "b", "\x03"]
    assert "\x02a" in grams and "b\x03" in grams and "\x02ab" in grams and "ab\x03" in grams
    assert len(grams) == 4 + 3 + 2


def test_ngrams_of_empty_text() -> None:
    assert ngrams("", (1, 2, 3)) == ["\x02", "\x03", "\x02\x03"]


# --- ビット特徴量 ---------------------------------------------------------------


CFG = FeatureConfig(width=512)


def test_extract_bits_shape_dtype_and_binary() -> None:
    bits = extract_bits(["右を見て", ""], CFG)
    assert bits.shape == (2, 512)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)) <= {0, 1}
    assert bits[0].sum() > 0 and bits[1].sum() > 0


def test_extract_bits_is_deterministic_and_order_fixed() -> None:
    a = extract_bits(["左を向いて"], CFG)
    b = extract_bits(["左を向いて"], CFG)
    assert np.array_equal(a, b)
    # ビット位置の仕様: idx = fmix32(fnv1a32(utf8(gram))) % width
    idx = fmix32(fnv1a32("左".encode())) % 512
    assert a[0, idx] == 1


def test_normalization_applied_before_hashing() -> None:
    assert np.array_equal(extract_bits(["ＡＢＣ"], CFG), extract_bits(["abc"], CFG))


def test_negation_changes_features() -> None:
    a, b = extract_bits(["右を見て", "右を見ないで"], CFG)
    assert not np.array_equal(a, b)


def test_multiple_hashes_set_more_bits() -> None:
    one = extract_bits(["うなずいて"], FeatureConfig(width=1024, num_hashes=1)).sum()
    two = extract_bits(["うなずいて"], FeatureConfig(width=1024, num_hashes=2)).sum()
    assert two > one


def test_config_roundtrip() -> None:
    cfg = FeatureConfig(width=256, ngram_orders=(2, 3), num_hashes=2)
    assert FeatureConfig.from_dict(cfg.to_dict()) == cfg


def test_too_long_text_is_rejected() -> None:
    from gateact.features.hashing import MAX_CODEPOINTS

    extract_bits(["あ" * MAX_CODEPOINTS], CFG)
    with pytest.raises(ValueError, match="too long"):
        extract_bits(["あ" * (MAX_CODEPOINTS + 1)], CFG)
    # 末尾の空白は除かれるので上限に数えない
    extract_bits(["あ" * MAX_CODEPOINTS + "   "], CFG)


def test_c_tables_are_up_to_date() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, str(root / "scripts" / "gen_c_tables.py"), "--check"])
    assert r.returncode == 0
