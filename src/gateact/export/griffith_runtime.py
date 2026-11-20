"""Griffith の C99 ランタイム(``runtime/c``)を固定 SHA から取得して使えるようにする。

背景(パッチの範囲と理由):
    griffith-logic v0.1.0(commit ``e0675f4``)の wheel には ``runtime/c/`` が含まれない。
    一方 ``griffith_logic.deploy.verify.RUNTIME_C`` はパッケージの2階層上(=リポジトリの
    ルート)を指すため、pip/uv でインストールした環境では ``build_c_model`` と
    ``griffith deploy codegen-c`` のランタイムコピーが失敗する。

    GateAct-JP は Griffith のソースを同梱せず、固定 commit のソースアーカイブから
    ``runtime/c`` だけを取得してユーザーキャッシュに置き、SHA-256 を検証したうえで
    ``RUNTIME_C`` をそのパスに差し替える(Griffith のコードは変更しない)。
    上流で runtime が wheel に入れば、このモジュールは不要になる。
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import urllib.request
from pathlib import Path

GRIFFITH_SHA = "e0675f4e36596954aa424d522e0ee966bba9a3e1"
ARCHIVE_URL = f"https://codeload.github.com/tsukushi-gate/Griffith/tar.gz/{GRIFFITH_SHA}"

#: runtime/c 以下の相対パス → SHA-256(commit e0675f4 で計測)
RUNTIME_FILES: dict[str, str] = {
    "include/mdl_kernels.h": "b1a4785f884d4954cb8ef1cdcb88bc5327dfc4f16a0cfb6b48cf69648fcda94e",
    "include/mdl_simd.h": "3b93cb9127459a144a0f12afc7d8a404ab24e80b28d369173baa0540a1e53eae",
    "include/mdl_stdio.h": "26c3d86d72f9ac476c8431d97a486f0bf9aec62d872ffe821fe9e9e91e1ac3bf",
    "src/mdl_kernels.c": "32415b1388d91b837ca008ab5e72cbd5150e404c421675ae073d0c1e8c3061a7",
}


class RuntimeVerificationError(RuntimeError):
    """ランタイムのファイルが無い、または SHA-256 が固定値と一致しない。"""


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "gateact" / "griffith-runtime" / GRIFFITH_SHA / "c"


def _verify(root: Path) -> None:
    for rel, expected in RUNTIME_FILES.items():
        path = root / rel
        if not path.is_file():
            raise RuntimeVerificationError(f"missing Griffith runtime file: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeVerificationError(
                f"SHA-256 mismatch for {path}: {actual} != {expected} (pinned {GRIFFITH_SHA})"
            )


def _download(dest: Path) -> None:
    with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as resp:  # noqa: S310 (固定URL)
        data = resp.read()
    prefix = f"Griffith-{GRIFFITH_SHA}/runtime/c/"
    tmp = dest.with_name(dest.name + ".tmp")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for rel in RUNTIME_FILES:
            member = tar.getmember(prefix + rel)
            f = tar.extractfile(member)
            if f is None:
                raise RuntimeVerificationError(f"not a file in archive: {member.name}")
            out = tmp / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(f.read())
    _verify(tmp)
    tmp.replace(dest)


def runtime_dir() -> Path:
    """検証済みの Griffith C ランタイムのディレクトリ(``include/`` と ``src/`` を持つ)を返す。

    ``GATEACT_GRIFFITH_RUNTIME`` 環境変数があればそれを使う(オフライン環境用)。
    無ければユーザーキャッシュを使い、未取得なら固定 SHA のアーカイブから取得する。

    Raises:
        RuntimeVerificationError: ファイル欠落または SHA-256 不一致。
    """
    override = os.environ.get("GATEACT_GRIFFITH_RUNTIME")
    if override:
        root = Path(override)
        _verify(root)
        return root
    root = _cache_root()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _download(root)
    _verify(root)
    return root


def patch_griffith_runtime() -> Path:
    """``griffith_logic.deploy.verify.RUNTIME_C`` を検証済みランタイムに差し替える。"""
    import griffith_logic.deploy.verify as verify

    root = runtime_dir()
    verify.RUNTIME_C = root
    return root
