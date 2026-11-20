import shutil

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """C コンパイラが無い環境では needs_cc テストを skip する。"""
    if shutil.which("cc") or shutil.which("clang") or shutil.which("gcc"):
        return
    skip = pytest.mark.skip(reason="C コンパイラ(cc/clang/gcc)が無い")
    for item in items:
        if "needs_cc" in item.keywords:
            item.add_marker(skip)
