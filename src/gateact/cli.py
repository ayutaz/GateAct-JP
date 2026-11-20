"""``gateact`` コマンドラインの入口。サブコマンドは各モジュールの ``register`` で追加する。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from gateact import __version__


def build_parser() -> argparse.ArgumentParser:
    """全サブコマンドを登録した ArgumentParser を返す。"""
    parser = argparse.ArgumentParser(prog="gateact", description="GateAct-JP command line")
    parser.add_argument("--version", action="version", version=f"gateact {__version__}")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI を実行し、終了コードを返す。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args) or 0)
