"""ファームウェア(実機またはホストビルド)との対話と、ログの解析。

ファームウェアは ``TAG {json}`` 形式の行を出す(``firmware/esp32s3/gateact/main/main.c``)。
同じ ``Transport`` で ESP32-S3 のシリアルとホストの実行ファイルを扱えるので、解析と照合の
コードはホストでテストできる。
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Transport(Protocol):
    """行単位の双方向通信。"""

    def readline(self, timeout: float) -> str | None:
        """1行読む(改行を除く)。タイムアウトなら None。"""
        ...

    def writeline(self, data: bytes) -> None:
        """1行書く(改行は付けて送る)。"""
        ...

    def close(self) -> None:
        """閉じる。"""
        ...


def parse_line(line: str) -> tuple[str, dict[str, Any] | None]:
    """``TAG {json}`` を (TAG, dict) にする。JSON でない行は (行, None)。"""
    tag, _, body = line.strip().partition(" ")
    if body.startswith("{"):
        try:
            return tag, json.loads(body)
        except json.JSONDecodeError:
            pass
    return line.strip(), None


@dataclass
class BootLog:
    """起動時の INFO / SELFTEST / BENCH / MEM と生ログ。"""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)
    ready: bool = False


def read_boot(t: Transport, *, timeout: float = 120.0) -> BootLog:
    """READY まで読み、タグごとの最後の JSON を集める。"""
    log = BootLog()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = t.readline(timeout=max(0.1, deadline - time.monotonic()))
        if line is None:
            continue
        log.raw.append(line)
        tag, body = parse_line(line)
        if body is not None:
            log.records[tag] = body
        elif tag == "READY":
            log.ready = True
            break
    return log


def query(t: Transport, texts: Iterable[str], *, timeout: float = 10.0) -> list[dict[str, Any]]:
    """各テキストを送り、RESULT の JSON を順に返す(他の行は読み飛ばす)。"""
    out: list[dict[str, Any]] = []
    for text in texts:
        t.writeline(text.encode("utf-8"))
        deadline = time.monotonic() + timeout
        while True:
            line = t.readline(timeout=max(0.1, deadline - time.monotonic()))
            if line is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no RESULT for {text!r}")
                continue
            tag, body = parse_line(line)
            if tag == "RESULT" and body is not None:
                out.append(body)
                break
    return out


class HostProcess:
    """ホストビルドのファームウェアを子プロセスとして動かす Transport。"""

    def __init__(self, exe: str | Path) -> None:
        """実行ファイルを起動する。"""
        self.proc = subprocess.Popen(
            [str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )

    def readline(self, timeout: float) -> str | None:
        """1行読む(子プロセスの出力はブロッキングで読む)。"""
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        return line.decode("utf-8", "replace").rstrip("\r\n") if line else None

    def writeline(self, data: bytes) -> None:
        """1行送る。"""
        assert self.proc.stdin is not None
        self.proc.stdin.write(data + b"\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        """終了させる。"""
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


class SerialPort:
    """ESP32-S3 のシリアル(USB-Serial/JTAG または UART)。pyserial が必要。"""

    def __init__(self, port: str, baud: int = 115200, *, reset: bool = True) -> None:
        """ポートを開き、必要ならリセットして起動ログを最初から読めるようにする。"""
        import serial

        self.ser = serial.Serial(port, baud, timeout=0.2)
        self._buf = b""
        if reset:
            # esptool の hard reset と同じ: RTS で EN を下げて上げる(IO0 は High のまま)
            self.ser.dtr = False
            self.ser.rts = True
            time.sleep(0.2)
            self.ser.rts = False
        self.ser.reset_input_buffer()

    def readline(self, timeout: float) -> str | None:
        """1行読む。"""
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            chunk = self.ser.read(4096)
            if chunk:
                self._buf += chunk
            elif time.monotonic() > deadline:
                return None
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode("utf-8", "replace").rstrip("\r")

    def writeline(self, data: bytes) -> None:
        """1行送る。"""
        self.ser.write(data + b"\n")
        self.ser.flush()

    def close(self) -> None:
        """閉じる。"""
        self.ser.close()


def compare_results(
    results: Sequence[dict[str, Any]], expected_scores: Sequence[Sequence[int]]
) -> dict[str, int]:
    """デバイスの RESULT と期待スコア(IR simulate)を照合する。"""
    mismatch = sum(
        1
        for r, e in zip(results, expected_scores, strict=True)
        if not r.get("ok") or list(r["scores"]) != list(e)
    )
    return {"n": len(results), "score_mismatch": mismatch}
