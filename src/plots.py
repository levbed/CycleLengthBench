from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any


WHITE = (255, 255, 255)
INK = (30, 34, 39)
GRID = (222, 226, 230)
BLUE = (31, 119, 180)
TEAL = (44, 160, 145)
ORANGE = (230, 126, 34)
RED = (214, 80, 80)


FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "010", "010"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    ".": ["000", "000", "000", "000", "000", "110", "110"],
    "+": ["000", "010", "010", "111", "010", "010", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    "%": ["101", "001", "010", "010", "010", "100", "101"],
}

LETTERS = {
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["011", "100", "100", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["011", "100", "100", "101", "101", "101", "011"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "001", "101", "101", "010"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101", "101", "101"],
    "N": ["101", "111", "111", "111", "101", "101", "101"],
    "O": ["010", "101", "101", "101", "101", "101", "010"],
    "P": ["110", "101", "101", "110", "100", "100", "100"],
    "Q": ["010", "101", "101", "101", "111", "011", "001"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "S": ["011", "100", "100", "010", "001", "001", "110"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["101", "101", "101", "101", "111", "111", "101"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
}
FONT.update(LETTERS)


def _blank(width: int, height: int, color=WHITE) -> list[list[tuple[int, int, int]]]:
    return [[color for _ in range(width)] for _ in range(height)]


def _set(canvas: list[list[tuple[int, int, int]]], x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
        canvas[y][x] = color


def _rect(canvas, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, y0), min(len(canvas), y1)):
        for x in range(max(0, x0), min(len(canvas[0]), x1)):
            canvas[y][x] = color


def _line(canvas, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set(canvas, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _circle(canvas, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                _set(canvas, x, y, color)


def _text(canvas, x: int, y: int, text: str, color: tuple[int, int, int] = INK, scale: int = 2) -> None:
    cursor = x
    for char in text.upper():
        pattern = FONT.get(char, FONT[" "])
        for row_idx, row in enumerate(pattern):
            for col_idx, pixel in enumerate(row):
                if pixel == "1":
                    _rect(
                        canvas,
                        cursor + col_idx * scale,
                        y + row_idx * scale,
                        cursor + (col_idx + 1) * scale,
                        y + (row_idx + 1) * scale,
                        color,
                    )
        cursor += (len(pattern[0]) + 1) * scale


def _write_png(path: str | Path, canvas: list[list[tuple[int, int, int]]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(canvas)
    width = len(canvas[0])
    raw = b"".join(b"\x00" + b"".join(bytes(pixel) for pixel in row) for row in canvas)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def save_mae_by_track(path: str | Path, scores: list[dict[str, Any]]) -> None:
    width, height = 900, 520
    canvas = _blank(width, height)
    left, right, top, bottom = 80, 40, 60, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_mae = max(float(row["mae"]) for row in scores) if scores else 1.0
    max_mae = max(1.0, math.ceil(max_mae))
    _text(canvas, 80, 24, "MAE BY TRACK", INK, 2)
    _line(canvas, left, top, left, top + plot_h, INK)
    _line(canvas, left, top + plot_h, left + plot_w, top + plot_h, INK)
    for i in range(6):
        y = top + plot_h - int(plot_h * i / 5)
        _line(canvas, left, y, left + plot_w, y, GRID)
        _text(canvas, 18, y - 7, f"{max_mae * i / 5:.1f}", INK, 1)
    labels = {
        "global_median": "MEDIAN",
        "previous_cycle": "PREV",
        "history_only": "HIST",
        "history_plus_wearables": "HIST+W",
        "history_plus_hormones": "HIST+H",
        "full_multimodal": "FULL",
    }
    colors = [BLUE, TEAL, ORANGE, RED, (125, 95, 180), (70, 145, 90)]
    bar_gap = 18
    bar_w = max(20, int((plot_w - bar_gap * (len(scores) + 1)) / max(1, len(scores))))
    for idx, row in enumerate(scores):
        mae = float(row["mae"])
        x0 = left + bar_gap + idx * (bar_w + bar_gap)
        y0 = top + plot_h - int(plot_h * mae / max_mae)
        _rect(canvas, x0, y0, x0 + bar_w, top + plot_h, colors[idx % len(colors)])
        _text(canvas, x0, top + plot_h + 15, labels.get(row["track"], row["track"])[:8], INK, 1)
        _text(canvas, x0, y0 - 18, f"{mae:.1f}", INK, 1)
    _write_png(path, canvas)


def save_predicted_vs_observed(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    selected = [row for row in predictions if row["track"] == "full_multimodal"]
    if not selected:
        selected = predictions
    values = [
        (float(row["observed_cycle_length"]), float(row["predicted_cycle_length"]))
        for row in selected
        if not math.isnan(float(row["predicted_cycle_length"]))
    ]
    width, height = 700, 620
    canvas = _blank(width, height)
    left, right, top, bottom = 80, 50, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    if values:
        min_value = math.floor(min(min(obs, pred) for obs, pred in values) - 2)
        max_value = math.ceil(max(max(obs, pred) for obs, pred in values) + 2)
    else:
        min_value, max_value = 10, 40
    span = max(1, max_value - min_value)
    _text(canvas, 80, 24, "PREDICTED VS OBSERVED", INK, 2)
    _line(canvas, left, top, left, top + plot_h, INK)
    _line(canvas, left, top + plot_h, left + plot_w, top + plot_h, INK)
    for i in range(6):
        x = left + int(plot_w * i / 5)
        y = top + plot_h - int(plot_h * i / 5)
        _line(canvas, x, top, x, top + plot_h, GRID)
        _line(canvas, left, y, left + plot_w, y, GRID)
    _line(canvas, left, top + plot_h, left + plot_w, top, RED)
    for obs, pred in values:
        x = left + int(plot_w * (obs - min_value) / span)
        y = top + plot_h - int(plot_h * (pred - min_value) / span)
        _circle(canvas, x, y, 3, BLUE)
    _text(canvas, 245, height - 35, "OBSERVED DAYS", INK, 2)
    _text(canvas, 8, 24, "PRED", INK, 1)
    _write_png(path, canvas)
