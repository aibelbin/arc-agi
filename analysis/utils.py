"""Utility helpers for visualization and pretty-printing ARC grids.

Provides zero-dependency (numpy optional) ANSI color rendering so examples can
be inspected quickly in a terminal / VS Code integrated console.

Public functions:
  color_block(digit: int, show_digit: bool = False) -> str
  grid_to_lines(grid, show_digit: bool = False) -> list[str]
  print_grid(grid, show_digit: bool = False, title: str | None = None) -> None
  print_side_by_side(grids: list, titles: list[str] | None = None, show_digit: bool = False) -> None

All functions gracefully degrade to plain text if ANSI colors are not
supported (very old terminals). Modern Windows 10+ terminals and VS Code
support ANSI escape sequences by default.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:
	import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy always present in this project
	_np = None  # fallback for type checkers

# ARC canonical palette (approximate) mapped to 256-color indices.
# 0-9 digits correspond to background colors; adjust if you prefer different hues.
_ARC_ANSI_BG = {
	0: 16,   # black
	1: 19,   # blue
	2: 160,  # red
	3: 34,   # green
	4: 226,  # yellow
	5: 250,  # gray
	6: 201,  # magenta/pink
	7: 208,  # orange
	8: 45,   # cyan/sky
	9: 88,   # maroon/brown
}

_RESET = "\x1b[0m"


def _supports_color() -> bool:
	# VS Code / modern terminals: assume yes; could refine detection if needed.
	import os, sys
	if os.environ.get("NO_COLOR") is not None:
		return False
	if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
		return False
	return True


_USE_COLOR = _supports_color()


def color_block(digit: int, show_digit: bool = False) -> str:
	"""Return a colored two-character block (or digit) representing a cell.

	If show_digit is True, the digit is printed centered on a padded block.
	"""
	d = int(digit)
	char = str(d) if show_digit else "  "
	if not _USE_COLOR:
		return char if show_digit else f"{d:2d}"  # plain fallback
	code = _ARC_ANSI_BG.get(d, 15)  # default bright white if unknown
	if show_digit:
		return f"\x1b[48;5;{code}m{d:^2}{_RESET}"
	return f"\x1b[48;5;{code}m  {_RESET}"


def _iter_rows(grid) -> Iterable[Sequence[int]]:
	if _np is not None and isinstance(grid, _np.ndarray):
		for r in grid.tolist():
			yield r
	else:
		# assume nested list-like
		for r in grid:
			yield r


def grid_to_lines(grid, show_digit: bool = False) -> List[str]:
	lines: List[str] = []
	for row in _iter_rows(grid):
		line = "".join(color_block(v, show_digit=show_digit) for v in row)
		lines.append(line)
	return lines


def print_grid(grid, show_digit: bool = False, title: str | None = None) -> None:
	if title:
		print(title)
	for line in grid_to_lines(grid, show_digit=show_digit):
		print(line)


def print_side_by_side(grids: List, titles: List[str] | None = None, show_digit: bool = False, spacing: int = 4) -> None:
	"""Print multiple grids side-by-side with optional titles.

	grids: list of 2D arrays / lists
	titles: optional list of same length with headings
	show_digit: include digits inside colored cells
	spacing: spaces between grid columns
	"""
	titles = titles or [""] * len(grids)
	max_title_height = 1 if any(titles) else 0
	rendered = [grid_to_lines(g, show_digit=show_digit) for g in grids]
	heights = [len(r) for r in rendered]
	max_h = max(heights)
	pad = " " * spacing

	# Print titles row
	if max_title_height:
		title_line = pad.join(f"{t}" for t in titles)
		print(title_line)

	for i in range(max_h):
		parts = []
		for lines in rendered:
			if i < len(lines):
				parts.append(lines[i])
			else:
				parts.append("" * len(lines[0]))
		print(pad.join(parts))


__all__ = [
	"color_block",
	"grid_to_lines",
	"print_grid",
	"print_side_by_side",
]
