"""Slide 9 — REINFORCE training curve.

Crops the top two panels (train + evaluation win rate) from the existing
training-results PNG. The third panel (average episode return) is dropped.

The original PNG is shipped at impzap/graphs/reinforce_50_000_training_results.png
and is in English already, so it is reused as-is rather than regenerated.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# _style is imported only for the OUT_DIR convention; PIL handles the actual save.
from _style import OUT_DIR


SOURCE = (
    Path(__file__).resolve().parents[3] / "graphs"
    / "reinforce_50_000_training_results.png"
)


def _detect_panel_split(img: Image.Image) -> int:
    """Find the y-coordinate of the gutter between panel 2 and panel 3.

    Strategy: enumerate all contiguous all-white horizontal runs, merge runs
    that are separated by less than 25 px (single text lines such as
    bottom x-axis labels are ~5-15 px tall and would otherwise split a real
    inter-panel gutter), then pick the longest merged run whose centre lies
    in the lower 60% of the image. That run is the panel-2/panel-3 gutter.
    Cut at its centre.
    """
    grayscale = img.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()

    sample_xs = [int(width * (i + 1) / 61) for i in range(60)]
    threshold = 248

    def row_is_gutter(y: int) -> bool:
        return all(pixels[x, y] >= threshold for x in sample_xs)

    runs: list[tuple[int, int]] = []
    in_gutter = False
    run_start = 0
    for y in range(height):
        gutter = row_is_gutter(y)
        if gutter and not in_gutter:
            in_gutter = True
            run_start = y
        elif not gutter and in_gutter:
            in_gutter = False
            runs.append((run_start, y - 1))
    if in_gutter:
        runs.append((run_start, height - 1))

    # Merge runs separated by a thin ink band (<= 25 px).
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= 25:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    lower_half_threshold = int(height * 0.40)
    candidates = [
        (start, end) for start, end in merged
        if (start + end) / 2 >= lower_half_threshold
        and end < height - 30  # exclude bottom page margin
    ]
    if candidates:
        start, end = max(candidates, key=lambda r: r[1] - r[0])
        return (start + end) // 2

    # Fallback: cut at 80% of the height (panels are roughly equal thirds).
    return int(height * 0.80)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Expected source image at {SOURCE}")

    img = Image.open(SOURCE)
    width, height = img.size

    cut_y = _detect_panel_split(img)
    # Sanity guard: never cut above 60% or below 90% of the image height.
    cut_y = max(int(height * 0.6), min(cut_y, int(height * 0.90)))

    cropped = img.crop((0, 0, width, cut_y))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "slide_09_reinforce_curve.png"
    cropped.save(out_path, dpi=(200, 200), optimize=True)
    print(f"wrote {out_path} ({cropped.size[0]}x{cropped.size[1]} px)")


if __name__ == "__main__":
    main()
