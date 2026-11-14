"""Plot token counts before and after token reduction for four GUI datasets."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["AndroidControl", "GUI Odyssey", "AITW", "Mind2Web"]
BEFORE = np.array([8940, 14573, 2638, 3660])
AFTER = np.array([3157, 3960, 1101, 1411])
REMOVED = BEFORE - AFTER

OUTDIR = Path("figures")
STEM = OUTDIR / "token_reduction_by_dataset"


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 30,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 1.5,
    })

    fig, ax = plt.subplots(figsize=(16.5, 10.2))
    x = np.arange(len(DATASETS))
    width = 0.76
    after_color = "#5A5A91"
    removed_color = "#AEBCE0"

    ax.bar(x, AFTER, width, color=after_color, edgecolor="white", linewidth=1.2,
           label="After token reduction", zorder=3)
    ax.bar(x, REMOVED, width, bottom=AFTER, color=removed_color,
           edgecolor="white", linewidth=1.2, label="Origin tokens", zorder=3)

    for xi, before, after in zip(x, BEFORE, AFTER):
        ax.text(xi, after / 2, f"{after:,}", ha="center", va="center",
                color="white", fontsize=42, fontweight="bold")
        ax.text(xi, before + 280, f"{before:,}", ha="center", va="bottom",
                color="black", fontsize=42, fontweight="bold")

    ax.set_ylabel("Token count", fontsize=38, fontweight="bold", labelpad=26)
    ax.set_xticks(x, DATASETS, fontsize=31, fontweight="bold")
    ax.set_ylim(0, 17250)
    ax.set_yticks(np.arange(0, 17501, 2500))
    ax.tick_params(axis="y", labelsize=30, width=2.0, length=8)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.75, 0.24), fontsize=40,
              frameon=False, handlelength=1.5, borderaxespad=0.7, labelspacing=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig(f"{STEM}.svg", bbox_inches="tight")
    fig.savefig(f"{STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{STEM}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{STEM}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
