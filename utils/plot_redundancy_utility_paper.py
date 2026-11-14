#!/usr/bin/env python3
"""Render the camera-ready redundancy--utility summary from fixed source estimates.

The estimates below are transcribed from the approved six-group summary figure.
This renderer intentionally does not read worker outputs or recompute any statistic.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path('OUTPUT_history_redundancy_utility/AndroidControl_Curated_High_Task_Improved')
OUT = ROOT / 'figures' / 'redundancy_utility_density'

# group, tertile label, n, mean, lower 95% CI, upper 95% CI, colour
SUMMARY = [
    ('Local regions', 'Low\n($r < 0.517$)', 204, -0.00165, -0.00571, 0.00259, '#377eb8'),
    ('Local regions', 'Medium\n($0.517 \\leq r < 0.818$)', 204, -0.00124, -0.00618, 0.00283, '#377eb8'),
    ('Local regions', 'High\n($r \\geq 0.818$)', 204, -0.00241, -0.00589, 0.00100, '#377eb8'),
    ('Full-frame regions', 'Low\n($r < 0.863$)', 68, 0.00306, -0.01196, 0.01900, '#e76f51'),
    ('Full-frame regions', 'Medium\n($0.863 \\leq r < 0.958$)', 68, -0.00642, -0.02266, 0.01072, '#e76f51'),
    ('Full-frame regions', 'High\n($r \\geq 0.958$)', 68, -0.01443, -0.03169, 0.00330, '#e76f51'),
]


def main():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10.5,
        'axes.labelsize': 11.5,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })

    # Double-column (183 mm) figure.  The title and explanatory sentence
    # belong in the manuscript caption, not in this standalone panel.
    fig, ax = plt.subplots(figsize=(7.5, 3.75))
    fig.subplots_adjust(left=.285, right=.95, bottom=.26, top=.95)
    y_positions = list(range(len(SUMMARY) - 1, -1, -1))

    for y, (_, _, _, mean, lo, hi, colour) in zip(y_positions, SUMMARY):
        ax.errorbar(
            mean, y, xerr=[[mean - lo], [hi - mean]], fmt='o',
            color=colour, ecolor=colour, elinewidth=1.35, capsize=3,
            markersize=6.2, markerfacecolor='white', markeredgewidth=1.6,
            zorder=3,
        )

    ax.axvline(0, color='#777777', lw=.7, ls=(0, (4, 3)), zorder=1)
    ax.axhline(2.5, color='#d0d0d0', lw=.65, zorder=1)
    ax.grid(axis='x', color='#dedede', lw=.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-.0342, .0214)
    ax.set_xticks([-.03, -.02, -.01, 0, .01, .02])
    ax.set_xlabel('Masking impact on gold-action log-likelihood', labelpad=5)
    ax.set_yticks(y_positions, [''] * len(SUMMARY))
    ax.tick_params(axis='y', length=0)
    # Separate text objects give the category a normal, readable line gap
    # while making the threshold itself visually subordinate and more compact.
    for y, item in zip(y_positions, SUMMARY):
        level, threshold = item[1].split('\n')
        ax.text(-.0348, y + .15, level, ha='right', va='center', fontsize=10.5,
                clip_on=False)
        ax.text(-.0348, y - .15, threshold, ha='right', va='center', fontsize=9.2,
                clip_on=False)
    ax.tick_params(axis='x', width=.8, length=3.5)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_linewidth(.8)

    # Region type is encoded once in the legend instead of repeated in all
    # six labels.  Its empty upper-left area avoids covering data.
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', color='#377eb8',
               markerfacecolor='white', markeredgewidth=1.9, markersize=7.8,
               label='Local regions'),
        Line2D([0], [0], marker='o', linestyle='None', color='#e76f51',
               markerfacecolor='white', markeredgewidth=1.9, markersize=7.8,
               label='Full-frame regions'),
    ]
    ax.legend(handles=handles, loc='upper left', frameon=False,
              handletextpad=.4, borderpad=.1, labelspacing=.28)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{OUT}.svg', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{OUT}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{OUT}.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{OUT}.tiff', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
