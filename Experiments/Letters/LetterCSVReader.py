import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# 1) Load the CSV
df = pd.read_csv("LetterExperimentData.csv")

# 2) Detect labels and conditions
actual = sorted(df['Character'].unique())
pred   = sorted(df['Predicted'].unique())
conds  = df['Light Condition'].unique()

# 3) Prepare colormap and normalizer (0–100%)
cmap = plt.cm.Blues
norm = colors.Normalize(vmin=0, vmax=100)

# 4) Containers for summary stats
all_accs = []
overall  = []

for cond in conds:
    sub = df[df['Light Condition'] == cond]
    counts = (pd.crosstab(sub['Character'], sub['Predicted'])
                .reindex(index=actual, columns=pred, fill_value=0))
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100

    # --- Confusion matrix heatmap ---
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pct.values, cmap=cmap, norm=norm, aspect='auto')
    for i, row in enumerate(pct.values):
        for j, v in enumerate(row):
            text_color = 'white' if norm(v) > 0.5 else 'black'
            ax.text(j, i, f"{v:.1f}%", ha='center', va='center', color=text_color)
    ax.set_xticks(range(len(pred))); ax.set_xticklabels(pred)
    ax.set_yticks(range(len(actual))); ax.set_yticklabels(actual)
    ax.set_title(f"{cond} — Confusion (%)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    cbar = fig.colorbar(im, ax=ax, format="%.0f%%")
    cbar.set_label("Percent")
    plt.tight_layout()
    plt.show()

    # --- Per-letter accuracy bar chart ---
    acc = [pct.at[l, l] if l in pct.columns else 0 for l in actual]
    all_accs += [(cond, l, pct.at[l, l] if l in pct.columns else 0) for l in actual]

    fig, ax = plt.subplots(figsize=(6, 4))
    bar_colors = cmap(norm(acc))
    bars = ax.bar(actual, acc, color=bar_colors, edgecolor='k')
    ax.set_ylim(0, 100)
    ax.set_title(f"{cond} — Accuracy by Letter")
    ax.set_xlabel("Letter")
    ax.set_ylabel("Accuracy (%)")

    # NOTE: Removed the loop that adds text labels on top of each bar
    # for bar, v in zip(bars, acc):
    #     text_color = 'white' if norm(v) > 0.5 else 'black'
    #     ax.text(bar.get_x() + bar.get_width()/2, v + 1,
    #             f"{v:.1f}%", ha='center', va='bottom', color=text_color)

    plt.tight_layout()
    plt.show()

    # --- Overall accuracy for this condition ---
    total   = counts.values.sum()
    correct = sum(counts.at[l, l] for l in actual if l in counts.columns)
    overall.append((cond, correct/total*100 if total else 0.0))

# 5) Overall accuracy by condition
overall_df = pd.DataFrame(overall, columns=["Light Condition","Overall Accuracy"])
fig, ax = plt.subplots(figsize=(6,4))

# Create the bar chart
ov_colors = cmap(norm(overall_df["Overall Accuracy"]))
bars = ax.bar(overall_df["Light Condition"], overall_df["Overall Accuracy"],
              color=ov_colors, edgecolor='k')

ax.set_ylim(0, 100)
ax.set_title("Overall Accuracy by Light Condition - Letter Recognition")
ax.set_xlabel("Light Condition")
ax.set_ylabel("Accuracy (%)")

# Label each bar on top (only in the overall accuracy chart)
for bar, v in zip(bars, overall_df["Overall Accuracy"]):
    ax.annotate(
        f"{v:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center',
        va='bottom',
        color='black',
    )

plt.tight_layout()
plt.show()
