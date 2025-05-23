import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── 1) Fill in with your actual measurements:
box_width  = 36/72   # inches (e.g. 36 pt ÷ 72)
box_height = 36/72   # inches

# ── 2) Use the same font size as your PDF text:
UNIFORM_FONTSIZE = 12

# ── 3) Apply theme with uniform font sizing:
sns.set_theme(style="whitegrid", rc={
    "font.size":       UNIFORM_FONTSIZE,
    "axes.labelsize":  UNIFORM_FONTSIZE,
    "xtick.labelsize": UNIFORM_FONTSIZE,
    "ytick.labelsize": UNIFORM_FONTSIZE,
})

# Sample data
data = {
    "Light Condition": [
        "Dark", "Dark",
        "Daylight\n+ Artificial", "Daylight\n+ Artificial",
        "Daylight\nNo Artificial", "Daylight\nNo Artificial"
    ],
    "Accuracy": [96, 86, 82, 56, 76, 72],
    "Type": ["Digit", "Letter"] * 3
}
df = pd.DataFrame(data)

# Create figure exactly the size of one PDF box:
plt.figure(figsize=(box_width, box_height), dpi=150)

ax = sns.barplot(
    data=df,
    x="Light Condition", y="Accuracy",
    hue="Type", palette="Set2",
    width=0.8, dodge=0.2
)

# Remove legend, set labels/titles in uniform font
if ax.get_legend(): ax.get_legend().remove()
ax.set_xlabel("Light Condition", fontsize=UNIFORM_FONTSIZE)
ax.set_ylabel("Accuracy (%)",       fontsize=UNIFORM_FONTSIZE)
ax.set_title("",                     fontsize=UNIFORM_FONTSIZE)  # omit if unneeded

# Annotate bars with same-size labels
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f%%", fontsize=UNIFORM_FONTSIZE)

# (Optional) add difference‐bracket logic here,
# using the same UNIFORM_FONTSIZE for any text.

plt.tight_layout(pad=0)
plt.savefig("inbox_chart.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0)
plt.show()
