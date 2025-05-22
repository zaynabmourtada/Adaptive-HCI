import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", font_scale=1.2, rc={"figure.dpi": 150})

data = {
    "Light Condition": [
        "Dark", "Dark",
        "Daylight w/ Artificial", "Daylight w/ Artificial",
        "Daylight no Artificial", "Daylight no Artificial"
    ],
    "Accuracy": [96, 86, 82, 56, 76, 72],
    "Type": ["Digit", "Letter", "Digit", "Letter", "Digit", "Letter"]
}
df = pd.DataFrame(data)

# Optionally shorten/wrap labels:
df["Light Condition"] = df["Light Condition"].replace({
    "Daylight w/ Artificial": "Daylight\n+ Artificial",
    "Daylight no Artificial": "Daylight\nNo Artificial"
})

# Increase the figure's height (second number)
plt.figure(figsize=(2, 3.5))
ax = sns.barplot(
    data=df,
    x="Light Condition",
    y="Accuracy",
    hue="Type",
    palette="Set2",
    width=0.8,
    dodge=0.2
)

# Remove the legend
if ax.get_legend() is not None:
    ax.get_legend().remove()

plt.ylim(0, 100)
plt.xlabel("Light Condition")
plt.ylabel("Accuracy")

# Label each bar with its own accuracy
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f%%')

# ---------------------------------------------------------
#  Add bracketed difference indicator if Digit bar is taller
# ---------------------------------------------------------
digit_bars, letter_bars = ax.containers

BRACKET_MARGIN     = 0.8
BRACKET_HALF_WIDTH = 0.02
TEXT_OFFSET        = 0.02
X_OFFSET_BRACKET   = -0.04
BRACKET_COLOR      = "blue"

for dbar, lbar in zip(digit_bars, letter_bars):
    x_dig = dbar.get_x() + dbar.get_width()/2
    y_dig = dbar.get_height()
    x_let = lbar.get_x() + lbar.get_width()/2
    y_let = lbar.get_height()

    if y_dig > y_let:
        diff_val = y_dig - y_let

        # Scale margin if difference is small
        min_required = 2 * BRACKET_MARGIN
        if diff_val < min_required:
            scaled_margin = diff_val / 4.0
        else:
            scaled_margin = BRACKET_MARGIN

        bracket_top = y_dig - scaled_margin
        bracket_bottom = y_let + scaled_margin

        x_bracket = x_dig + X_OFFSET_BRACKET
        if bracket_bottom < bracket_top:
            # Vertical line
            ax.plot([x_bracket, x_bracket],
                    [bracket_bottom, bracket_top],
                    color=BRACKET_COLOR, linewidth=1.2)
            # T-ends
            ax.plot([x_bracket - BRACKET_HALF_WIDTH, x_bracket + BRACKET_HALF_WIDTH],
                    [bracket_top, bracket_top],
                    color=BRACKET_COLOR, linewidth=1.2)
            ax.plot([x_bracket - BRACKET_HALF_WIDTH, x_bracket + BRACKET_HALF_WIDTH],
                    [bracket_bottom, bracket_bottom],
                    color=BRACKET_COLOR, linewidth=1.2)

            # Difference label
            y_mid = (bracket_bottom + bracket_top) / 2
            ax.text(
                x_bracket + TEXT_OFFSET, y_mid,
                f"+{diff_val:.1f}%",
                va='center', ha='left',
                color=BRACKET_COLOR, fontsize=9
            )

plt.tight_layout()
plt.show()
