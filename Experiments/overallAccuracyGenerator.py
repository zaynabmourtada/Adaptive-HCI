import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1) Font/Theme
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
sns.set_theme(style="whitegrid", font_scale=0.75, rc={"figure.dpi":193})

# 2) Example data
data = {
    "Light Condition": [
        "Dark", "Dark",
        "Daylight with Artificial Light", "Daylight with Artificial Light",
        "Daylight without Artificial Light", "Daylight without Artificial Light"
    ],
    "Accuracy": [96, 86, 82, 56, 76, 72],
    "Type": ["Digit", "Letter", "Digit", "Letter", "Digit", "Letter"]
}
df = pd.DataFrame(data)

# 3) Plot bars close together
plt.figure(figsize=(3.4, 5))
ax = sns.barplot(
    data=df,
    x="Light Condition",
    y="Accuracy",
    hue="Type",
    palette="Set2",
    width=0.8,  # wider bars
    dodge=0.2   # less gap
)
plt.ylim(0, 100)
plt.xlabel("Light Condition", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xticks(rotation=0, fontsize=9)
plt.yticks(fontsize=9)
ax.legend(title="Type", fontsize=10, title_fontsize=10)

# 4) Label each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f%%', fontsize=9)

# 5) Draw difference brackets if Digit bar is taller
digit_bars, letter_bars = ax.containers

# -- Bracket settings --
BRACKET_MARGIN     = 0.5      # vertical gap from top/bottom
BRACKET_HALF_WIDTH = 0.02     # T‐end half‐width
TEXT_OFFSET        = 0.02     # text offset from bracket
X_OFFSET_BRACKET   = -0.04    # shift the bracket line left of the bar center
BRACKET_COLOR      = "blue"  # color that stands out

for dbar, lbar in zip(digit_bars, letter_bars):
    x_dig = dbar.get_x() + dbar.get_width() / 2
    y_dig = dbar.get_height()
    x_let = lbar.get_x() + lbar.get_width() / 2
    y_let = lbar.get_height()

    if y_dig > y_let:
        diff_val = y_dig - y_let
        # 1) Adaptive margin for small differences
        min_required = 2 * BRACKET_MARGIN
        if diff_val < min_required:
            scaled_margin = diff_val / 4.0
        else:
            scaled_margin = BRACKET_MARGIN

        bracket_top = y_dig - scaled_margin
        bracket_bottom = y_let + scaled_margin

        # 2) Shift bracket left so it doesn't overlap the bar label
        x_bracket = x_dig + X_OFFSET_BRACKET

        # Draw the bracket if there's at least a tiny space
        if bracket_bottom < bracket_top:
            # Vertical line
            ax.plot([x_bracket, x_bracket],
                    [bracket_bottom, bracket_top],
                    color=BRACKET_COLOR, linewidth=1.2)
            # T ends
            ax.plot([x_bracket - BRACKET_HALF_WIDTH, x_bracket + BRACKET_HALF_WIDTH],
                    [bracket_top, bracket_top], color=BRACKET_COLOR, linewidth=1.2)
            ax.plot([x_bracket - BRACKET_HALF_WIDTH, x_bracket + BRACKET_HALF_WIDTH],
                    [bracket_bottom, bracket_bottom], color=BRACKET_COLOR, linewidth=1.2)

            # Middle label
            y_mid = (bracket_bottom + bracket_top) / 2
            ax.text(
                x_bracket + TEXT_OFFSET, y_mid,
                f"{diff_val:.1f}%",
                va='center', ha='left',
                color=BRACKET_COLOR, fontsize=9
            )

plt.tight_layout()
plt.show()
