import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load the CSV
df = pd.read_csv("DigitExperimentData.csv")

# 2) Define actual and predicted digit sets
actual_digits    = [0, 1, 3, 6, 7]             # rows
predicted_digits = [0, 1, 3, 4, 5, 6, 7, 8, 9] # columns

# 3) Identify unique Light Conditions
light_conditions = df['Light Condition'].unique()

# Keep track of accuracies per digit across all conditions
all_accs = []  # to collect (Condition, Digit, Accuracy) for the grouped bar chart
overall_accuracies = []  # to collect (Condition, OverallAccuracy)

# 4) For each Light Condition:
for condition in light_conditions:
    df_cond = df[df['Light Condition'] == condition]

    # build raw counts
    counts = np.zeros((len(actual_digits), len(predicted_digits)), dtype=int)
    for _, row in df_cond.iterrows():
        t, p = row["Character"], row["Predicted"]
        if t in actual_digits and p in predicted_digits:
            counts[actual_digits.index(t), predicted_digits.index(p)] += 1

    # convert to row‐percentages
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    percent_matrix = counts / row_sums * 100

    # 4a) Heatmap
    labels = np.array([[f"{v:.1f}%" for v in row] for row in percent_matrix])
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        percent_matrix,
        annot=labels, fmt="",
        xticklabels=predicted_digits,
        yticklabels=actual_digits,
        vmin=0, vmax=100,
        cmap="Blues",
        cbar_kws={'format':'%.0f%%','label':'Percent'}
    )
    plt.title(f"{condition} — Confusion Matrix (%)")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")
    plt.tight_layout()
    plt.show()

    # 4b) Per‐digit accuracy bar chart
    # (find the diagonal entries: predicted == actual)
    accs = []
    for i, digit in enumerate(actual_digits):
        j = predicted_digits.index(digit)
        acc = percent_matrix[i, j]
        accs.append(acc)
        all_accs.append((condition, digit, acc))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=actual_digits, y=accs, palette="Blues")
    plt.ylim(0, 100)
    plt.title(f"{condition} — Recognition Accuracy by Digit")
    plt.xlabel("Digit")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.show()

    # 4c) Compute overall accuracy for this condition
    total_sum = counts.sum()  # total attempts for this condition
    diag_sum = 0
    for i, digit in enumerate(actual_digits):
        if digit in predicted_digits:
            diag_sum += counts[i, predicted_digits.index(digit)]
    # Avoid division by zero
    if total_sum > 0:
        overall_acc = diag_sum / total_sum * 100
    else:
        overall_acc = 0.0
    overall_accuracies.append((condition, overall_acc))

# 5) Finally, a single grouped‐bar chart across all light conditions (per‐digit accuracy)
acc_df = pd.DataFrame(all_accs, columns=["Light Condition","Digit","Accuracy"])

plt.figure(figsize=(8, 5))
sns.barplot(
    data=acc_df,
    x="Digit",
    y="Accuracy",
    hue="Light Condition",
    palette="Set2"
)
plt.ylim(0, 100)
plt.title("Overall Digit Recognition Accuracy by Light Condition (Per Digit)")
plt.xlabel("Digit")
plt.ylabel("Accuracy (%)")
plt.legend(title="Light Condition")
plt.tight_layout()
plt.show()

# 6) NEW: Bar chart of overall accuracy per light condition
overall_df = pd.DataFrame(overall_accuracies, columns=["Light Condition", "Overall Accuracy"])
plt.figure(figsize=(6, 4))
ax = sns.barplot(
    data=overall_df,
    x="Light Condition",
    y="Overall Accuracy",
    palette="Set2"
)
plt.ylim(0, 100)
plt.title("Overall Accuracy by Light Condition - Digit Recognition")
plt.xlabel("Light Condition")
plt.ylabel("Accuracy (%)")

# Optionally, annotate each bar with its numeric value
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%')

plt.tight_layout()
plt.show()
