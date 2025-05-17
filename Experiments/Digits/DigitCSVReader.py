import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load the CSV
df = pd.read_csv("data.csv")

# 2) Define actual and predicted digit sets
actual_digits = [0, 1, 3, 6, 7]            # rows
predicted_digits = [0, 1, 3, 6, 7, 8, 9]   # columns

# 3) Identify unique Light Conditions
light_conditions = df['Light Condition'].unique()

# 4) For each Light Condition, build and plot a 5x7 confusion matrix
for condition in light_conditions:
    df_cond = df[df['Light Condition'] == condition]

    # Prepare 5x7 matrix for counts (rows=5 actual, cols=7 predicted)
    counts = np.zeros((len(actual_digits), len(predicted_digits)), dtype=int)

    # Populate the matrix
    for _, row in df_cond.iterrows():
        true_val = row["Character"]
        pred_val = row["Predicted"]
        if true_val in actual_digits and pred_val in predicted_digits:
            row_idx = actual_digits.index(true_val)
            col_idx = predicted_digits.index(pred_val)
            counts[row_idx, col_idx] += 1

    # Convert counts to row-based percentages
    row_sums = counts.sum(axis=1, keepdims=True)  # sums across columns per row
    row_sums[row_sums == 0] = 1  # to avoid division by zero
    percent_matrix = (counts / row_sums) * 100

    # Plot as heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        percent_matrix,
        annot=True,
        fmt=".1f",
        xticklabels=predicted_digits,  # columns
        yticklabels=actual_digits      # rows
    )
    plt.title(f"{condition} — 5×7 Confusion Matrix in Percent")
    plt.xlabel("Predicted Digits (7 columns)")
    plt.ylabel("Actual Digits (5 rows)")
    plt.tight_layout()
    plt.show()