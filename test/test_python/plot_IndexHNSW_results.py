import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

file_name = "test_IndexHNSW_recall_dim128_size500000_normal0.csv"
df = pd.read_csv(os.path.join(os.path.dirname(__file__),file_name))

figure = plt.figure(figsize=(12,8))

colors = [
    "#2E86AB", "#A23B72", "#F18F01", "#3CBBB1", "#D63230", "#6A4C93", "#88D18A",
    "#FFD700", "#4B0082", "#FF6B6B", "#20B2AA", "#FFA07A", "#9370DB", "#32CD32"
]

for i, (k, color) in enumerate(zip(df["k"].unique(), colors)):
    df_certain_k = df[df["k"]==k]
    plt.plot(
        df_certain_k["efSearch"],
        df_certain_k["mean_recall"],
        "-o",
        color=color,
        label=f"k={k}" 
    )

plt.legend(title="K Values")
plt.xlabel("efSearch")
plt.ylabel("Mean Recall")
plt.title("Recall Performance by efSearch and K")
plt.grid(True)
plt.savefig(os.path.join(os.path.dirname(__file__),file_name + ".png"))
plt.show()
