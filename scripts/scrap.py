# %%
from pathlib import Path
import pandas as pd

# %%
df = pd.read_csv(Path(__file__).parents[1] / "data" / "animal_spirits.csv")


# %%
df.columns

# %%
df.head()

# %%
df.describe()

# %%
df.info()

# %%
import matplotlib.pyplot as plt

# Assume inflation series is in a column called 'inflation'
# If necessary, change 'inflation' to match the correct column name

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Inflation as a line plot
axs[0].plot(df.index, df["430"], label="Inflation", color="tab:blue")
axs[0].set_title("Inflation Time Series (Line Plot)")
axs[0].set_ylabel("Inflation Rate")
axs[0].legend()

# Plot 2: Inflation as a bar plot
axs[1].plot(df.index, df["1012"], color="tab:orange", label="Inflation")
axs[1].set_title("Inflation Time Series (Bar Plot)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Inflation Rate")
axs[1].legend()

plt.tight_layout()
plt.show()

# %%
# Load the CSV file containing the inflation data
decisions_df = pd.read_csv(Path(__file__).parents[1] / "data" / "csv_export" / "combined" / "decisions_all_enriched.csv")

import matplotlib.pyplot as plt

# Filter for the two inflation groups
df_430 = decisions_df[decisions_df["participant.inflation"] == 430]

# Group by 'Month' and take mean for 'Actual'
mean_actual_430 = df_430.groupby("Month")["Actual"].mean().dropna()

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot for participant.inflation == 430
axs[0].plot(mean_actual_430.index, mean_actual_430.values, label="Inflation 430", color="tab:blue")
axs[0].set_title("Mean Actual per Month (Inflation 430)")
axs[0].set_ylabel("Mean Actual")
axs[0].legend()

# Plot for participant.inflation == 1012
axs[1].plot(mean_actual_1012.index, mean_actual_1012.values, label="Inflation 1012", color="tab:orange")
axs[1].set_title("Mean Actual per Month (Inflation 1012)")
axs[1].set_xlabel("Month")
axs[1].set_ylabel("Mean Actual")
axs[1].legend()

plt.tight_layout()
plt.show()


