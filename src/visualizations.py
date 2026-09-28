import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/processed/ott_shows_cleaned.csv"

df = pd.read_csv(file_path)

budget_data = df.dropna(
    subset=["budget", "popularity"]
)

correlation = budget_data["budget"].corr(
    budget_data["popularity"]
)

print("\nBUDGET VS POPULARITY CORRELATION")
print(correlation)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=budget_data,
    x="budget",
    y="popularity",
    alpha=0.3
)

plt.title("Budget vs Popularity")
plt.xlabel("Budget")
plt.ylabel("Popularity")
plt.tight_layout()

plt.savefig(
    "visualizations/budget_vs_popularity.png",
    dpi=300
)

plt.show()