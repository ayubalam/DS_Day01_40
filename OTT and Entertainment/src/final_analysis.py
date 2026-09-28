import pandas as pd

file_path = "data/processed/ott_shows_cleaned.csv"

df = pd.read_csv(file_path)

median_popularity = df["popularity"].median()

df["success"] = (
    df["popularity"] >= median_popularity
).astype(int)

print("\nSUCCESSFUL TITLES")
print(df[df["success"] == 1].shape[0])

print("\nLESS SUCCESSFUL TITLES")
print(df[df["success"] == 0].shape[0])

print("\nTOP 10 GENRES")
print(
    df.groupby("genres")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP 10 LANGUAGES")
language_counts = df["original_language"].value_counts()

valid_languages = language_counts[
    language_counts >= 100
].index

print(
    df[df["original_language"].isin(valid_languages)]
    .groupby("original_language")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nRATING VS POPULARITY CORRELATION")
print(
    df["vote_average"].corr(
        df["popularity"]
    )
)

print("\nBUDGET VS POPULARITY CORRELATION")
budget_data = df.dropna(
    subset=["budget", "popularity"]
)

print(
    budget_data["budget"].corr(
        budget_data["popularity"]
    )
)

print("\nRATING STATISTICS")
print(df["vote_average"].describe())

print("\nPOPULARITY STATISTICS")
print(df["popularity"].describe())

print("\nRUNTIME STATISTICS")
print(df["runtime"].describe())

print("\nTOP 10 TITLES BY POPULARITY")
print(
    df[
        [
            "title",
            "genres",
            "original_language",
            "vote_average",
            "popularity"
        ]
    ]
    .sort_values(
        by="popularity",
        ascending=False
    )
    .head(10)
)