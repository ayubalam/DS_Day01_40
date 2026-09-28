import pandas as pd

file_path = "data/processed/ott_shows_cleaned.csv"

df = pd.read_csv(file_path)

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

print("\nAVERAGE RATING")
print(df["vote_average"].mean())

print("\nMEDIAN RATING")
print(df["vote_average"].median())

print("\nAVERAGE POPULARITY")
print(df["popularity"].mean())

print("\nMEDIAN POPULARITY")
print(df["popularity"].median())

print("\nAVERAGE RUNTIME")
print(df["runtime"].mean())

print("\nMEDIAN RUNTIME")
print(df["runtime"].median())

print("\nTOP 10 GENRES")
print(df["genres"].value_counts().head(10))

print("\nTOP 15 LANGUAGES")
print(df["original_language"].value_counts().head(15))

print("\nCORRELATION MATRIX")
print(
    df[
        [
            "runtime",
            "budget",
            "vote_average",
            "vote_count",
            "popularity",
            "revenue"
        ]
    ].corr()
)