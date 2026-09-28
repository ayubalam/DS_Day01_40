import pandas as pd

input_file = "data/raw/ott_shows.csv"
output_file = "data/processed/ott_shows_cleaned.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

df = df.drop_duplicates()

df["release_date"] = pd.to_datetime(
    df["release_date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year

df["budget"] = df["budget"].replace(0, pd.NA)
df["revenue"] = df["revenue"].replace(0, pd.NA)
df["runtime"] = df["runtime"].replace(0, pd.NA)
df["vote_count"] = df["vote_count"].replace(0, pd.NA)

df = df.dropna(
    subset=[
        "title",
        "genres",
        "original_language",
        "vote_average",
        "popularity",
        "runtime",
        "release_year"
    ]
)

df = df[
    (df["vote_average"] >= 0) &
    (df["vote_average"] <= 10) &
    (df["runtime"] > 0) &
    (df["popularity"] > 0)
]

columns_to_keep = [
    "id",
    "title",
    "genres",
    "original_language",
    "runtime",
    "release_year",
    "budget",
    "vote_average",
    "vote_count",
    "popularity",
    "revenue"
]

df = df[columns_to_keep]

df.to_csv(output_file, index=False)

print("Cleaned shape:", df.shape)
print("Cleaned dataset saved to:", output_file)
print("\nMissing values after cleaning:")
print(df.isnull().sum())