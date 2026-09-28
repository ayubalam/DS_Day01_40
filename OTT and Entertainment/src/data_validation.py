import pandas as pd

file_path = "data/raw/ott_shows.csv"

df = pd.read_csv(file_path)

print("\nBUDGET ZERO VALUES")
print((df["budget"] == 0).sum())

print("\nREVENUE ZERO VALUES")
print((df["revenue"] == 0).sum())

print("\nRUNTIME ZERO VALUES")
print((df["runtime"] == 0).sum())

print("\nVOTE COUNT ZERO VALUES")
print((df["vote_count"] == 0).sum())

print("\nPOPULARITY ZERO VALUES")
print((df["popularity"] == 0).sum())

print("\nVOTE AVERAGE RANGE")
print(df["vote_average"].min(), "to", df["vote_average"].max())

print("\nTOP LANGUAGES")
print(df["original_language"].value_counts().head(15))

print("\nTOP GENRES")
print(df["genres"].value_counts().head(15))

print("\nRELEASE DATE SAMPLE")
print(df["release_date"].head(10))

print("\nRUNTIME STATISTICS")
print(df["runtime"].describe())

print("\nBUDGET STATISTICS")
print(df["budget"].describe())

print("\nREVENUE STATISTICS")
print(df["revenue"].describe())

print("\nPOPULARITY STATISTICS")
print(df["popularity"].describe())