import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="OTT Content Performance Dashboard",
    layout="wide"
)

file_path = "data/processed/ott_shows_cleaned.csv"

df = pd.read_csv(file_path)

median_popularity = df["popularity"].median()

df["success"] = (
    df["popularity"] >= median_popularity
).astype(int)

st.title("OTT Content Performance Dashboard")

st.write(
    "Analysis of content characteristics and their relationship "
    "with audience performance."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Titles",
    f"{len(df):,}"
)

col2.metric(
    "Average Rating",
    f"{df['vote_average'].mean():.2f}"
)

col3.metric(
    "Average Popularity",
    f"{df['popularity'].mean():.2f}"
)

col4.metric(
    "Average Runtime",
    f"{df['runtime'].mean():.1f} min"
)

st.subheader("Genre Performance")

genre_performance = (
    df.groupby("genres")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    x=genre_performance.values,
    y=genre_performance.index,
    ax=ax
)

ax.set_xlabel("Average Popularity")
ax.set_ylabel("Genre")
ax.set_title("Top 10 Genres by Average Popularity")

st.pyplot(fig)

st.subheader("Language Performance")

language_counts = df["original_language"].value_counts()

valid_languages = language_counts[
    language_counts >= 100
].index

language_performance = (
    df[df["original_language"].isin(valid_languages)]
    .groupby("original_language")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    x=language_performance.values,
    y=language_performance.index,
    ax=ax
)

ax.set_xlabel("Average Popularity")
ax.set_ylabel("Language")
ax.set_title("Top 10 Languages by Average Popularity")

st.pyplot(fig)

st.subheader("Rating vs Popularity")

fig, ax = plt.subplots(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="vote_average",
    y="popularity",
    alpha=0.4,
    ax=ax
)

ax.set_xlabel("Rating")
ax.set_ylabel("Popularity")
ax.set_title("Rating vs Popularity")

st.pyplot(fig)

st.subheader("Budget vs Popularity")

budget_data = df.dropna(
    subset=["budget", "popularity"]
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.scatterplot(
    data=budget_data,
    x="budget",
    y="popularity",
    alpha=0.3,
    ax=ax
)

ax.set_xlabel("Budget")
ax.set_ylabel("Popularity")
ax.set_title("Budget vs Popularity")

st.pyplot(fig)

st.subheader("Success Distribution")

success_counts = df["success"].value_counts()

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    x=["Less Successful", "Successful"],
    y=[
        success_counts.get(0, 0),
        success_counts.get(1, 0)
    ],
    ax=ax
)

ax.set_xlabel("Content Performance")
ax.set_ylabel("Number of Titles")
ax.set_title("Successful vs Less Successful Titles")

st.pyplot(fig)