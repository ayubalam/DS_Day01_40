import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

file_path = "data/processed/ott_shows_cleaned.csv"

df = pd.read_csv(file_path)

median_popularity = df["popularity"].median()

df["success"] = (
    df["popularity"] >= median_popularity
).astype(int)

features = [
    "genres",
    "original_language",
    "runtime",
    "release_year",
    "budget",
    "vote_average",
    "vote_count"
]

X = df[features]
y = df["success"]

categorical_features = [
    "genres",
    "original_language"
]

numeric_features = [
    "runtime",
    "release_year",
    "budget",
    "vote_average",
    "vote_count"
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

print("\nRANDOM FOREST TRAINING COMPLETED")

print("\nNUMBER OF TREES")
print(model.n_estimators)

print("\nTRAINING SCORE")
print(model.score(X_train_processed, y_train))

print("\nTESTING SCORE")
print(model.score(X_test_processed, y_test))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

y_pred = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMODEL EVALUATION")

print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
feature_names = preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(feature_importance.head(15))

import joblib

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

model_pipeline.fit(X_train, y_train)

joblib.dump(
    model_pipeline,
    "models/random_forest_model.joblib"
)

print("\nMODEL SAVED")
print("models/random_forest_model.joblib")