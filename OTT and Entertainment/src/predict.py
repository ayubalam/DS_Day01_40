import pandas as pd
import joblib

model = joblib.load("models/random_forest_model.joblib")

new_title = pd.DataFrame({
    "genres": ["Action"],
    "original_language": ["English"],
    "runtime": [120],
    "release_year": [2026],
    "budget": [50000000],
    "vote_average": [7.5],
    "vote_count": [1000]
})

prediction = model.predict(new_title)

if prediction[0] == 1:
    print("\nPREDICTION")
    print("The title is predicted to be SUCCESSFUL")
else:
    print("\nPREDICTION")
    print("The title is predicted to be LESS SUCCESSFUL")