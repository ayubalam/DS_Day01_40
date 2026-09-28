import subprocess
import sys

steps = [
    "src/data_cleaning.py",
    "src/eda.py",
    "src/visualizations.py",
    "src/random_forest_model.py",
    "src/predict.py",
    "src/final_analysis.py"
]

for script in steps:
    print(f"\nRunning {script}...")
    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:
        print(f"\nFailed: {script}")
        sys.exit(result.returncode)

print("\nAll project steps completed.")
print("\nStarting dashboard...\n")

subprocess.run(
    ["streamlit", "run", "dashboard/app.py"]
)