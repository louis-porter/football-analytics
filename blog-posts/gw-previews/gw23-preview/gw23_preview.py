import pandas as pd
import sys
import os
from pathlib import Path

# Get the absolute path of the current file
current_dir = Path(__file__).resolve().parent

# Go up two levels to reach the project root (from gw23-preview to football-analytics)
project_root = current_dir.parent.parent.parent

# Add project root to Python path
sys.path.append(str(project_root))

# Now you can import from viz
from viz.model_pred_viz.gw_predictions import primary_pred
from viz.model_pred_viz.gw_handicap_bar import twoway_expected_goals

# Read the file using pathlib for better path handling
file_path = project_root / "models" / "team-strength-models" / "predictions" / "ensmeble_23_preds.csv"

# Read the CSV
df = pd.read_csv(file_path)

# Print the shape of your dataframe
print("DataFrame shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))

# Convert to records AFTER confirming data looks correct
df_records = df.to_dict(orient="records")

# Now call the functions
primary_pred(df_records)
twoway_expected_goals(df_records)
