import pandas as pd
import sys
import os

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# Read the file
file_path = os.path.join("models", "team-strength-models", "predictions", "ensmeble_gw22_preds.csv")      
df = pd.read_csv(file_path)

# Print the shape of your dataframe
print("DataFrame shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))

# Convert to records AFTER confirming data looks correct
df_records = df.to_dict(orient="records")

# Now call the function
from viz.model_pred_viz.gw_predictions import primary_pred
primary_pred(df_records)

from viz.model_pred_viz.gw_handicap_bar import twoway_expected_goals
twoway_expected_goals(df_records)