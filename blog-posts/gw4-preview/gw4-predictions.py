import pandas as pd
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'plots'))
from gw_predictions import primary_pred, secondary_pred


df = pd.read_csv(r"models\team-strength-models\predictions\ensmeble_gw4_preds.csv")
df = df.to_dict(orient="records")

primary_pred(df)
secondary_pred(df)