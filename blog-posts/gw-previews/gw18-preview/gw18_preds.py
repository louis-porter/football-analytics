import pandas as pd
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'viz', 'model-pred-viz'))
from gw_predictions import primary_pred
from gw_handicap_bar import twoway_expected_goals


file_path = os.path.join("models", "team-strength-models", "predictions", "ensmeble_gw18_preds.csv")



df = pd.read_csv(file_path)

df = df.to_dict(orient="records")

primary_pred(df)
twoway_expected_goals(df)
