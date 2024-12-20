import pandas as pd

df_possession = pd.read_html("https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats",
                  attrs={"id": "stats_passing"})[0]

print(df_possession.columns)