import pymc as pm
import numpy as np
import pandas as pd

# Assuming you have a DataFrame 'data' with columns: home_team, away_team, home_goals, away_goals, home_xg, away_xg, league

import pandas as pd
import requests
import json
import os
import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
from pprint import pprint
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
import io
import multiprocessing




def build_model(data):
    with pm.Model() as model:
        # Priors for team strengths
        num_teams = len(set(data['home_team']) | set(data['away_team']))
        team_strength = pm.Normal('team_strength', mu=0, sigma=1, shape=num_teams)
        
        # League effect
        league_effect = pm.Normal('league_effect', mu=0, sigma=0.5)
        
        # Home advantage
        home_advantage = pm.Normal('home_advantage', mu=0.1, sigma=0.1)
        
        # Model parameters
        beta_goals = pm.Beta('beta_goals', alpha=3, beta=7)  # 30% weight
        beta_xg = pm.Beta('beta_xg', alpha=7, beta=3)  # 70% weight
        
        # Expected score
        home_idx = pd.Categorical(data['home_team']).codes
        away_idx = pd.Categorical(data['away_team']).codes
        league_idx = pd.Categorical(data['division']).codes
        
        home_expect = (
            team_strength[home_idx] + home_advantage + 
            league_effect * league_idx
        )
        away_expect = team_strength[away_idx]
        
        # Convert data to tensors
        home_xg = pm.Data('home_xg', data['home_xgoals'].values)
        away_xg = pm.Data('away_xg', data['away_xgoals'].values)
        
        # Likelihood
        home_goals_like = pm.Poisson('home_goals', 
                                     mu=pm.math.exp(home_expect) * beta_goals + home_xg * beta_xg, 
                                     observed=data['home_goals'].values)
        away_goals_like = pm.Poisson('away_goals', 
                                     mu=pm.math.exp(away_expect) * beta_goals + away_xg * beta_xg, 
                                     observed=data['away_goals'].values)
    
    return model

# Function to run the model
def run_model(data, samples=2000, chains=2):
    model = build_model(data)
    with model:
        trace = pm.sample(samples, chains=chains, return_inferencedata=True)
    return trace

# Function to get team strengths
def get_team_strengths(trace, data):
    team_names = list(set(data['home_team']) | set(data['away_team']))
    strengths = trace.posterior['team_strength'].mean(dim=("chain", "draw")).values
    return pd.DataFrame({'team': team_names, 'strength': strengths}).sort_values('strength', ascending=False)


def main():
    
    # Suppress divide by zero warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")


    API_KEY = os.getenv("API_KEY")
    url = 'https://data-service.beatthebookie.blog/data'
    headers = {"x-api-key": API_KEY}
    params = {'division':'Premier League'}
    response = requests.get(url, headers=headers, params=params)
    json_str = response.content.decode('utf-8')
    prem_df = pd.read_json(io.StringIO(json_str))
    prem_teams_25 = prem_df[prem_df["season"] == 20242025]
    prem_teams_25 = pd.concat([prem_teams_25['home_team'], prem_teams_25['away_team']]).unique()

    params = {'division':'Championship'}
    response = requests.get(url, headers=headers, params=params)
    json_str = response.content.decode('utf-8')
    champ_df = pd.read_json(io.StringIO(json_str))

    df = pd.concat([champ_df, prem_df])
    #df = df[(df['home_team'].isin(prem_teams_25)) | (df['away_team'].isin(prem_teams_25))]

    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df[df["match_date"] > '2023-06-01']

    print(df[["season", "match_date", "home_team", "away_team", "home_goals", "home_xgoals", "away_goals", "away_xgoals"]].tail())
    
    # Run the model
    trace = run_model(df)
    
    # Get team strengths
    team_strengths = get_team_strengths(trace, df)
    print(team_strengths)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line
    main()