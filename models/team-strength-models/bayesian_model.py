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
import pymc as pm
import arviz as az



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


# Load and preprocess data (assuming df is already loaded as in the original code)

# Create team to index mapping
teams = np.sort(np.unique(np.concatenate([df['home_team'], df['away_team']])))
team_to_index = {team: idx for idx, team in enumerate(teams)}

# Convert team names to indices
home_team_idx = np.array([team_to_index[team] for team in df['home_team']])
away_team_idx = np.array([team_to_index[team] for team in df['away_team']])

# Create PyMC model
with pm.Model() as model:
    # Priors
    home_advantage = pm.Normal("home_advantage", mu=0.25, sigma=0.25)
    
    n_teams = len(teams)
    
    # Attack and defense parameters for each team
    attack = pm.Normal("attack", mu=0, sigma=0.5, shape=n_teams)
    defense = pm.Normal("defense", mu=0, sigma=0.5, shape=n_teams)
    
    # Rho parameter for Dixon-Coles adjustment
    rho = pm.Uniform("rho", lower=-1, upper=1)

    # Calculate expected goals
    home_expected = pm.math.exp(attack[home_team_idx] + 
                                defense[away_team_idx] + 
                                home_advantage)
    away_expected = pm.math.exp(attack[away_team_idx] + 
                                defense[home_team_idx])

    # Dixon-Coles adjustment
    def dc_adjustment(home_goals, away_goals, home_exp, away_exp, rho):
        adj = pm.math.switch(
            (home_goals == 0) & (away_goals == 0),
            1 - (home_exp * away_exp * rho),
            pm.math.switch(
                (home_goals == 0) & (away_goals == 1),
                1 + (home_exp * rho),
                pm.math.switch(
                    (home_goals == 1) & (away_goals == 0),
                    1 + (away_exp * rho),
                    pm.math.switch(
                        (home_goals == 1) & (away_goals == 1),
                        1 - rho,
                        1.0
                    )
                )
            )
        )
        return adj

    # Likelihood
    home_goals = pm.Poisson("home_goals", mu=home_expected, observed=df['home_goals'])
    away_goals = pm.Poisson("away_goals", mu=away_expected, observed=df['away_goals'])
    
    # Apply Dixon-Coles adjustment
    adj = dc_adjustment(df['home_goals'], df['away_goals'], home_expected, away_expected, rho)
    pm.Potential("dc_adjustment", pm.math.log(adj))

# Inference
with model:
    trace = pm.sample(1000, tune=1000, chains=2)

# Analyze results
az.summary(trace, var_names=['home_advantage', 'rho'])

# Extract team strengths
team_strengths = pd.DataFrame({
    'Team': teams,
    'Attack': az.summary(trace, var_names=['attack'])['mean'].values,
    'Defense': az.summary(trace, var_names=['defense'])['mean'].values
})
print(team_strengths)

# Prediction function
def predict_bayesian(trace, home_team, away_team):
    home_idx = team_to_index[home_team]
    away_idx = team_to_index[away_team]
    
    home_attack = trace.posterior['attack'][:, :, home_idx].mean()
    home_defense = trace.posterior['defense'][:, :, home_idx].mean()
    away_attack = trace.posterior['attack'][:, :, away_idx].mean()
    away_defense = trace.posterior['defense'][:, :, away_idx].mean()
    home_adv = trace.posterior['home_advantage'].mean()
    
    home_goals_exp = np.exp(home_attack + away_defense + home_adv)
    away_goals_exp = np.exp(away_attack + home_defense)
    
    return {
        'home_goals_exp': home_goals_exp,
        'away_goals_exp': away_goals_exp
    }

# Example prediction
print(predict_bayesian(trace, 'Arsenal', 'Chelsea'))