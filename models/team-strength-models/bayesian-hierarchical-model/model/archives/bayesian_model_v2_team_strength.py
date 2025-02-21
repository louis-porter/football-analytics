# build model
import psutil


def build_bayesian_model(home_teams, away_teams, home_goals, away_goals, home_xg, away_xg, dates, leagues):
    print("Building Bayesian model...")
    print(f"Dataset size: {len(home_teams)} matches")
    print(f"Time span: {dates.min()} to {dates.max()}")
    
    # get unique teams and leagues
    teams = sorted(list(set(home_teams) | set(away_teams))) # alphabetically sorts and de-dupes list of team names
    unique_leagues = sorted(list(set(leagues)))

    # sets index values for each team/league within a dict
    team_indices = {team: idx for idx, team in enumerate(teams)}
    league_indices = {league: idx for idx, league in enumerate(unique_leagues)}

    # convert date into time differences
    max_date = np.max(dates)
    time_diffs = (max_date - dates).dt.days

    # convert team names to index vals
    home_idx = [team_indices[team] for team in home_teams]
    away_idx = [team_indices[team] for team in away_teams]

    # Get league index for each team directly from the data
    home_league_idx = [league_indices[league] for league in leagues]
    away_league_idx = [league_indices[league] for league in leagues]
    
    # Create array of league indices for each team
    team_league_idx = np.zeros(len(teams), dtype=int)
    for team, idx in team_indices.items():
        # Find first occurrence of this team and use its league
        if team in home_teams:
            first_idx = list(home_teams).index(team)
            team_league_idx[idx] = home_league_idx[first_idx]
        else:
            first_idx = list(away_teams).index(team)
            team_league_idx[idx] = away_league_idx[first_idx]

    with pm.Model() as model:
        # league level parameters for league strengths
        league_attack_mu = pm.Normal("league_attack_mu", mu=0, sigma=0.5) # using a normal distribution to infer average league attack value
        league_attack_sigma = pm.HalfNormal("league_attack_sigma", sigma=0.5) # using a half normal dist to infer league attack spread, half normal as std must be positive
        league_defense_mu = pm.Normal("league_defense_mu", mu=0, sigma=0.5)
        league_defense_sigma = pm.HalfNormal("league_defense_sigma", sigma=0.5)

        # creating raw league strengths for all leagues EXCEPT Premier League
        premier_league_idx = league_indices["Premier League"]
        league_strength_raw = pm.Normal("league_strength_raw", mu=-0.5, sigma=0.3, shape=len(unique_leagues)-1) # setting mu to -0.5 as other leagues are expected to be weaker. shape = -1 as Premier league will be 0
        league_strength = pm.Deterministic( # deterministic variable as derived from other random variables (league strengths)
            "league_strength",
            pm.math.concatenate([
                league_strength_raw[:premier_league_idx],
                pm.math.zeros(1), # creating array that will have all league strengths with Premier league in the "middle" with 0
                league_strength_raw[premier_league_idx:]
            ])
        )

        # team strength initalisation
        attack_raw = pm.Normal("attack_raw", mu=0, sigma=1, shape=len(teams)) # initalising normal distribution for relative attacking strength with mean 0 and std of 1
        defense_raw = pm.Normal('defense_raw', mu=0, sigma=1, shape=len(teams))

        # scale team strengths by league
        attack = pm.Deterministic(
            "attack",
            attack_raw * league_attack_sigma + league_attack_mu + league_strength[team_league_idx] # combining raw team strength with league average/std and then penalising by league overall strength
        )
        defense = pm.Deterministic(
            "defense",
            defense_raw * league_defense_sigma + league_defense_mu + league_strength[team_league_idx]
        )

        # initalise time decay parameter
        decay_rate = pm.HalfNormal("decay_rate", sigma=1.5/365) # balanced prior for decay rate, divided by 365 to account for daily rate

        # initalise home advantage
        home_advantage = pm.Normal("home_advantage", mu=0.2, sigma=0.1, shape=len(unique_leagues)) # initalises home_adv to 0.2 and has std of 0.1 so val can extend or reduce that much. Also separating home adv by league

        # create time decay factor to apply to expected goals
        time_factor = pm.math.exp(-decay_rate * time_diffs)

        # expected goals parameter for both xG and goals, applied time decay
        home_theta = time_factor * pm.math.exp(attack[home_idx] - defense[away_idx] + home_advantage) # we use exponential so it's always positive and team strengths are multiplicative
        away_theta = time_factor * pm.math.exp(attack[away_idx] - defense[home_idx])

        # goals likelihood (poisson for actual goals)
        home_goals_like = pm.Poisson("home_goals", mu=home_theta, observed=home_goals) 
        away_goals_like = pm.Poisson("away_goals", mu=away_theta, observed=away_goals)

        # xG likelihood (gamma for expected goals)
        xg_alpha = pm.HalfNormal("xg_alpha", sigma=1.0) # shape parameter (must be positive hence half normal) - alpha shapes basic form of distribution
        home_xg_beta = xg_alpha / home_theta # beta is rate parameter - scales where that form sits on the axis
        away_xg_beta = xg_alpha / away_theta # we are setting the mean of the xg distribution to be equal to our team strength rating

        # add small constant to not allow 0s which breaks Gamma dist
        epsilon = 0.00001
        home_xg_adj = home_xg + epsilon
        away_xg_adj = away_xg + epsilon

        home_xg_like = pm.Gamma("home_xg", alpha=xg_alpha, beta=home_xg_beta, observed=home_xg_adj)
        away_xg_like = pm.Gamma("away_xg", alpha=xg_alpha, beta=away_xg_beta, observed=away_xg_adj)

        print("Model building completed!")

    return model, team_indices, league_indices

def fit_bayesian_model(model, draws=500):
    n_cores = min(4, multiprocessing.cpu_count() - 1)
    
    print(f"Starting model fitting with {n_cores} cores...")
    print(f"Planning {draws} draws with 500 tuning steps...")
    
    with model:
        trace = pm.sample(
            draws=draws,
            tune=500,
            chains=n_cores,
            cores=n_cores,
            progressbar=True,
            return_inferencedata=True,
            init='adapt_diag',
            target_accept=0.95,
            nuts={"max_treedepth": 15}  # Correctly nested NUTS parameter
        )
        
        # Print sampling diagnostics
        print("\nSampling Statistics:")
        print(f"Number of divergences: {trace.sample_stats.diverging.sum().values}")
        
        return trace
    
# Function to monitor memory usage
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Setup logging
logging.getLogger('pymc').setLevel(logging.INFO)



def get_league_strengths(trace, league_indices):
    leagues = list(league_indices.keys())
    league_strength_means = trace.posterior['league_strength'].mean(dim=['chain', 'draw']).values
    
    results = pd.DataFrame({
        'league': leagues,
        'league_strength': league_strength_means
    })
    
    return results.round(3).sort_values('league_strength', ascending=False)

def get_hierarchical_team_strengths(trace, team_indices, league_indices, team_leagues, current_teams):
    teams = list(team_indices.keys())
    attack_means = trace.posterior['attack'].mean(dim=['chain', 'draw']).values
    defense_means = trace.posterior['defense'].mean(dim=['chain', 'draw']).values
    home_adv = trace.posterior['home_advantage'].mean(dim=['chain', 'draw']).values
    
    # Get league strengths for reference
    league_strengths = get_league_strengths(trace, league_indices)
    
    results = pd.DataFrame({
        'team': teams,
        'league': [team_leagues.get(team, 'Unknown') for team in teams],  # Correctly map teams to leagues
        'attack_strength': attack_means,
        'defense_strength': defense_means,
        'overall_strength': (np.exp(attack_means - np.mean(defense_means)) - 
                           np.exp(np.mean(attack_means) - defense_means)),
        'home_advantage': home_adv
    })
    
    # Merge with league strengths
    results = results.merge(
        league_strengths,
        left_on='league',
        right_on='league',
        how='left'
    )
    
    # Filter current teams and sort
    results = (results[results['team'].isin(current_teams)]
              .round(3)
              .sort_values('overall_strength', ascending=False))
    
    return results, home_adv

def analyze_league_strengths(trace, league_indices, team_indices, team_leagues):
    # Get basic league strengths
    leagues = list(league_indices.keys())
    league_strength_means = trace.posterior['league_strength'].mean(dim=['chain', 'draw']).values
    
    # Get the posterior distributions for additional analysis
    league_attack_mu = trace.posterior['league_attack_mu'].mean(dim=['chain', 'draw']).values
    league_attack_sigma = trace.posterior['league_attack_sigma'].mean(dim=['chain', 'draw']).values
    league_defense_mu = trace.posterior['league_defense_mu'].mean(dim=['chain', 'draw']).values
    league_defense_sigma = trace.posterior['league_defense_sigma'].mean(dim=['chain', 'draw']).values
    
    # Calculate league-specific metrics
    detailed_results = []
    
    for league in leagues:
        league_idx = league_indices[league]
        league_teams = [team for team, l in team_leagues.items() if l == league]
        
        league_data = {
            'league': league,
            'base_strength': league_strength_means[league_idx],
            'attack_variation': league_attack_sigma,  # How much attack strength varies within the league
            'defense_variation': league_defense_sigma,  # How much defense strength varies within the league
            'num_teams': len(league_teams),
            'teams': ', '.join(sorted(league_teams)[:5]) + ('...' if len(league_teams) > 5 else '')
        }
        
        detailed_results.append(league_data)
    
    results_df = pd.DataFrame(detailed_results)
    
    # Calculate expected goals adjustment between leagues
    for idx, row in results_df.iterrows():
        base_league_strength = row['base_strength']
        results_df.loc[idx, 'expected_goals_vs_avg'] = np.exp(base_league_strength) - 1
    
    return results_df.round(3).sort_values('base_strength', ascending=False)
    