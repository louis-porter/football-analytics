import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import StringIO

def convert_value_string(value_str):
    """Convert a value string like '£18.5M' or '£300K' to numeric"""
    if pd.isna(value_str) or value_str == '' or value_str == 'Not for Sale':
        return np.nan
    
    cleaned = str(value_str).replace('£', '').replace(',', '').strip()
    
    try:
        if 'M' in cleaned:
            number = float(cleaned.replace('M', '')) * 1_000_000
        elif 'K' in cleaned:
            number = float(cleaned.replace('K', '')) * 1_000
        else:
            number = float(cleaned)
        return number
    except (ValueError, AttributeError):
        return np.nan

def prepare_html_data(file_path, is_my_squad=False):
    """
    Read and preprocess Football Manager HTML data file into a pandas DataFrame
    
    Parameters:
    file_path (str): Path to the HTML file
    is_my_squad (bool): If True, uses more conservative valuation (closer to min value)
    
    Returns:
    pandas.DataFrame: Preprocessed DataFrame with cleaned attributes and encoded positions
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    df = pd.read_html(StringIO(html_content))[0]
    
    df.columns = df.columns.str.strip()
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    
    if 'Wage' in df.columns:
        df['Wage'] = df['Wage'].str.replace('£', '').str.replace(' p/w', '').str.replace(',', '')
        df['Wage'] = pd.to_numeric(df['Wage'], errors='coerce')
    
    numeric_columns = ['Age', 'Com', 'Ecc', 'Pun', '1v1', 'Acc', 'Aer', 'Agg', 'Agi', 'Ant', 
                      'Bal', 'Bra', 'Cmd', 'Cnt', 'Cmp', 'Cro', 'Dec', 'Det', 'Dri', 'Fin',
                      'Fir', 'Fla', 'Han', 'Hea', 'Jum', 'Kic', 'Ldr', 'Lon', 'Mar', 'OtB',
                      'Pac', 'Pas', 'Pos', 'Ref', 'Sta', 'Str', 'Tck', 'Tea', 'Tec', 'Thr',
                      'TRO', 'Vis', 'Wor', 'Cor', 'Fre', 'L Th', 'Pen']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # First, let's modify the position_features dictionary:
    position_features = {
        "is_gk": r"GK",
        "is_dc": r"D \((?:C|LC|RC|RLC)\)",
        "is_lcb": None,  # Will handle separately due to complex condition
        "is_rcb": None,  # Will handle separately due to complex condition
        "is_lb": r"D \(L\)$|WB \(L\)$|LWB$",
        "is_rb": r"D \(R\)$|WB \(R\)$|RWB$",
        "is_dm": r"DM",
        "is_mc": r"M \([LR]*C[LR]*\)",
        "is_amc": r"AM \(C\)",
        "is_aml": r"AM \(L\)|M\/AM \(L\)",
        "is_amr": r"AM \(R\)|M\/AM \(R\)",
        "is_st": r"ST|ST \([LRC]\)"
    }

    # Then modify the position assignment section in prepare_html_data:
    df["Position"] = df["Position"].fillna("")
    df["Left Foot"] = df["Left Foot"].fillna("")
    df["Right Foot"] = df["Right Foot"].fillna("")

    # First handle the regular position features
    for feature, pattern in position_features.items():
        if pattern is not None:  # Skip lcb and rcb as they need special handling
            df[feature] = df["Position"].str.contains(pattern, regex=True).astype(int)

    # Special handling for LCB and RCB
    # LCB: Must be able to play DC and have strong left foot
    df['is_lcb'] = ((df["Position"].str.contains(r"D \((?:C|LC|RC|RLC)\)", regex=True)) & 
                    (df["Left Foot"].str.contains("Strong", case=False, na=False))).astype(int)

    # RCB: Must be able to play DC and have strong right foot
    df['is_rcb'] = ((df["Position"].str.contains(r"D \((?:C|LC|RC|RLC)\)", regex=True)) & 
                    (df["Right Foot"].str.contains("Strong", case=False, na=False))).astype(int)
    
    if 'Transfer Value' in df.columns:       
        df['Transfer Value'] = df['Transfer Value'].fillna('')
    
        has_range = df['Transfer Value'].str.contains(' - ', na=False)
        
        range_df = df[has_range].copy()
        if not range_df.empty:
            range_df[['Min Value', 'Max Value']] = range_df['Transfer Value'].str.split(' - ', expand=True)
            range_df['Min Value'] = range_df['Min Value'].apply(convert_value_string)
            range_df['Max Value'] = range_df['Max Value'].apply(convert_value_string)
            # Use conservative upper range of value for incoming players
            range_df['Avg Value'] = range_df['Min Value'] + (range_df['Max Value'] - range_df['Min Value']) * 0.9
        
        single_df = df[~has_range].copy()
        if not single_df.empty:
            single_df['Avg Value'] = single_df['Transfer Value'].apply(convert_value_string)
            single_df['Min Value'] = single_df['Avg Value']
            single_df['Max Value'] = single_df['Avg Value']
        
        if not range_df.empty:
            df.loc[range_df.index, ['Min Value', 'Max Value', 'Avg Value']] = range_df[['Min Value', 'Max Value', 'Avg Value']]
        if not single_df.empty:
            df.loc[single_df.index, ['Min Value', 'Max Value', 'Avg Value']] = single_df[['Min Value', 'Max Value', 'Avg Value']]
            

    # Handle missing values using regression for transfer targets
    if not is_my_squad and 'Avg Value' in df.columns and df['Avg Value'].isna().any():
        feature_cols = [col for col in numeric_columns if col in df.columns]
        if 'Wage' in df.columns:
            feature_cols.append('Wage')
        
        # Add position dummy variables
        for pos in position_features.keys():
            if pos in df.columns:
                feature_cols.append(pos)
        
        # Remove rows with NaN in features
        features_df = df[feature_cols].copy()
        mask = ~features_df.isna().any(axis=1) & ~df['Avg Value'].isna()
        
        if mask.sum() >= 5:  # Only use regression if we have enough complete samples
            X = features_df.loc[mask]
            y = df.loc[mask, 'Avg Value']
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            r2_score = model.score(X, y)
            
            missing_mask = df['Avg Value'].isna() & ~features_df.isna().any(axis=1)
            if missing_mask.any():
                X_missing = features_df.loc[missing_mask]
                predicted_values = model.predict(X_missing)
                df.loc[missing_mask, 'Avg Value'] = predicted_values
                print(f"Estimated {missing_mask.sum()} missing values using regression")
    
    remaining_nans = df['Avg Value'].isna().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} rows still have NaN values for Avg Value")
        for pos in position_features.keys():
            pos_mask = (df[pos] == 1) & (df['Avg Value'].isna())
            if pos_mask.any():
                median_value = df.loc[(df[pos] == 1) & (~df['Avg Value'].isna()), 'Avg Value'].median()
                if pd.isna(median_value):  
                    median_value = df['Avg Value'].median()  
                if pd.isna(median_value):  
                    median_value = 1000000  
                df.loc[pos_mask, 'Avg Value'] = median_value
    
    # Ensure no NaN or negative values in final output
    df['Avg Value'] = df['Avg Value'].fillna(1000000).clip(lower=0)
    df['Min Value'] = df['Min Value'].fillna(df['Avg Value']).clip(lower=0)
    df['Max Value'] = df['Max Value'].fillna(df['Avg Value']).clip(lower=0)

    if 'Av Rat' in df.columns:
        df['Av Rat'] = pd.to_numeric(df['Av Rat'].astype(str).str.replace(',', '.'), errors='coerce')
    
    return df







class PlayerAttributeWeights:
    """Define position-specific and DNA attribute weights"""
    
    @staticmethod
    def dna_weights():
        return {
            'Agg': .25, 'Bra': .25, 'Det': .25, 'Wor': .25
        }

    @staticmethod
    def gk_weights():
        return {
            'Ref': 0.09, 'Pos': 0.09, '1v1': 0.08, 'Com': 0.08,
            'TRO': 0.07, 'Kic': 0.07, 'Agi': 0.07, 'Ant': 0.07,
            'Cmp': 0.07, 'Cnt': 0.07, 'Aer': 0.07, 'Fir': 0.06,
            'Han': 0.05, 'Pas': 0.04, 'Thr': 0.04
        }
    
    @staticmethod
    def ccb_weights():
        return {
            'Str': 0.0417, 'Jum': 0.0417, 'Cmp': 0.0417, 'Bra': 0.0417,
            'Pac': 0.0833, 'Mar': 0.0833, 'Tck': 0.0833, 'Hea': 0.0833,
            'Ant': 0.125, 'Cnt': 0.125, 'Dec': 0.125, 'Pos': 0.125
        }

    @staticmethod
    def lcb_weights():
        return {
            'Pos': 0.1, 'Dec': 0.1, 'Pac': 0.1, 'Str': 0.1,
            'Jum': 0.075, 'Tck': 0.075, 'Mar': 0.075, 'Hea': 0.075,
            'Dri': 0.05, 'Pas': 0.05, 'Agi': 0.05, 'Cmp': 0.05,
            'Ant': 0.025, 'Cnt': 0.025, 'Fir': 0.025, 'Sta': 0.025
        }  
    
    @staticmethod
    def rcb_weights():
        return {
            'Pos': 0.1, 'Dec': 0.1, 'Pac': 0.1, 'Str': 0.1,
            'Jum': 0.075, 'Tck': 0.075, 'Mar': 0.075, 'Hea': 0.075,
            'Dri': 0.05, 'Pas': 0.05, 'Agi': 0.05, 'Cmp': 0.05,
            'Ant': 0.025, 'Cnt': 0.025, 'Fir': 0.025, 'Sta': 0.025
        }  
    
    @staticmethod
    def rb_weights():
        return {
            'Acc': 0.1, 'Sta': 0.1, 'Wor': 0.1, 'Tea': 0.1,
            'Cro': 0.075, 'Mar': 0.075, 'Tck': 0.075, 'Pos': 0.075,
            'Dri': 0.05, 'Ant': 0.05, 'Pac': 0.05, 'Agi': 0.05,
            'Pas': 0.025, 'OtB': 0.025, 'Dec': 0.025, 'Cnt': 0.025
        }

    @staticmethod
    def lb_weights():
        return {
            'Acc': 0.1, 'Sta': 0.1, 'Wor': 0.1, 'Tea': 0.1,
            'Cro': 0.075, 'Mar': 0.075, 'Tck': 0.075, 'Pos': 0.075,
            'Dri': 0.05, 'Ant': 0.05, 'Pac': 0.05, 'Agi': 0.05,
            'Pas': 0.025, 'OtB': 0.025, 'Dec': 0.025, 'Cnt': 0.025
        } 
    
    @staticmethod
    def dm_weights():
        return {
            'Pos': 0.125, 'Cnt': 0.125, 'Ant': 0.125, 'Fir': 0.125,
            'Tck': 0.0833, 'Dec': 0.0833, 'Tea': 0.0833, 'Cmp': 0.0833,
            'Wor': 0.0417, 'Mar': 0.0417, 'Pas': 0.0417, 'Str': 0.0417
        }
    
    @staticmethod
    def cm_weights():
        return {
            'Sta': 0.1, 'Wor': 0.1, 'Tea': 0.1, 'OtB': 0.1,
            'Pas': 0.1, 'Tck': 0.1, 'Bal': 0.075, 'Ant': 0.075,
            'Dec': 0.05, 'Cmp': 0.05, 'Acc': 0.05, 'Fir': 0.05,
            'Str': 0.025, 'Dri': 0.025, 'Pac': 0.025, 'Tec': 0.025
        }
    
    @staticmethod
    def am_weights():
        return {
            'Vis': 0.111, 'Pas': 0.111, 'Tec': 0.111, 'Fir': 0.111,
            'Cmp': 0.089, 'Dec': 0.089, 'OtB': 0.083, 'Tea': 0.083,
            'Dri': 0.056, 'Acc': 0.056, 'Agi': 0.056, 'Fla': 0.056,
            'Ant': 0.028
        }
    
    @staticmethod
    def ss_weights():
        return {
            'OtB': 0.1, 'Acc': 0.1, 'Cmp': 0.1, 'Ant': 0.1,           
            'Dri': 0.075, 'Fin': 0.075, 'Fir': 0.075, 'Wor': 0.075,   
            'Dec': 0.05, 'Pas': 0.05, 'Tec': 0.05, 'Pac': 0.05,       
            'Agi': 0.025, 'Bal': 0.025, 'Cnt': 0.025, 'Sta': 0.025    
        }
    
    @staticmethod
    def rw_weights():
        return {
            'Dri': 0.121, 'Acc': 0.121, 'Agi': 0.121, 'Tec': 0.121,           
            'Cro': 0.121, 'Fir': 0.091, 'Bal': 0.091, 'Pac': 0.091,          
            'Sta': 0.061, 'Pas': 0.061, 'OtB': 0.061, 'Wor': 0.061          
        }
    
    @staticmethod
    def lw_weights():
        return {
            'Dri': 0.121, 'Acc': 0.121, 'Agi': 0.121, 'Tec': 0.121,           
            'Cro': 0.121, 'Fir': 0.091, 'Bal': 0.091, 'Pac': 0.091,          
            'Sta': 0.061, 'Pas': 0.061, 'OtB': 0.061, 'Wor': 0.061          
        }
    
    @staticmethod
    def lif_weights():
        return {
            'Dri': 0.121, 'Acc': 0.121, 'Agi': 0.121, 'Tec': 0.121,           
            'Fin': 0.121, 'OtB': 0.091, 'Ant': 0.091, 'Fir': 0.091,          
            'Sta': 0.061, 'Pac': 0.061, 'Bal': 0.061, 'Cmp': 0.061          
        }
    
    @staticmethod
    def rif_weights():
        return {
            'Dri': 0.121, 'Acc': 0.121, 'Agi': 0.121, 'Tec': 0.121,           
            'Fin': 0.121, 'OtB': 0.091, 'Ant': 0.091, 'Fir': 0.091,          
            'Sta': 0.061, 'Pac': 0.061, 'Bal': 0.061, 'Cmp': 0.061          
        }
    
    @staticmethod
    def dlf_weights():
        return {
            'Fir': 0.097, 'OtB': 0.097, 'Tea': 0.097, 'Dec': 0.097,
            'Cmp': 0.097, 'Ant': 0.073, 'Fin': 0.073, 'Str': 0.073,
            'Bal': 0.049, 'Pas': 0.049, 'Tec': 0.049, 'Vis': 0.049,
            'Jum': 0.097
        }
    
    @staticmethod
    def af_weights():
        return {
            'Fin': 0.097, 'Acc': 0.097, 'OtB': 0.097, 'Cmp': 0.097,           
            'Dri': 0.097, 'Fir': 0.073, 'Tec': 0.073, 'Ant': 0.073,          
            'Dec': 0.049, 'Wor': 0.049, 'Agi': 0.049, 'Bal': 0.049,          
            'Pac': 0.049, 'Sta': 0.049
        }
    



class PlayerRankingSystem:
    def __init__(self):
        self.attribute_weights = PlayerAttributeWeights()
    
    # def get_rating_multiplier(self, avg_rating):
    #     if pd.isna(avg_rating):
    #         return 1.0
            
    #     base_rating = 6.8
    #     max_rating = 7.5
    #     avg_rating = min(float(avg_rating), max_rating)
        
    #     if avg_rating >= base_rating:
    #         rating_diff = avg_rating - base_rating
    #         return 1 + (rating_diff * 0.1)
    #     else:
    #         rating_diff = base_rating - avg_rating
    #         return 1 - (rating_diff * 0.5)
    
    def calculate_position_score(self, player_data, position_weights, age, is_goalkeeper=False):
        """Calculate score for a specific position including DNA and age factors"""
        dna_weights = self.attribute_weights.dna_weights()
        
        attribute_score = sum(player_data[attr] * weight 
                            for attr, weight in position_weights.items()) * 0.85
        
        dna_score = sum(player_data[attr] * weight 
                    for attr, weight in dna_weights.items()) * 0.15
        
        base_score = attribute_score + dna_score
        
        # Position-specific age multipliers based on our data analysis
        age_multipliers = {
            'gk': {
                18: 1.15, 20: 1.11, 22: 1.08, 24: 1.06,
                26: 1.04, 28: 1.02, 30: 1.01, 32: 1.0 
            },
            'ccb': {  
                18: 1.11, 20: 1.08, 22: 1.06, 24: 1.04,
                26: 0.98, 28: 0.97, 30: 0.96, 32: 0.95
            },
            'fb': {  
                18: 1.13, 20: 1.09, 22: 1.06, 24: 1.03,
                26: 0.99, 28: 0.95, 30: 0.93, 32: 0.86
            },
            'dm': {  
                18: 1.12, 20: 1.09, 22: 1.07, 24: 1.05,
                26: 0.97, 28: 0.95, 30: 0.94, 32: 0.9
            },
            'cm': {  
                18: 1.11, 20: 1.08, 22: 1.05, 24: 1.03,
                26: 0.99, 28: 0.98, 30: 0.96, 32: 0.91
            },
            'am': {  
                18: 1.09, 20: 1.07, 22: 1.05, 24: 1.04,
                26: 0.97, 28: 0.96, 30: 0.95, 32: 0.9
            },
            'wing': {  
                18: 1.06, 20: 1.04, 22: 1.03, 24: 1.02,
                26: 0.99, 28: 0.98, 30: 0.95, 32: 0.85
            },
            'st': { 
                18: 1.14, 20: 1.12, 22: 1.1, 24: 1.05,
                26: 0.99, 28: 0.98, 30: 0.97, 32: 0.96
            }
        }
        
        # Determine position type from the weights method name
        if is_goalkeeper:
            pos_type = 'gk'
        elif any(role in str(position_weights) for role in ['ccb', 'lcb', 'rcb']):
            pos_type = 'ccb'
        elif any(role in str(position_weights) for role in ['lb', 'rb']):
            pos_type = 'fb'
        elif 'dm' in str(position_weights):
            pos_type = 'dm'
        elif 'cm' in str(position_weights):
            pos_type = 'cm'
        elif 'am' in str(position_weights):
            pos_type = 'am'
        elif any(role in str(position_weights) for role in ['lw', 'rw', 'lif', 'rif']):
            pos_type = 'wing'
        else:
            pos_type = 'st'
        
        # Get multiplier based on age
        multipliers = age_multipliers[pos_type]
        # Find the closest age bracket
        ages = sorted(multipliers.keys())
        if age <= ages[0]:
            age_factor = multipliers[ages[0]]
        elif age >= ages[-1]:
            age_factor = multipliers[ages[-1]]
        else:
            # Linear interpolation between age brackets
            for i in range(len(ages)-1):
                if ages[i] <= age <= ages[i+1]:
                    lower_age = ages[i]
                    upper_age = ages[i+1]
                    lower_mult = multipliers[lower_age]
                    upper_mult = multipliers[upper_age]
                    age_factor = lower_mult + (upper_mult - lower_mult) * (age - lower_age) / (upper_age - lower_age)
                    break
        
        score = base_score * age_factor
        return score

    def get_top_players_by_position(self, players_df, position_flag, top_n=5):
        """Get top N players for a specific position, evaluating all possible roles"""
        position_players = players_df[players_df[position_flag] == 1].copy()
        
        if position_players.empty:
            return pd.DataFrame()
        
        # Define which weight calculations to use for each position
        position_role_weights = {
            'is_gk': ['gk_weights'],
            'is_dc': ['ccb_weights'],
            'is_lcb': ['lcb_weights'],
            'is_rcb': ['rcb_weights'],
            'is_lb': ['lb_weights'],
            'is_rb': ['rb_weights'],
            'is_dm': ['dm_weights'],
            'is_mc': ['cm_weights'],
            'is_amc': ['am_weights', 'ss_weights'],
            'is_aml': ['lw_weights', 'lif_weights'],
            'is_amr': ['rw_weights', 'rif_weights'],
            'is_st': ['af_weights', 'dlf_weights']
        }
        
        # Get the relevant weight calculations for this position
        role_weights = position_role_weights[position_flag]
        
        # Calculate scores for each role
        all_scores = []
        for player_idx, player in position_players.iterrows():
            player_scores = []
            for weight_func in role_weights:
                position_method = getattr(self.attribute_weights, weight_func)()
                score = self.calculate_position_score(
                    player, 
                    position_method,
                    player['Age'],
                    is_goalkeeper=(position_flag == 'is_gk')
                )
                player_scores.append({
                    'Name': player['Name'],
                    'Age': player['Age'],
                    'Position': player['Position'],
                    'Wage': player['Wage'],
                    'Av Rat': player['Av Rat'] if 'Av Rat' in player else None,
                    'Role': weight_func.replace('_weights', '').upper(),
                    'Score': score
                })
            all_scores.extend(player_scores)
        
        # Convert to DataFrame and get top players
        scores_df = pd.DataFrame(all_scores)
        
        if scores_df.empty:
            return pd.DataFrame()
        
        # Sort by score and get top N for each role
        top_by_role = []
        for role in scores_df['Role'].unique():
            role_df = scores_df[scores_df['Role'] == role].nlargest(top_n, 'Score')
            top_by_role.append(role_df)
        
        return pd.concat(top_by_role)

    def rank_all_positions(self, players_df, top_n=5):
        """Get top N players for all positions"""
        positions = {
            'Goalkeeper': 'is_gk',
            'Central Defender': 'is_dc',
            'Right Central Defender': 'is_rcb',
            'Left Central Defender': 'is_lcb',
            'Left Back': 'is_lb',
            'Right Back': 'is_rb',
            'Defensive Midfielder': 'is_dm',
            'Central Midfielder': 'is_mc',
            'Attacking Midfielder (C)': 'is_amc',
            'Attacking Midfielder (L)': 'is_aml',
            'Attacking Midfielder (R)': 'is_amr',
            'Striker': 'is_st'
        }
        
        rankings = {}
        for position_name, flag in positions.items():
            rankings[position_name] = self.get_top_players_by_position(players_df, flag, top_n)
        
        return rankings

def format_rankings(rankings):
        """Format rankings for display with role-specific sections"""
        output = []
        for position, df in rankings.items():
            output.append(f"\n=== {position} ===")
            if not df.empty:
                for role in df['Role'].unique():
                    output.append(f"\nTop players as {role}:")
                    role_df = df[df['Role'] == role].copy()
                    role_df['Score'] = role_df['Score'].round(2)
                    role_df['Wage'] = role_df['Wage'].apply(lambda x: f"£{x:,.0f}" if pd.notna(x) else "N/A")
                    output.append(role_df[['Name', 'Age', 'Position', 'Wage', 'Score']].to_string(index=False))
        return "\n".join(output)


df_transfer_targets = prepare_html_data(r"C:\Users\Owner\dev\football-analytics\optimisation\5-player-position-scouting-fm24\playerts.html")


# Initialize the ranking system
ranker = PlayerRankingSystem()

# Get rankings for all positions
rankings = ranker.rank_all_positions(df_transfer_targets)

# Print formatted results
print(format_rankings(rankings))