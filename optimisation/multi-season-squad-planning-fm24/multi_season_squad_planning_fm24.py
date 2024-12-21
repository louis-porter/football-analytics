import pandas as pd
from io import StringIO
import pulp
import numpy as np
import re
from sklearn.linear_model import LinearRegression


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
    
    position_features = {
        "is_gk": r"GK",
        "is_cb": r"D \((?:C|LC|RC|RLC)\)",
        "is_lfb": r"D \(L\)$|WB \(L\)$|LWB$",
        "is_rfb": r"D \(R\)$|WB \(R\)$|RWB$",
        "is_cm": r"(?:DM|M) \([LR]*C[LR]*\)",
        "is_am": r"AM \([LRC]*\)|M\/AM \([LR]\)",
        "is_st": r"ST|ST \([LRC]\)"
    }
    
    df["Position"] = df["Position"].fillna("")
    for feature, pattern in position_features.items():
        df[feature] = df["Position"].str.contains(pattern, regex=True).astype(int)
    
    if 'Transfer Value' in df.columns:
        print("\nDebug Transfer Values:")
        print(df['Transfer Value'].head())
        
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

def prepare_csv_data(file_path):
    df = pd.read_csv(file_path)
    
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
    
    df['is_gk'] = (df['Act Pos'] == 'GK').astype(int)
    df['is_cb'] = (df['Act Pos'] == 'DC').astype(int)
    df['is_lfb'] = (df['Act Pos'] == 'LB').astype(int)
    df['is_rfb'] = (df['Act Pos'] == 'RB').astype(int)
    df['is_cm'] = (df['Act Pos'] == 'CM').astype(int)
    df['is_am'] = (df['Act Pos'] == 'AM').astype(int)
    df['is_st'] = (df['Act Pos'] == 'ST').astype(int)
    
    # Create Position column for compatibility
    position_mapping = {
        'GK': 'GK',
        'DC': 'D (C)',
        'LB': 'D (L)',
        'RB': 'D (R)',
        'CM': 'M (C)',
        'AM': 'AM (C)',
        'ST': 'ST'
    }
    df['Position'] = df['Act Pos'].apply(lambda x: position_mapping.get(x, 'Unknown'))
    
    if 'Transfer Value' in df.columns:
        print("\nDebug Transfer Values:")
        print(df['Transfer Value'].head())
        
        df['Transfer Value'] = df['Transfer Value'].fillna('')
        
        has_range = df['Transfer Value'].str.contains(' - ', na=False)
        
        range_df = df[has_range].copy()
        if not range_df.empty:
            range_df[['Min Value', 'Max Value']] = range_df['Transfer Value'].str.split(' - ', expand=True)
            range_df['Min Value'] = range_df['Min Value'].apply(convert_value_string)
            range_df['Max Value'] = range_df['Max Value'].apply(convert_value_string)
            # Use conservative 5% weight for club's own players
            range_df['Avg Value'] = range_df['Min Value'] + (range_df['Max Value'] - range_df['Min Value']) * 0.05
        
        single_df = df[~has_range].copy()
        if not single_df.empty:
            single_df['Avg Value'] = single_df['Transfer Value'].apply(convert_value_string)
            single_df['Min Value'] = single_df['Avg Value']
            single_df['Max Value'] = single_df['Avg Value']

        
        if not range_df.empty:
            df.loc[range_df.index, ['Min Value', 'Max Value', 'Avg Value']] = range_df[['Min Value', 'Max Value', 'Avg Value']]
        if not single_df.empty:
            df.loc[single_df.index, ['Min Value', 'Max Value', 'Avg Value']] = single_df[['Min Value', 'Max Value', 'Avg Value']]
    
    elif 'Value' in df.columns:
        df['Value'] = df['Value'].fillna('')
        df['Avg Value'] = df['Value'].apply(convert_value_string)
        df['Min Value'] = df['Avg Value']
        df['Max Value'] = df['Avg Value']
    
    if 'Avg Value' in df.columns:
        remaining_nans = df['Avg Value'].isna().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} rows have NaN values for Avg Value")
            for pos in ['is_gk', 'is_cb', 'is_lfb', 'is_rfb', 'is_cm', 'is_am', 'is_st']:
                pos_mask = (df[pos] == 1) & (df['Avg Value'].isna())
                if pos_mask.any():
                    median_value = df.loc[(df[pos] == 1) & (~df['Avg Value'].isna()), 'Avg Value'].median()
                    if pd.isna(median_value): 
                        median_value = df['Avg Value'].median() 
                    if pd.isna(median_value):  
                        median_value = 1000000  
                    df.loc[pos_mask, 'Avg Value'] = median_value
    
    if 'Av Rat' in df.columns:
        df['Av Rat'] = pd.to_numeric(df['Av Rat'].astype(str).str.replace(',', '.'), errors='coerce')

    return df

df_transfer_targets = prepare_html_data(r"optimisation\multi-season-squad-planning-fm24\transfer_targets_jun24.html")
df_palace_squad = prepare_csv_data(r"optimisation\multi-season-squad-planning-fm24\palace_squad_jun24.csv")



df_transfer_targets = df_transfer_targets.drop(0)


class PlayerAttributeWeights:
    """Define position-specific and DNA attribute weights"""
    
    @staticmethod
    def dna_weights():
        return {
        'Ant': 0.15, 'Tea': 0.15, 'Wor': 0.15, 'Sta': 0.15,
        'Mar': 0.05, 'Dec': 0.11, 'Det': 0.11, 'Acc': 0.08,
        'Agg': 0.05
    }

    @staticmethod
    def goalkeeper_weights():
        return {
            'Ref': 0.09, 'Pos': 0.09, '1v1': 0.08, 'Com': 0.08,
            'TRO': 0.07, 'Kic': 0.07, 'Agi': 0.07, 'Ant': 0.07,
            'Cmp': 0.07, 'Cnt': 0.07, 'Aer': 0.07, 'Fir': 0.06,
            'Han': 0.05, 'Pas': 0.04, 'Thr': 0.04
        }
    
    @staticmethod
    def center_back_weights():
        return {
            'Pos': 0.09, 'Mar': 0.09, 'Tck': 0.09, 'Acc': 0.08,
            'Hea': 0.08, 'Jum': 0.08, 'Str': 0.07, 'Ant': 0.07,
            'Bra': 0.07, 'Cnt': 0.07, 'Fir': 0.06, 'Pas': 0.06,
            'Dec': 0.05, 'Cmp': 0.05, 'Dri': 0.04, 'Wor': 0.04
        }
    
    @staticmethod
    def wingback_weights():
        return {
            'Sta': 0.08, 'Wor': 0.08, 'Acc': 0.08, 'Cro': 0.07,
            'Dri': 0.07, 'Tck': 0.07, 'Pos': 0.06, 'Tea': 0.06,
            'Pac': 0.06, 'OtB': 0.06, 'Tec': 0.06, 'Ant': 0.06,
            'Dec': 0.05, 'Agi': 0.05, 'Fir': 0.05, 'Pas': 0.05,
            'Bal': 0.05
        }
    
    @staticmethod
    def midfielder_weights():
        return {
            'Pas': 0.10, 'Tck': 0.10, 'Fir': 0.10, 'Tea': 0.10,
            'Ant': 0.09, 'Cmp': 0.09, 'Dec': 0.09, 'Wor': 0.08,
            'Sta': 0.07, 'Vis': 0.07, 'OtB': 0.06, 'Tec': 0.06,
            'Pos': 0.04
        }
    @staticmethod
    def winger_weights():
        return {
            'Dri': 0.09, 'Fir': 0.09, 'OtB': 0.09, 'Acc': 0.09,
            'Agi': 0.08, 'Tec': 0.08, 'Fin': 0.08, 'Wor': 0.08,
            'Ant': 0.07, 'Cmp': 0.07, 'Bal': 0.07, 'Pac': 0.07,
            'Fla': 0.03, 'Pas': 0.03, 'Lon': 0.03, 'Sta': 0.03
        }
    
    @staticmethod
    def striker_weights():
        return {
            'Ant': 0.09, 'Cmp': 0.09, 'Fin': 0.09, 'OtB': 0.09,
            'Acc': 0.08, 'Dec': 0.08, 'Fir': 0.08, 'Hea': 0.08,
            'Tec': 0.07, 'Str': 0.07, 'Agi': 0.07, 'Wor': 0.07,
            'Tea': 0.04, 'Bal': 0.04, 'Jum': 0.04
        }
    

class FMTransferOptimizer:
    def __init__(self, current_budget, wage_budget):
        """
        Initialize the transfer window optimizer
        
        Args:
            current_budget (float): Available transfer budget
            wage_budget (float): Available wage budget
        """
        self.transfer_budget = current_budget
        self.wage_budget = wage_budget
        self.attribute_weights = PlayerAttributeWeights()

    def get_rating_multiplier(self, avg_rating):
        """
        Calculate performance multiplier based on average rating, capped at 7.5
        
        Args:
            avg_rating (float): Player's average rating
            
        Returns:
            float: Performance multiplier
        """
        if pd.isna(avg_rating):
            return 1.0
            
        base_rating = 6.8
        max_rating = 7.5
        
        # Cap the rating at max_rating
        avg_rating = min(float(avg_rating), max_rating)
        
        if avg_rating >= base_rating:
            # For ratings above 6.8: Each 0.1 increase = 1% boost
            rating_diff = avg_rating - base_rating
            return 1 + (rating_diff * 0.1)
        else:
            # For ratings below 6.8: More aggressive penalty
            rating_diff = base_rating - avg_rating
            return 1 - (rating_diff * 0.5)  # Each 0.1 decrease = 5% penalty
    
    def calculate_position_specific_score(self, player_data, position_weights, age, is_goalkeeper=False):
        """Calculate score for a specific position including DNA and age factors"""
        dna_weights = self.attribute_weights.dna_weights()
        
        # Calculate base score from position-specific attributes (80%)
        attribute_score = sum(player_data[attr] * weight 
                            for attr, weight in position_weights.items()) * 0.8

        # Calculate DNA score (20%)
        dna_score = sum(player_data[attr] * weight 
                        for attr, weight in dna_weights.items()) * 0.2      

        # Combine scores
        base_score = attribute_score + dna_score
        
        # Apply age factor
        if is_goalkeeper:
            if age > 33:
                age_factor = 0.9
            elif age < 25:
                age_factor = 1.1
            else:
                age_factor = 1.0
        else:
            if age > 32:
                age_factor = 0.75
            elif age >= 30:
                age_factor = 0.87
            elif age >= 27:
                age_factor = 0.97
            elif age <= 24:
                age_factor = 1.05
            elif age < 21:
                age_factor = 1.07
            else:
                age_factor = 1.0
        
        score = base_score * age_factor
        
        # Apply rating multiplier if available
        if 'Av Rat' in player_data:
            rating_multiplier = self.get_rating_multiplier(player_data['Av Rat'])
            score *= rating_multiplier
        
        return score

    def calculate_position_scores(self, row):
        """Calculate scores for all positions a player can play"""
        position_scores = {}
        
        if row['is_gk'] == 1:
            position_scores['is_gk'] = self.calculate_position_specific_score(
                row,
                self.attribute_weights.goalkeeper_weights(),
                row['Age'],
                is_goalkeeper=True
            )
        
        if row['is_cb'] == 1:
            position_scores['is_cb'] = self.calculate_position_specific_score(
                row,
                self.attribute_weights.center_back_weights(),
                row['Age']
            )
        
        if row['is_lfb'] == 1:
            position_scores['is_lfb'] = self.calculate_position_specific_score(
                row,
                self.attribute_weights.wingback_weights(),
                row['Age']
            )
            
        if row['is_rfb'] == 1:
            position_scores['is_rfb'] = self.calculate_position_specific_score(
                row,
                self.attribute_weights.wingback_weights(),
                row['Age']
            )
            
        if row['is_cm'] == 1:
            position_scores['is_cm'] = self.calculate_position_specific_score(
                row,
                self.attribute_weights.midfielder_weights(),
                row['Age']
            )
            
        if row['is_am'] == 1:
            position_scores['is_am'] = self.calculate_position_specific_score(
                row,
                self.attribute_weights.winger_weights(),
                row['Age']
            )
            
        if row['is_st'] == 1:
            position_scores['is_st'] = self.calculate_position_specific_score(
                row,
                self.attribute_weights.striker_weights(),
                row['Age']
            )
        
        return position_scores
    
    def _calculate_metrics(self, current_squad, players_to_buy, players_to_sell):
        """Calculate improvement metrics after transfer decisions"""
        metrics = {
            'financial': {
                'total_spend': players_to_buy['Avg Value'].sum() if not players_to_buy.empty else 0,
                'total_income': players_to_sell['Avg Value'].sum() if not players_to_sell.empty else 0,
                'wage_change': (players_to_buy['Wage'].sum() if not players_to_buy.empty else 0) - 
                            (players_to_sell['Wage'].sum() if not players_to_sell.empty else 0)
            },
            'squad': {
                'avg_age_change': (players_to_buy['Age'].mean() if not players_to_buy.empty else 0) -
                                (players_to_sell['Age'].mean() if not players_to_sell.empty else 0),
                'total_score_change': (players_to_buy['position_score'].sum() if not players_to_buy.empty else 0) -
                                    (players_to_sell['position_score'].sum() if not players_to_sell.empty else 0)
            },
            'position_changes': {
                pos: {
                    'incoming': len(players_to_buy[players_to_buy['assigned_position'] == pos]) if not players_to_buy.empty else 0,
                    'outgoing': len(players_to_sell[players_to_sell['assigned_position'] == pos]) if not players_to_sell.empty else 0
                }
                for pos in set(current_squad['Position'].unique())
            }
        }
        
        metrics['financial']['net_spend'] = (metrics['financial']['total_spend'] - 
                                        metrics['financial']['total_income'])
        
        return metrics


    def optimise_transfers(self, current_squad, available_players, required_positions, 
                        max_transfers=2, locked_players=None, banned_players=None):

        locked_players = locked_players or []
        banned_players = banned_players or []
        
        current_squad_scores = current_squad.apply(self.calculate_position_scores, axis=1)
        available_players_scores = available_players.apply(self.calculate_position_scores, axis=1)
        
        self.problem = pulp.LpProblem("FM_Transfer_Optimization", pulp.LpMaximize)
        
        # Create position-specific binary variables for buying and selling
        buy_vars = {}
        sell_vars = {}
        
        # Create variables for each position a player can play
        for pos in required_positions.keys():
            for idx in available_players.index:
                if pos in available_players_scores[idx]:
                    buy_vars[(idx, pos)] = pulp.LpVariable(f"buy_{idx}_{pos}", cat='Binary')
            
            for idx in current_squad.index:
                if pos in current_squad_scores[idx]:
                    sell_vars[(idx, pos)] = pulp.LpVariable(f"sell_{idx}_{pos}", cat='Binary')
        
        # Ensure a player is only bought/sold for one position
        for idx in available_players.index:
            player_positions = [pos for pos in required_positions.keys() 
                            if pos in available_players_scores[idx]]
            if player_positions:
                self.problem += (
                    pulp.lpSum(buy_vars.get((idx, pos), 0) for pos in player_positions) <= 1,
                    f"Single_Position_Buy_{idx}"
                )
        
        for idx in current_squad.index:
            player_positions = [pos for pos in required_positions.keys() 
                            if pos in current_squad_scores[idx]]
            if player_positions:
                self.problem += (
                    pulp.lpSum(sell_vars.get((idx, pos), 0) for pos in player_positions) <= 1,
                    f"Single_Position_Sell_{idx}"
                )
        
        # Constraints for locked and banned players
        for player_name in locked_players:
            locked_indices = current_squad[current_squad['Name'] == player_name].index
            for idx in locked_indices:
                for pos in required_positions.keys():
                    if (idx, pos) in sell_vars:
                        self.problem += sell_vars[(idx, pos)] == 0, f"Lock_Player_{player_name}_{pos}"
        
        for player_name in banned_players:
            banned_indices = available_players[available_players['Name'] == player_name].index
            for idx in banned_indices:
                for pos in required_positions.keys():
                    if (idx, pos) in buy_vars:
                        self.problem += buy_vars[(idx, pos)] == 0, f"Ban_Player_{player_name}_{pos}"
        
        # Protect third-choice goalkeeper
        current_gks = current_squad[current_squad['is_gk'] == 1].copy()
        if len(current_gks) >= 3:
            gk_scores = []
            for idx, gk in current_gks.iterrows():
                gk_score = self.calculate_position_specific_score(
                    gk, 
                    self.attribute_weights.goalkeeper_weights(),
                    gk['Age'],
                    is_goalkeeper=True
                )
                gk_scores.append({'index': idx, 'score': gk_score})
            
            gk_scores_df = pd.DataFrame(gk_scores)
            worst_gk_idx = gk_scores_df.nsmallest(1, 'score')['index'].iloc[0]
            
            if (worst_gk_idx, 'is_gk') in sell_vars:
                self.problem += sell_vars[(worst_gk_idx, 'is_gk')] == 0, "Protect_Third_GK"
        
        # Calculate total squad quality using position-specific scores
        base_score = 0
        for scores in current_squad_scores:
            if scores:  
                base_score += max(scores.values())
            else:
                base_score += 0  

        score_from_sales = pulp.lpSum(
            -sell_vars.get((i, pos), 0) * current_squad_scores[i].get(pos, 0)
            for i in current_squad.index
            for pos in required_positions.keys()
            if pos in current_squad_scores[i]
        )

        score_from_purchases = pulp.lpSum(
            buy_vars.get((i, pos), 0) * available_players_scores[i].get(pos, 0)
            for i in available_players.index
            for pos in required_positions.keys()
            if pos in available_players_scores[i]
        )

        self.problem += base_score + score_from_sales + score_from_purchases, "Total_Squad_Quality"
        
        # Transfer limits
        incoming_transfers = pulp.lpSum(
            buy_vars.get((i, pos), 0)
            for i in available_players.index
            for pos in required_positions.keys()
            if (i, pos) in buy_vars
        )
        
        outgoing_transfers = pulp.lpSum(
            sell_vars.get((i, pos), 0)
            for i in current_squad.index
            for pos in required_positions.keys()
            if (i, pos) in sell_vars
        )
        
        self.problem += incoming_transfers <= max_transfers, "Max_Incoming_Transfers"
        self.problem += outgoing_transfers <= max_transfers, "Max_Outgoing_Transfers"
        
        # Squad size constraints
        initial_squad_size = len(current_squad)
        self.problem += (
            initial_squad_size - outgoing_transfers + incoming_transfers <= 25
        ), "Max_Squad_Size"
        
        self.problem += (
            initial_squad_size - outgoing_transfers + incoming_transfers >= 20
        ), "Min_Squad_Size"
        
        # Financial constraints
        sales_income = pulp.lpSum(
            sell_vars.get((i, pos), 0) * current_squad.loc[i, 'Avg Value']
            for i in current_squad.index
            for pos in required_positions.keys()
            if (i, pos) in sell_vars
        )
        
        purchase_costs = pulp.lpSum(
            buy_vars.get((i, pos), 0) * available_players.loc[i, 'Avg Value']
            for i in available_players.index
            for pos in required_positions.keys()
            if (i, pos) in buy_vars
        )
        
        self.problem += purchase_costs - sales_income <= self.transfer_budget, "Transfer_Budget"
        
        # Wage budget constraints
        wage_change = pulp.lpSum(
            buy_vars.get((i, pos), 0) * available_players.loc[i, 'Wage']
            for i in available_players.index
            for pos in required_positions.keys()
            if (i, pos) in buy_vars
        ) - pulp.lpSum(
            sell_vars.get((i, pos), 0) * current_squad.loc[i, 'Wage']
            for i in current_squad.index
            for pos in required_positions.keys()
            if (i, pos) in sell_vars
        )
        
        self.problem += wage_change <= self.wage_budget, "Wage_Budget"
        
        # Position-specific constraints
        for position_flag, (min_players, max_players) in required_positions.items():
            current_position_count = pulp.lpSum(
                1 - sell_vars.get((i, position_flag), 0)
                for i in current_squad[current_squad[position_flag] == 1].index
                if (i, position_flag) in sell_vars
            )
            
            new_position_count = pulp.lpSum(
                buy_vars.get((i, position_flag), 0)
                for i in available_players[available_players[position_flag] == 1].index
                if (i, position_flag) in buy_vars
            )
            
            current_pos_players = len(current_squad[current_squad[position_flag] == 1])
            
            if current_pos_players > max_players:
                self.problem += new_position_count == 0, f"No_Buys_{position_flag}"
                min_allowed = current_pos_players - max_transfers
                self.problem += current_position_count >= min_allowed, f"Max_Reduction_{position_flag}"
            else:
                self.problem += (
                    current_position_count + new_position_count >= min_players,
                    f"Min_{position_flag}"
                )
                self.problem += (
                    current_position_count + new_position_count <= max_players,
                    f"Max_{position_flag}"
                )
        
        # Solve the optimization problem
        self.problem.solve()
        
        # Extract results
        players_to_buy = []
        players_to_sell = []
        
        # Collect players to buy with their assigned positions
        for i in available_players.index:
            for pos in required_positions.keys():
                if (i, pos) in buy_vars and buy_vars[(i, pos)].value() > 0.5:
                    player_data = available_players.loc[i].copy()
                    player_data['assigned_position'] = pos
                    player_data['position_score'] = available_players_scores[i][pos]
                    players_to_buy.append(player_data)
        
        # Collect players to sell with their positions
        for i in current_squad.index:
            for pos in required_positions.keys():
                if (i, pos) in sell_vars and sell_vars[(i, pos)].value() > 0.5:
                    player_data = current_squad.loc[i].copy()
                    player_data['assigned_position'] = pos
                    player_data['position_score'] = current_squad_scores[i][pos]
                    players_to_sell.append(player_data)
        
        players_to_buy = pd.DataFrame(players_to_buy) if players_to_buy else pd.DataFrame()
        players_to_sell = pd.DataFrame(players_to_sell) if players_to_sell else pd.DataFrame()
        
        metrics = self._calculate_metrics(current_squad, players_to_buy, players_to_sell)
        return players_to_buy, players_to_sell, metrics

# Defining the players per positions required     
required_positions = {
    'is_gk': (3, 3),
    'is_cb': (5, 6),
    'is_lfb': (2, 2),
    'is_rfb': (2, 2),
    'is_cm': (3, 4),
    'is_am': (3, 5),
    'is_st': (2, 3)
}

def print_squad_composition(squad_df, required_positions):
    """Print current squad composition vs requirements"""
    print("\nSQUAD COMPOSITION ANALYSIS:")
    print("-" * 50)
    
    position_counts = {
        'is_gk': squad_df[squad_df['is_gk'] == 1].shape[0],
        'is_cb': squad_df[squad_df['is_cb'] == 1].shape[0],
        'is_lfb': squad_df[squad_df['is_lfb'] == 1].shape[0],
        'is_rfb': squad_df[squad_df['is_rfb'] == 1].shape[0],
        'is_cm': squad_df[squad_df['is_cm'] == 1].shape[0],
        'is_am': squad_df[squad_df['is_am'] == 1].shape[0],
        'is_st': squad_df[squad_df['is_st'] == 1].shape[0]
    }
    
    for pos, count in position_counts.items():
        min_req, max_req = required_positions[pos]
        status = "OK" if min_req <= count <= max_req else "NEED MORE" if count < min_req else "TOO MANY"
        print(f"{pos}: {count} players (Required: {min_req}-{max_req}) - {status}")

print_squad_composition(df_palace_squad, required_positions)

locked_players = [] 
banned_players = ["Konstantin Tyukavin", "Ignacio Miramón", "Lorenzo Pirola"]

optimiser = FMTransferOptimizer(
    current_budget= 22000000,
    wage_budget=66000
)

players_to_buy, players_to_sell, metrics = optimiser.optimise_transfers(
    current_squad=df_palace_squad,
    available_players=df_transfer_targets,
    required_positions=required_positions,
    max_transfers=2,
    locked_players=locked_players,
    banned_players=banned_players
)




def format_currency(value):
    """Format value to £XM or £XK format"""
    if abs(value) >= 1000000:
        return f"£{value/1000000:.1f}M"
    elif abs(value) >= 1000:
        return f"£{value/1000:.1f}K"
    else:
        return f"£{value:.0f}"

def print_transfer_results(players_to_buy, players_to_sell, metrics, current_squad):
    # Calculate current wage bill
    current_wage_bill = current_squad['Wage'].sum()
    
    # Calculate new wage bill
    new_wage_bill = current_wage_bill + metrics['financial']['wage_change']
    
    print("\nPLAYERS TO BUY:")
    buy_data = []
    for _, player in players_to_buy.iterrows():
        buy_data.append({
            'Name': player['Name'],
            'Age': player['Age'],
            'Position': player['Position'],
            'Value': format_currency(player['Avg Value']),
            'Wage': format_currency(player['Wage']),
            'Score': f"{player['position_score']:.2f}" 
        })
    if buy_data:
        print(pd.DataFrame(buy_data).to_string(index=False))
    else:
        print("None")
    
    print("\nPLAYERS TO SELL:")
    sell_data = []
    for _, player in players_to_sell.iterrows():
        sell_data.append({
            'Name': player['Name'],
            'Age': player['Age'],
            'Position': player['Position'],
            'Value': format_currency(player['Avg Value']),
            'Wage': format_currency(player['Wage']),
            'Score': f"{player['position_score']:.2f}" 
        })
    if sell_data:
        print(pd.DataFrame(sell_data).to_string(index=False))
    else:
        print("None")
    
    print("\nWAGE SUMMARY:")
    print(f"Current weekly wage bill: {format_currency(current_wage_bill)}/w")
    print(f"New weekly wage bill: {format_currency(new_wage_bill)}/w")
    print(f"Weekly wage change: {format_currency(metrics['financial']['wage_change'])}/w")
    
    print("\nFINANCIAL SUMMARY:")
    print(f"Total spend: {format_currency(metrics['financial']['total_spend'])}")
    print(f"Total income: {format_currency(metrics['financial']['total_income'])}")
    print(f"Net spend: {format_currency(metrics['financial']['net_spend'])}")

print_transfer_results(players_to_buy, players_to_sell, metrics, df_palace_squad)