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
    
    # Clean string and extract number
    cleaned = str(value_str).replace('£', '').replace(',', '').strip()
    
    try:
        # Handle K/M suffix
        if 'M' in cleaned:
            # Remove M and convert to millions
            number = float(cleaned.replace('M', '')) * 1_000_000
        elif 'K' in cleaned:
            # Remove K and convert to thousands
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
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Read HTML content using pandas
    df = pd.read_html(StringIO(html_content))[0]
    
    # Clean up column names and empty values
    df.columns = df.columns.str.strip()
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    
    # Convert wage column to numeric
    if 'Wage' in df.columns:
        df['Wage'] = df['Wage'].str.replace('£', '').str.replace(' p/w', '').str.replace(',', '')
        df['Wage'] = pd.to_numeric(df['Wage'], errors='coerce')
    
    # Convert numeric attribute columns
    numeric_columns = ['Age', 'Com', 'Ecc', 'Pun', '1v1', 'Acc', 'Aer', 'Agg', 'Agi', 'Ant', 
                      'Bal', 'Bra', 'Cmd', 'Cnt', 'Cmp', 'Cro', 'Dec', 'Det', 'Dri', 'Fin',
                      'Fir', 'Fla', 'Han', 'Hea', 'Jum', 'Kic', 'Ldr', 'Lon', 'Mar', 'OtB',
                      'Pac', 'Pas', 'Pos', 'Ref', 'Sta', 'Str', 'Tck', 'Tea', 'Tec', 'Thr',
                      'TRO', 'Vis', 'Wor', 'Cor', 'Fre', 'L Th', 'Pen']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Encode player positions
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
    
    # Handle transfer values
    if 'Transfer Value' in df.columns:
        print("\nDebug Transfer Values:")
        print(df['Transfer Value'].head())
        
        # First convert all values to numeric regardless of whether they're ranges
        df['Transfer Value'] = df['Transfer Value'].fillna('')
        
        # Identify which rows contain ranges
        has_range = df['Transfer Value'].str.contains(' - ', na=False)
        
        # Handle range values
        range_df = df[has_range].copy()
        if not range_df.empty:
            range_df[['Min Value', 'Max Value']] = range_df['Transfer Value'].str.split(' - ', expand=True)
            range_df['Min Value'] = range_df['Min Value'].apply(convert_value_string)
            range_df['Max Value'] = range_df['Max Value'].apply(convert_value_string)
            range_df['Avg Value'] = range_df['Min Value'] + (range_df['Max Value'] - range_df['Min Value']) * 0.9
        
        # Handle single values
        single_df = df[~has_range].copy()
        if not single_df.empty:
            single_df['Avg Value'] = single_df['Transfer Value'].apply(convert_value_string)
            single_df['Min Value'] = single_df['Avg Value']
            single_df['Max Value'] = single_df['Avg Value']
        
        # Combine the results back
        if not range_df.empty:
            df.loc[range_df.index, ['Min Value', 'Max Value', 'Avg Value']] = range_df[['Min Value', 'Max Value', 'Avg Value']]
        if not single_df.empty:
            df.loc[single_df.index, ['Min Value', 'Max Value', 'Avg Value']] = single_df[['Min Value', 'Max Value', 'Avg Value']]
            

    # Handle missing values using regression for transfer targets
    if not is_my_squad and 'Avg Value' in df.columns and df['Avg Value'].isna().any():
        # Prepare features for regression
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
            # Prepare data for regression
            X = features_df.loc[mask]
            y = df.loc[mask, 'Avg Value']
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Print relationship stats
            r2_score = model.score(X, y)
            print(f"Value estimation R² score: {r2_score:.3f}")
            
            # Predict missing values
            missing_mask = df['Avg Value'].isna() & ~features_df.isna().any(axis=1)
            if missing_mask.any():
                X_missing = features_df.loc[missing_mask]
                predicted_values = model.predict(X_missing)
                df.loc[missing_mask, 'Avg Value'] = predicted_values
                print(f"Estimated {missing_mask.sum()} missing values using regression")
    
    # Fill remaining missing values using position-based medians
    remaining_nans = df['Avg Value'].isna().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} rows still have NaN values for Avg Value")
        for pos in position_features.keys():
            pos_mask = (df[pos] == 1) & (df['Avg Value'].isna())
            if pos_mask.any():
                median_value = df.loc[(df[pos] == 1) & (~df['Avg Value'].isna()), 'Avg Value'].median()
                if pd.isna(median_value):  # If no median available for position
                    median_value = df['Avg Value'].median()  # Use overall median
                if pd.isna(median_value):  # If still no median available
                    median_value = 1000000  # Default value
                df.loc[pos_mask, 'Avg Value'] = median_value
    
    # Ensure no NaN or negative values in final output
    df['Avg Value'] = df['Avg Value'].fillna(1000000).clip(lower=0)
    df['Min Value'] = df['Min Value'].fillna(df['Avg Value']).clip(lower=0)
    df['Max Value'] = df['Max Value'].fillna(df['Avg Value']).clip(lower=0)
    
    return df

def prepare_csv_data(file_path):
    """
    Read and preprocess Football Manager CSV data file into a pandas DataFrame
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Preprocessed DataFrame with cleaned attributes and encoded positions
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert wage column to numeric if present
    if 'Wage' in df.columns:
        df['Wage'] = df['Wage'].str.replace('£', '').str.replace(' p/w', '').str.replace(',', '')
        df['Wage'] = pd.to_numeric(df['Wage'], errors='coerce')
    
    # Convert numeric attribute columns
    numeric_columns = ['Age', 'Com', 'Ecc', 'Pun', '1v1', 'Acc', 'Aer', 'Agg', 'Agi', 'Ant', 
                      'Bal', 'Bra', 'Cmd', 'Cnt', 'Cmp', 'Cro', 'Dec', 'Det', 'Dri', 'Fin',
                      'Fir', 'Fla', 'Han', 'Hea', 'Jum', 'Kic', 'Ldr', 'Lon', 'Mar', 'OtB',
                      'Pac', 'Pas', 'Pos', 'Ref', 'Sta', 'Str', 'Tck', 'Tea', 'Tec', 'Thr',
                      'TRO', 'Vis', 'Wor', 'Cor', 'Fre', 'L Th', 'Pen']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Map Act Pos to position flags
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
    
    # Handle Transfer Value
    if 'Transfer Value' in df.columns:
        print("\nDebug Transfer Values:")
        print(df['Transfer Value'].head())
        
        # First convert all values to numeric regardless of whether they're ranges
        df['Transfer Value'] = df['Transfer Value'].fillna('')
        
        # Identify which rows contain ranges
        has_range = df['Transfer Value'].str.contains(' - ', na=False)
        
        # Handle range values
        range_df = df[has_range].copy()
        if not range_df.empty:
            range_df[['Min Value', 'Max Value']] = range_df['Transfer Value'].str.split(' - ', expand=True)
            range_df['Min Value'] = range_df['Min Value'].apply(convert_value_string)
            range_df['Max Value'] = range_df['Max Value'].apply(convert_value_string)
            # Use conservative 5% weight for club's own players
            range_df['Avg Value'] = range_df['Min Value'] + (range_df['Max Value'] - range_df['Min Value']) * 0.05
        
        # Handle single values
        single_df = df[~has_range].copy()
        if not single_df.empty:
            single_df['Avg Value'] = single_df['Transfer Value'].apply(convert_value_string)
            single_df['Min Value'] = single_df['Avg Value']
            single_df['Max Value'] = single_df['Avg Value']

        
        # Combine the results back
        if not range_df.empty:
            df.loc[range_df.index, ['Min Value', 'Max Value', 'Avg Value']] = range_df[['Min Value', 'Max Value', 'Avg Value']]
        if not single_df.empty:
            df.loc[single_df.index, ['Min Value', 'Max Value', 'Avg Value']] = single_df[['Min Value', 'Max Value', 'Avg Value']]
    
    elif 'Value' in df.columns:
        # If there's a single Value column, use the same improved logic
        df['Value'] = df['Value'].fillna('')
        df['Avg Value'] = df['Value'].apply(convert_value_string)
        df['Min Value'] = df['Avg Value']
        df['Max Value'] = df['Avg Value']
    
    # Handle missing values
    if 'Avg Value' in df.columns:
        remaining_nans = df['Avg Value'].isna().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} rows have NaN values for Avg Value")
            for pos in ['is_gk', 'is_cb', 'is_lfb', 'is_rfb', 'is_cm', 'is_am', 'is_st']:
                pos_mask = (df[pos] == 1) & (df['Avg Value'].isna())
                if pos_mask.any():
                    median_value = df.loc[(df[pos] == 1) & (~df['Avg Value'].isna()), 'Avg Value'].median()
                    if pd.isna(median_value):  # If no median available for position
                        median_value = df['Avg Value'].median()  # Use overall median
                    if pd.isna(median_value):  # If still no median available
                        median_value = 1000000  # Default value
                    df.loc[pos_mask, 'Avg Value'] = median_value
    
    return df

df_transfer_targets = prepare_html_data(r"optimisation\multi-season-squad-planning-fm24\transfer_targets_jan24.html", is_my_squad=True)
df_palace_squad = prepare_csv_data(r"optimisation\multi-season-squad-planning-fm24\palace_squad_jan24.csv")


#df_palace_squad = df_palace_squad.drop(0)
print(df_palace_squad[["Name", "Transfer Value", "Avg Value"]])

df_transfer_targets = df_transfer_targets.drop(0)

print(df_transfer_targets[df_transfer_targets['Name'] == 'Fran Pérez'][["Name", "Transfer Value"]])

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
        Calculate performance multiplier based on average rating
        
        Args:
            avg_rating (float): Player's average rating
            
        Returns:
            float: Performance multiplier
        """
        if pd.isna(avg_rating):
            return 1.0
            
        base_rating = 6.8
        
        if avg_rating >= base_rating:
            # For ratings above 6.8: Each 0.1 increase = 2.5% boost
            rating_diff = avg_rating - base_rating
            return 1 + (rating_diff * 0.25)
        else:
            # For ratings below 6.8: More aggressive penalty
            rating_diff = base_rating - avg_rating
            return 1 - (rating_diff * 0.5)  # Each 0.1 decrease = 5% penalty
    
    def calculate_player_score(self, player_data, position, is_transfer_target=False):
        """
        Calculate player score, applying rating multiplier only for transfer targets
        
        Args:
            player_data: Row of player attributes
            position: Player's position
            is_transfer_target: Whether this is a potential transfer target
        """
        dna_weights = self.attribute_weights.dna_weights()

        # Get relevant attribute weights based on position
        if re.search(r"GK", position):
            weights = self.attribute_weights.goalkeeper_weights()
        elif re.search(r"D \((C|LC|RC|RLC)\)", position):
            weights = self.attribute_weights.center_back_weights()
        elif re.search(r"D \(L\)$|D \(R\)$|WB \([LR]\)$|[LR]WB$", position):
            weights = self.attribute_weights.wingback_weights()
        elif re.search(r"[DMA]M \([LR]*C[LR]*\)", position):
            weights = self.attribute_weights.midfielder_weights()
        elif re.search(r"(M|AM|M/AM) \([LR]+\)|AM \([LR]*C[LR]*\)", position): 
            weights = self.attribute_weights.winger_weights()
        elif re.search(r"ST", position):
            weights = self.attribute_weights.striker_weights()
        else:
            weights = self.attribute_weights.midfielder_weights()
        
        # Calculate base score from regular attributes (80%)
        attribute_score = sum(player_data[attr] * weight 
                            for attr, weight in weights.items()) * 0.8

        # Calculate DNA score (20%)
        dna_score = sum(player_data[attr] * weight 
                        for attr, weight in dna_weights.items()) * 0.2      

        # Combine scores
        base_score = attribute_score + dna_score
        
        # Debug age factor
        age = player_data['Age']
        age_factor = 1.0
        
        if player_data['is_gk'] == 1:
            if age > 33:
                age_factor = 0.9
            elif age < 25:
                age_factor = 1.1
        else:
            if age > 32:
                age_factor = 0.7
            elif age >= 30:
                age_factor = 0.87
            elif age >= 27:
                age_factor = 0.97
            elif age <= 24:
                age_factor = 1.05
            elif age < 21:
                age_factor = 1.07
        
        score = base_score * age_factor
        
        return score


    def optimise_transfers(self, current_squad, available_players, required_positions, 
                         max_transfers=2, locked_players=None, banned_players=None):
        """
        Optimize transfer decisions based on player attributes
        
        Args:
            current_squad (pd.DataFrame): Current squad data
            available_players (pd.DataFrame): Available transfer targets
            required_positions (dict): Position requirements {position: (min, max)}
            max_transfers (int): Maximum number of transfers allowed
            locked_players (list): List of player names that cannot be sold
            banned_players (list): List of player names that cannot be bought
        """
        locked_players = locked_players or []
        banned_players = banned_players or []
        
        self.problem = pulp.LpProblem("FM_Transfer_Optimization", pulp.LpMaximize)

        # Calculate player scores
        current_squad['player_score'] = current_squad.apply(
            lambda x: self.calculate_player_score(x, x["Position"]),
            axis=1
        )
        available_players['player_score'] = available_players.apply(
            lambda x: self.calculate_player_score(x, x["Position"]),
            axis=1
        )

        # Create binary decision variables
        buy_vars = pulp.LpVariable.dicts("buy",
                                        ((i) for i in available_players.index),
                                        cat='Binary')
        
        sell_vars = pulp.LpVariable.dicts("sell",
                                        ((i) for i in current_squad.index),
                                        cat='Binary')
        
        # constraints for locked and banned players
        for player_name in locked_players:
            locked_indices = current_squad[current_squad['Name'] == player_name].index
            for idx in locked_indices:
                self.problem += sell_vars[idx] == 0, f"Lock_Player_{player_name}"
                
        for player_name in banned_players:
            banned_indices = available_players[available_players['Name'] == player_name].index
            for idx in banned_indices:
                self.problem += buy_vars[idx] == 0, f"Ban_Player_{player_name}"

        # Protect third-choice goalkeeper
        current_gks = current_squad[current_squad['is_gk'] == 1].copy()
        if len(current_gks) >= 3:
            worst_gk_idx = current_gks.nsmallest(1, 'player_score').index[0]
            self.problem += sell_vars[worst_gk_idx] == 0, "Protect_Third_GK"

        base_score = current_squad['player_score'].sum()

        score_from_sales = pulp.lpSum(
            -sell_vars[i] * current_squad.loc[i, 'player_score']
            for i in current_squad.index
        )

        score_from_purchases = pulp.lpSum(
            buy_vars[i] * available_players.loc[i, 'player_score']
            for i in available_players.index
        )

        self.problem += base_score + score_from_sales + score_from_purchases, "Total_Squad_Quality"

        incoming_transfers = pulp.lpSum(buy_vars[i] for i in available_players.index)
        outgoing_transfers = pulp.lpSum(sell_vars[i] for i in current_squad.index)

        self.problem += incoming_transfers <= max_transfers, "Max_Incoming_Transfers"
        self.problem += outgoing_transfers <= max_transfers, "Max_Outgoing_Transfers"

        initial_squad_size = len(current_squad)
        self.problem += (initial_squad_size - 
                        pulp.lpSum(sell_vars[i] for i in current_squad.index) +
                        pulp.lpSum(buy_vars[i] for i in available_players.index) <= 25), "Max_Squad_Size"
        
        self.problem += (initial_squad_size - 
                        pulp.lpSum(sell_vars[i] for i in current_squad.index) +
                        pulp.lpSum(buy_vars[i] for i in available_players.index) >= 20), "Min_Squad_Size"

        sales_income = pulp.lpSum(
            sell_vars[i] * current_squad.loc[i, 'Avg Value']
            for i in current_squad.index
        )
        purchase_costs = pulp.lpSum(
            buy_vars[i] * available_players.loc[i, 'Avg Value']
            for i in available_players.index
        )
        self.problem += purchase_costs - sales_income <= self.transfer_budget, "Transfer_Budget"

        wage_change = pulp.lpSum(
            buy_vars[i] * available_players.loc[i, 'Wage']
            for i in available_players.index
        ) - pulp.lpSum(
            sell_vars[i] * current_squad.loc[i, 'Wage']
            for i in current_squad.index
        )
        
        # Net wage change cannot exceed spare wage budget
        self.problem += wage_change <= self.wage_budget, "Wage_Budget"


        for position_flag, (min_players, max_players) in required_positions.items():
            current_position_count = pulp.lpSum(
                1 - sell_vars[i]
                for i in current_squad[current_squad[position_flag] == 1].index
            )
            new_position_count = pulp.lpSum(
                buy_vars[i]
                for i in available_players[available_players[position_flag] == 1].index
            )
            
            current_pos_players = len(current_squad[current_squad[position_flag] == 1])
            
            if current_pos_players > max_players:
                self.problem += new_position_count == 0, f"No_Buys_{position_flag}"
                min_allowed = current_pos_players - max_transfers
                self.problem += current_position_count >= min_allowed, f"Max_Reduction_{position_flag}"
            else:
                self.problem += current_position_count + new_position_count >= min_players, f"Min_{position_flag}"
                self.problem += current_position_count + new_position_count <= max_players, f"Max_{position_flag}"

        self.problem.solve()

        players_to_buy = available_players[
            [buy_vars[i].value() > 0.5 for i in available_players.index]
        ].copy()
        
        players_to_sell = current_squad[
            [sell_vars[i].value() > 0.5 for i in current_squad.index]
        ].copy()

        metrics = self._calculate_metrics(current_squad, players_to_buy, players_to_sell)
        return players_to_buy, players_to_sell, metrics

    def _calculate_metrics(self, current_squad, players_to_buy, players_to_sell):
        """Calculate improvement metrics after transfer decisions"""
        metrics = {
            'financial': {
                'total_spend': players_to_buy['Avg Value'].sum(),
                'total_income': players_to_sell['Avg Value'].sum(),
                'net_spend': players_to_buy['Avg Value'].sum() - players_to_sell['Avg Value'].sum(),
                'wage_change': players_to_buy['Wage'].sum() - players_to_sell['Wage'].sum()
            },
            'squad': {
                'avg_age_change': (players_to_buy['Age'].mean() if not players_to_buy.empty else 0) -
                                (players_to_sell['Age'].mean() if not players_to_sell.empty else 0),
                'total_score_change': (players_to_buy['player_score'].sum() if not players_to_buy.empty else 0) -
                                    (players_to_sell['player_score'].sum() if not players_to_sell.empty else 0)
            },
            'position_changes': {
                pos: {
                    'incoming': len(players_to_buy[players_to_buy['Position'] == pos]),
                    'outgoing': len(players_to_sell[players_to_sell['Position'] == pos])
                }
                for pos in set(current_squad['Position'].unique())
            }
        }
        return metrics
    
required_positions = {
    'is_gk': (3, 3),
    'is_cb': (5, 6),
    'is_lfb': (2, 2),
    'is_rfb': (2, 2),
    'is_cm': (3, 4),
    'is_am': (3, 4),
    'is_st': (2, 2)
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

locked_players = ["Ismaïla Sarr", "Daichi Kamada", "Jefferson Lerma"] 
banned_players = ["Ante Budimir", "Diego Moreno", "Nathan Ngoumou"]

optimiser = FMTransferOptimizer(
    current_budget= 600000,
    wage_budget=16000
)

players_to_buy, players_to_sell, metrics = optimiser.optimise_transfers(
    current_squad=df_palace_squad,
    available_players=df_transfer_targets,
    required_positions=required_positions,
    max_transfers=1,
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
            'Score': f"{player['player_score']:.2f}"
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
            'Score': f"{player['player_score']:.2f}"
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