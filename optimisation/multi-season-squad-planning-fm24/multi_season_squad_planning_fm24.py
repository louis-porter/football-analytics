import pandas as pd
from io import StringIO
import pulp
import numpy as np
import re
from sklearn.linear_model import LinearRegression


def prepare_html_data(file_path, is_my_squad=False):
    """
    Read and preprocess Football Manager HTML data file into a pandas DataFrame
    
    Parameters:
    file_path (str): Path to the HTML file
    
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
        "is_lfb": r"D \(L\)$|WB \(L\)$|LWB$",  # Left full back/wing back only
        "is_rfb": r"D \(R\)$|WB \(R\)$|RWB$",  # Right full back/wing back only
        "is_cm": r"(?:DM|M) \([LR]*C[LR]*\)",   # Central midfielders
        "is_am": r"AM \([LRC]*\)|M\/AM \([LR]\)",  # Attacking midfielders
        "is_st": r"ST|ST \([LRC]\)"  # Strikers
    }
    
    df["Position"] = df["Position"].fillna("")
    for feature, pattern in position_features.items():
        df[feature] = df["Position"].str.contains(pattern, regex=True).astype(int)
    
    # Clean and process transfer values
    df[["Min Value", "Max Value"]] = df["Transfer Value"].str.split(" - ", expand=True)
    df["Min Value"] = df["Min Value"].str.strip().str.replace("£", "")
    df["Min Value"] = df["Min Value"].replace("Not for Sale", np.nan)
    df["Max Value"] = df["Max Value"].str.strip().str.replace("£", "")
    
    def convert_values(value):
        if pd.isna(value):
            return np.nan
        elif "M" in value:
            return float(value.replace("M", "")) * 1000000
        elif "K" in value:
            return float(value.replace("K", "")) * 1000
        else:
            return float(value)
    
    df["Min Value"] = df["Min Value"].apply(convert_values)
    df["Max Value"] = df["Max Value"].apply(convert_values)

    if is_my_squad:
        # For squad, use 25th percentile (conservative valuation)
        df["Avg Value"] = df["Min Value"] + (df["Max Value"] - df["Min Value"]) * 0.05
    else:
        # For transfer targets, use 75th percentile (higher valuation)
        df["Avg Value"] = df["Min Value"] + (df["Max Value"] - df["Min Value"]) * 0.9
    
    # Estimate missing values based on wages
    mask = ~df['Avg Value'].isna() & ~df['Wage'].isna()
    if mask.sum() >= 2:  # Need at least 2 points for regression
        # Prepare data for regression
        X = df.loc[mask, 'Wage'].values.reshape(-1, 1)
        y = df.loc[mask, 'Avg Value'].values
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Print relationship stats
        value_to_wage_ratio = y / X.flatten()
        median_ratio = np.median(value_to_wage_ratio)
        print(f"Median value-to-wage ratio: {median_ratio:.2f}")
        print(f"R² score: {model.score(X, y):.3f}")
        
        # Estimate missing values
        missing_mask = df['Avg Value'].isna() & ~df['Wage'].isna()
        if missing_mask.any():
            X_missing = df.loc[missing_mask, 'Wage'].values.reshape(-1, 1)
            df.loc[missing_mask, 'Avg Value'] = model.predict(X_missing)
            
            # Apply constraints to estimated values
            df.loc[missing_mask, 'Avg Value'] = df.loc[missing_mask, 'Avg Value'].clip(
                lower=df.loc[mask, 'Avg Value'].min(),
                upper=df.loc[mask, 'Avg Value'].max()
            )
            print(f"Estimated {missing_mask.sum()} missing values")
    
    # Handle any remaining NaN values using position-based medians
    remaining_nans = df['Avg Value'].isna().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} rows still have NaN values for Avg Value")
        for pos in position_features.keys():
            pos_mask = (df[pos] == 1) & (df['Avg Value'].isna())
            if pos_mask.any():
                median_value = df.loc[(df[pos] == 1) & (~df['Avg Value'].isna()), 'Avg Value'].median()
                df.loc[pos_mask, 'Avg Value'] = median_value
    
    return df

def prepare_csv_data(file_path):
    """
    Read and preprocess Football Manager squad data from CSV file into a pandas DataFrame
    
    Parameters:
    file_path (str): Path to the CSV file containing squad data
    
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
    
    # Create Position column for compatibility with rest of the code
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
    
    if all(col in df.columns for col in ['Min Value', 'Max Value']):
        df['Min Value'] = pd.to_numeric(df['Min Value'], errors='coerce')
        df['Max Value'] = pd.to_numeric(df['Max Value'], errors='coerce')
        df['Avg Value'] = df['Min Value'] + (df['Max Value'] - df['Min Value']) * 0.05
    elif 'Value' in df.columns:  # If there's a single Value column
        df['Avg Value'] = pd.to_numeric(df['Value'], errors='coerce')
    else:  # If no value columns exist, estimate from wages
        print("Warning: No value columns found. Estimating values from wages.")
        if 'Wage' in df.columns:
            # Use a simple multiplier as fallback (e.g., 52 weeks * 2 years)
            df['Avg Value'] = df['Wage'] * 52 * 2
        else:
            # If no wage data either, use a default value
            print("Warning: No wage data found. Using default values.")
            df['Avg Value'] = 1000000  # Default value, adjust as needed

    # Now handle missing values using wages if available
    if 'Wage' in df.columns:
        mask = ~df['Avg Value'].isna() & ~df['Wage'].isna()
        if mask.sum() >= 2:
            X = df.loc[mask, 'Wage'].values.reshape(-1, 1)
            y = df.loc[mask, 'Avg Value'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            value_to_wage_ratio = y / X.flatten()
            median_ratio = np.median(value_to_wage_ratio)
            print(f"Median value-to-wage ratio: {median_ratio:.2f}")
            print(f"R² score: {model.score(X, y):.3f}")
            
            missing_mask = df['Avg Value'].isna() & ~df['Wage'].isna()
            if missing_mask.any():
                X_missing = df.loc[missing_mask, 'Wage'].values.reshape(-1, 1)
                df.loc[missing_mask, 'Avg Value'] = model.predict(X_missing)
                
                df.loc[missing_mask, 'Avg Value'] = df.loc[missing_mask, 'Avg Value'].clip(
                    lower=df.loc[mask, 'Avg Value'].min(),
                    upper=df.loc[mask, 'Avg Value'].max()
                )
                print(f"Estimated {missing_mask.sum()} missing values")

    # Handle any remaining NaN values using position-based medians
    remaining_nans = df['Avg Value'].isna().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} rows still have NaN values for Avg Value")
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

df_transfer_targets = prepare_html_data(r"optimisation\multi-season-squad-planning-fm24\transfer_targets.html", is_my_squad=True)
df_palace_squad = prepare_csv_data(r"optimisation\multi-season-squad-planning-fm24\palace_squad.csv")


#df_palace_squad = df_palace_squad.drop(0)
print(df_palace_squad)
df_transfer_targets = df_transfer_targets.drop(0)

class PlayerAttributeWeights:
    """Define position-specific attribute weights"""
    
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
    
    def calculate_player_score(self, player_data, position):
        """
        [Previous calculate_player_score method remains exactly the same]
        """
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
        
        # Calculate weighted score
        score = sum(player_data[attr] * weight 
                   for attr, weight in weights.items())
        
        # Apply age factor
        age = player_data['Age']
        age_factor = 1.0
        if position.startswith('GK'):
            if age > 33:
                age_factor = 0.9
            elif age < 25:
                age_factor = 1.1
        else:
            if age < 21:
                age_factor = 1.07 
            elif age <= 23:
                age_factor = 1.05
            elif age >= 28:
                age_factor = 0.97  
            elif age >= 30:
                age_factor = 0.85
            elif age > 32:
                age_factor = 0.8
        
        return score * age_factor


    def optimise_transfers(self, current_squad, available_players, required_positions, max_transfers=2):
        """
        Optimize transfer decisions based on player attributes with core constraints only
        """
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
        
        # Protect third-choice goalkeeper
        current_gks = current_squad[current_squad['is_gk'] == 1].copy()
        if len(current_gks) >= 3:
            # Sort goalkeepers by score and protect the lowest-rated one
            worst_gk_idx = current_gks.nsmallest(1, 'player_score').index[0]
            self.problem += sell_vars[worst_gk_idx] == 0, "Protect_Third_GK"

        # Start with current squad total score
        base_score = current_squad['player_score'].sum()

        # Calculate score changes from transfers
        score_from_sales = pulp.lpSum(
            -sell_vars[i] * current_squad.loc[i, 'player_score']
            for i in current_squad.index
        )

        score_from_purchases = pulp.lpSum(
            buy_vars[i] * available_players.loc[i, 'player_score']
            for i in available_players.index
        )

        # Objective: Maximize final score
        self.problem += base_score + score_from_sales + score_from_purchases, "Total_Squad_Quality"

        # Core Constraints:
        
        # 1. Maximum transfers constraint
        # Core transfer limits first
        incoming_transfers = pulp.lpSum(buy_vars[i] for i in available_players.index)
        outgoing_transfers = pulp.lpSum(sell_vars[i] for i in current_squad.index)

        # Enforce max transfers as our primary constraint
        self.problem += incoming_transfers <= max_transfers, "Max_Incoming_Transfers"
        self.problem += outgoing_transfers <= max_transfers, "Max_Outgoing_Transfers"

        # 2. Squad size constraints (20-25 players)
        initial_squad_size = len(current_squad)
        self.problem += (initial_squad_size - 
                        pulp.lpSum(sell_vars[i] for i in current_squad.index) +
                        pulp.lpSum(buy_vars[i] for i in available_players.index) <= 25), "Max_Squad_Size"
        
        self.problem += (initial_squad_size - 
                        pulp.lpSum(sell_vars[i] for i in current_squad.index) +
                        pulp.lpSum(buy_vars[i] for i in available_players.index) >= 20), "Min_Squad_Size"

        # 3. Budget constraints
        sales_income = pulp.lpSum(
            sell_vars[i] * current_squad.loc[i, 'Avg Value']
            for i in current_squad.index
        )
        purchase_costs = pulp.lpSum(
            buy_vars[i] * available_players.loc[i, 'Avg Value']
            for i in available_players.index
        )
        self.problem += purchase_costs - sales_income <= self.transfer_budget, "Transfer_Budget"

        # 4. Wage budget constraints
        current_wages = pulp.lpSum(
            (1 - sell_vars[i]) * current_squad.loc[i, 'Wage']
            for i in current_squad.index
        )
        new_wages = pulp.lpSum(
            buy_vars[i] * available_players.loc[i, 'Wage']
            for i in available_players.index
        )
        self.problem += current_wages + new_wages <= self.wage_budget, "Wage_Budget"

        # 5. Position constraints
        # Position constraints
        for position_flag, (min_players, max_players) in required_positions.items():
            current_position_count = pulp.lpSum(
                1 - sell_vars[i]
                for i in current_squad[current_squad[position_flag] == 1].index
            )
            new_position_count = pulp.lpSum(
                buy_vars[i]
                for i in available_players[available_players[position_flag] == 1].index
            )
            
            # Get current number of players in this position
            current_pos_players = len(current_squad[current_squad[position_flag] == 1])
            
            if current_pos_players > max_players:
                # If we're over the maximum:
                # 1. Prevent any purchases for this position
                self.problem += new_position_count == 0, f"No_Buys_{position_flag}"
                
                # 2. Limit how many we can sell based on max_transfers
                min_allowed = current_pos_players - max_transfers
                self.problem += current_position_count >= min_allowed, f"Max_Reduction_{position_flag}"
            else:
                # Normal case - enforce minimum and maximum requirements
                self.problem += current_position_count + new_position_count >= min_players, f"Min_{position_flag}"
                self.problem += current_position_count + new_position_count <= max_players, f"Max_{position_flag}"

        # Solve optimization
        self.problem.solve()

        # Get results
        players_to_buy = available_players[
            [buy_vars[i].value() > 0.5 for i in available_players.index]
        ].copy()
        
        players_to_sell = current_squad[
            [sell_vars[i].value() > 0.5 for i in current_squad.index]
        ].copy()

        # Calculate metrics
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

# Add this after loading the squad data:
print_squad_composition(df_palace_squad, required_positions)

optimiser = FMTransferOptimizer(
    current_budget= 50000000,
    wage_budget=1600000
)

players_to_buy, players_to_sell, metrics = optimiser.optimise_transfers(
    current_squad=df_palace_squad,
    available_players=df_transfer_targets,
    required_positions=required_positions,
    max_transfers=5
)





def format_currency(value):
    """Format value to £XM or £XK format"""
    if abs(value) >= 1000000:
        return f"£{value/1000000:.1f}M"
    elif abs(value) >= 1000:
        return f"£{value/1000:.1f}K"
    else:
        return f"£{value:.0f}"

def print_transfer_results(players_to_buy, players_to_sell, metrics):
    # Print players to buy
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

    # Print players to sell
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

    # Print financial summary
    print("\nFINANCIAL SUMMARY:")
    print(f"Total spend: {format_currency(metrics['financial']['total_spend'])}")
    print(f"Total income: {format_currency(metrics['financial']['total_income'])}")
    print(f"Net spend: {format_currency(metrics['financial']['net_spend'])}")
    print(f"Weekly wage change: {format_currency(metrics['financial']['wage_change'])}/w")

# Use it
print_transfer_results(players_to_buy, players_to_sell, metrics)