import pandas as pd
from io import StringIO


def parse_html_file(file_path):
    """
    Read an HTML file and parse its table content into a pandas DataFrame
    
    Parameters:
    file_path (str): Path to the HTML file
    
    Returns:
    pandas.DataFrame: Parsed table data
    """
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Read HTML content using pandas
    df = pd.read_html(StringIO(html_content))[0]
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Replace empty strings with NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    
    # Convert wage column to numeric, removing '£' and 'p/w'
    if 'Wage' in df.columns:
        df['Wage'] = df['Wage'].str.replace('£', '').str.replace(' p/w', '').str.replace(',', '')
        df['Wage'] = pd.to_numeric(df['Wage'], errors='coerce')
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['Age', 'Com', 'Ecc', 'Pun', '1v1', 'Acc', 'Aer', 'Agg', 'Agi', 'Ant', 
                      'Bal', 'Bra', 'Cmd', 'Cnt', 'Cmp', 'Cro', 'Dec', 'Det', 'Dri', 'Fin',
                      'Fir', 'Fla', 'Han', 'Hea', 'Jum', 'Kic', 'Ldr', 'Lon', 'Mar', 'OtB',
                      'Pac', 'Pas', 'Pos', 'Ref', 'Sta', 'Str', 'Tck', 'Tea', 'Tec', 'Thr',
                      'TRO', 'Vis', 'Wor', 'Cor', 'Fre', 'L Th', 'Pen']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

df_palace_squad = parse_html_file(r"optimisation\multi-season-squad-planning-fm24\palace_squad.html")
df_transfer_targets = parse_html_file(r"optimisation\multi-season-squad-planning-fm24\transfer_targets.html")

print(df_palace_squad.head())



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
        self.problem = pulp.LpProblem("FM_Transfer_Optimization", pulp.LpMaximize)
        self.attribute_weights = PlayerAttributeWeights()
    
    def calculate_player_score(self, player_data, position):
        """
        Calculate player's score based on attributes weighted by position
        
        Args:
            player_data (pd.Series): Player's attributes
            position (str): Player's position
        
        Returns:
            float: Weighted attribute score
        """
        # Get relevant attribute weights based on position
        if position.str.startswith('GK'):
            weights = self.attribute_weights.goalkeeper_weights()
        elif position.str.contains(r"D \((C|LC|RC|RLC)\)", regex=True):
            weights = self.attribute_weights.center_back_weights()
        elif position.str.contains(r"D \([LR]+[C]?\)|WB", regex=True):
            weights = self.attribute_weights.fullback_weights()
        elif position.str.contains(r"M \([LR]*C[LR]*\)", regex=True):
            weights = self.attribute_weights.midfielder_weights()
        elif position.str.contains(r"(M|AM|M/AM) \([LR]+\)|AM \([LR]*C[LR]*\)", regex=True):
            weights = self.attribute_weights.winger_weights()
        elif position.str.contains(r"ST", regex=True):
            weights = self.attribute_weights.striker_weights()
        else:
            # Default to midfielder weights if position unclear
            weights = self.attribute_weights.midfielder_weights()
        
        # Calculate weighted score
        score = sum(player_data[attr] * weight 
                   for attr, weight in weights.items())
        
        # Apply age factor (prefer younger players when scores are close)
        age = player_data['Age']
        age_factor = 1.0
        if position.startswith('GK'):
            if age > 33:
                age_factor = 0.9
            elif age < 25:
                age_factor = 1.1
        else:
            if age < 21:
                age_factor = 1.2  
            elif age <= 23:
                age_factor = 1.1
            elif age >= 27:
                age_factor = 0.9
            elif age >= 30:
                age_factor = 0.8  
            elif age > 32:
                age_factor = 0.7
        
        return score * age_factor