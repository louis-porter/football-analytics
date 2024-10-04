import matplotlib.pyplot as plt
import numpy as np

premier_league_data = [
    {"team": "Newcastle", "xG": 1.6, "color": "#41393B"},
    {"team": "Man City", "xG": 0.9, "color": "#6CABDD"},
    {"team": "Chelsea", "xG": 4.2, "color": "#034694"},
    {"team": "Brighton", "xG": 1.1, "color": "#0057B8"},
    {"team": "Arsenal", "xG": 4.4, "color": "#EF0107"},
    {"team": "Leicester City", "xG": 0.3, "color": "#003090"},
    {"team": "Brentford", "xG": 0.4, "color": "#E30613"},
    {"team": "West Ham", "xG": 1.0, "color": "#7A263A"},
    {"team": "Everton", "xG": 0.9, "color": "#003399"},
    {"team": "Crystal Palace", "xG": 0.9, "color": "#1B458F"},
    {"team": "Forest", "xG": 0.8, "color": "#DD0000"},
    {"team": "Fulham", "xG": 1.3, "color": "#FFFFFF"},
    {"team": "Wolves", "xG": 0.6, "color": "#FDB913"},
    {"team": "Liverpool", "xG": 2.5, "color": "#C8102E"},
    {"team": "Ipswich", "xG": 1.2, "color": "#0000FF"},
    {"team": "Aston Villa", "xG": 0.8, "color": "#95BFE5"},
    {"team": "Man Utd", "xG": 1.0, "color": "#DA291C"},
    {"team": "Tottenham", "xG": 4.4, "color": "#132257"},
    {"team": "Bournemouth", "xG": 1.3, "color": "#DA291C"},
    {"team": "Southampton", "xG": 0.6, "color": "#D71920"}
]

# Sort the data by xG value in descending order
premier_league_data.sort(key=lambda x: x['xG'])

# Extract sorted data
teams = [team['team'] for team in premier_league_data]
xg_values = [team['xG'] for team in premier_league_data]
colors = [team['color'] for team in premier_league_data]

# Set up the plot
plt.figure(figsize=(14, 11))
plt.style.use('dark_background')
# Set the font to Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create horizontal bar chart
bars = plt.barh(teams, xg_values, color=colors)


# Customize the plot
plt.title('3 Teams Produced Over 4 Expected Goals Last Weekend', fontsize=16, color='white', loc="left", pad=20, fontdict={"weight":"bold"})
plt.text(0, 20.2, 'GW6 Premier League | Opta data via FBref', fontsize=12, color='white', alpha=0.8)
plt.ylabel('Teams', fontsize=12, color='white')
plt.xlabel('xG', fontsize=12, color='white')

# Customize axis colors
plt.gca().spines[['top', 'right', 'left', "bottom"]].set_visible(False)
plt.gca().tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.gca().tick_params(axis='y', colors='white')

# Add value labels on the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.03, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
             ha='left', va='center', fontsize=10, fontweight='bold', color='white')

# Set background color to black
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()