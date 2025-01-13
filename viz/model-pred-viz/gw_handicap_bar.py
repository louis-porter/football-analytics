import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Gameweek dictionary
gw_dict = {
    "GW12": "2024-11-18",  # Confirmed as current date
    "GW13": "2024-11-26",
    "GW14": "2024-12-03",
    "GW15": "2024-12-06",
    "GW16": "2024-12-10",
    "GW17": "2024-12-17",  # Boxing Day
    "GW18": "2024-12-23",  # Midweek
    "GW19": "2024-12-28",
    "GW20": "2025-01-03",
    "GW21": "2025-01-13",  # After FA Cup break
    "GW22": "2025-01-21",
    "GW23": "2025-01-27",
    "GW24": "2025-02-03",  # After international break
    "GW25": "2025-02-19",
}
def get_current_gameweek(gw_dict):
    """
    Determines the current gameweek based on the current date.
    Returns the most recent gameweek that has occurred.
    """
    current_date = datetime.now()
    
    # Convert all dates in dictionary to datetime objects for comparison
    date_objects = {gw: datetime.strptime(date, '%Y-%m-%d') 
                   for gw, date in gw_dict.items()}
    
    # Sort gameweeks by date
    sorted_gws = sorted(date_objects.items(), key=lambda x: x[1])
    
    # Find the most recent gameweek
    current_gw = sorted_gws[0][0]  # Default to first GW
    
    for gw, date in sorted_gws:
        if current_date >= date:
            current_gw = gw
        else:
            break
            
    return current_gw

def twoway_expected_goals(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.rcParams['font.family'] = 'arial'

    # Get current gameweek
    current_gw = get_current_gameweek(gw_dict)

    # Set background color and dark theme
    fig.patch.set_facecolor('#161314')
    ax.set_facecolor('#161314')
    
    rows = len(df)
    y_pos = np.arange(rows)
    ax.set_ylim(-1, rows)

    # Extract home and away goal expectations
    home_goals = [d['home_goal_expectation'] for d in df]
    away_goals = [d['away_goal_expectation'] for d in df]
    home_teams = [d['home_team'] for d in df]
    away_teams = [d['away_team'] for d in df]

    # Calculate xG differences
    xg_diffs = [home - away for home, away in zip(home_goals, away_goals)]

    # Determine colors based on favorite/underdog
    favorite_color = '#FF4500'
    underdog_color = '#1E90FF'
    home_colors = [favorite_color if home > away else underdog_color for home, away in zip(home_goals, away_goals)]
    away_colors = [favorite_color if away > home else underdog_color for home, away in zip(home_goals, away_goals)]

    # Calculate the maximum xG to set the x-axis limits
    max_xg = max(max(home_goals), max(away_goals))
    ax.set_xlim(-max_xg * 1.2, max_xg * 1.2)  # Increased margin for labels

    # Create horizontal bar chart
    bar_height = 0.35
    ax.barh(y_pos, [-g for g in home_goals], align='center', color=home_colors, height=bar_height, label='Favorite' if home_colors[0] == favorite_color else 'Underdog')
    ax.barh(y_pos, away_goals, align='center', color=away_colors, height=bar_height, label='Favorite' if away_colors[0] == favorite_color else 'Underdog')

    # Customize the plot
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # Labels read top-to-bottom

    ax.set_xlabel('Expected Goals', color='white')
    ax.set_title(f'{current_gw} Predictions: Handicaps', color='white', fontsize=16, fontweight='bold', pad=20)

    plt.subplots_adjust(top=0.93, bottom=0.1)

    # Add "Home" and "Away" labels
    ax.text(0.02, 1, 'Home', transform=ax.transAxes, color='white', fontsize=14, fontweight='bold', ha='left', va='top')
    ax.text(0.98, 1, 'Away', transform=ax.transAxes, color='white', fontsize=14, fontweight='bold', ha='right', va='top')

    # Add team names, value labels, and xG difference on the bars
    for i, (home_v, away_v, home_team, away_team, xg_diff) in enumerate(zip(home_goals, away_goals, home_teams, away_teams, xg_diffs)):
        # Home team - adjusted position
        ax.text(-max_xg * 1.15, i, f'{home_team}', va='center', ha='right', color='white', fontweight='bold')
        ax.text(-home_v - 0.1, i, f'{home_v:.2f}', va='center', ha='right', color='white')
        
        # Away team - adjusted position
        ax.text(max_xg * 1.15, i, f'{away_team}', va='center', ha='left', color='white', fontweight='bold')
        ax.text(away_v + 0.1, i, f'{away_v:.2f}', va='center', ha='left', color='white')
        
        # xG difference line and label
        diff_color = 'yellow' if abs(xg_diff) > 0.1 else 'gray'

        # Calculate the position for the difference line and label
        if home_v > away_v:
            line_x = -away_v
            label_x = line_x + 0.1  # Moved slightly right
        else:
            line_x = home_v
            label_x = line_x - 0.1  # Moved slightly left

        # Draw the difference line
        ax.plot([line_x, line_x], [i-0.15, i+0.15], color='yellow', linewidth=2)
        
        # Add the difference label with adjusted position
        align = 'left' if home_v > away_v else 'right'
        ax.text(label_x, i, f'{(-1*abs(xg_diff)):+.2f}', va='center', ha=align, color=diff_color, fontweight='bold')

    # Customize grid, spines, and ticks
    ax.grid(color='gray', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', colors='white')

    # Add a vertical line at x=0
    ax.axvline(x=0, color='white', linewidth=0.8)

    # Customize legend
    ax.legend(loc='lower right', facecolor='#161314', edgecolor='white', labelcolor='white')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()