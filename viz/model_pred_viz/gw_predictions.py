import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from datetime import datetime
import os
import sys

import os

logos_dict = {
    "Arsenal": os.path.join("viz", "logos", "arsenal.png"),
    "Aston Villa": os.path.join("viz", "logos", "aston-villa.png"),
    "Bournemouth": os.path.join("viz", "logos", "bournemouth.png"),
    "Brentford": os.path.join("viz", "logos", "brentford.png"),
    "Brighton": os.path.join("viz", "logos", "brighton.png"),
    "Chelsea": os.path.join("viz", "logos", "chelsea.png"),
    "Crystal Palace": os.path.join("viz", "logos", "crystal-palace.png"),
    "Everton": os.path.join("viz", "logos", "everton.png"),
    "Nott'm Forest": os.path.join("viz", "logos", "forest.png"),
    "Fulham": os.path.join("viz", "logos", "fulham.png"),
    "Ipswich": os.path.join("viz", "logos", "ipswich.png"),
    "Leicester": os.path.join("viz", "logos", "leicester.png"),
    "Liverpool": os.path.join("viz", "logos", "liverpool.png"),
    "Man City": os.path.join("viz", "logos", "man-city.png"),
    "Man United": os.path.join("viz", "logos", "man-utd.png"),
    "Newcastle": os.path.join("viz", "logos", "newcastle.png"),
    "Southampton": os.path.join("viz", "logos", "southampton.png"),
    "Tottenham": os.path.join("viz", "logos", "tottenham.png"),
    "West Ham": os.path.join("viz", "logos", "west-ham.png"),
    "Wolves": os.path.join("viz", "logos", "wolves.png")
}

# Gameweek dictionary
gw_dict = {
    'GW24': '2025-01-28',
    'GW25': '2025-02-13',
    'GW26': '2025-02-17',
    'GW27': '2025-02-24',
    'GW28': '2025-02-28',
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

def primary_pred(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['font.family'] = 'arial'

    # Set background color and dark theme
    fig.patch.set_facecolor('#161314')
    ax.set_facecolor('#161314')

    rows = 10
    cols = 17.5

    ax.set_ylim(-1, rows + 1)
    ax.set_xlim(0, cols + .5)

    text_color = 'white'

    # Create arrays for home win, draw, and away win probabilities for background coloring
    home_win_probs = np.array([d['home_win_prob'] for d in df])
    draw_probs = np.array([d['draw_prob'] for d in df])
    away_win_probs = np.array([d['away_win_prob'] for d in df])

    # Normalize the probabilities so that they add up to 1
    total_probs = home_win_probs + draw_probs + away_win_probs
    home_win_probs = home_win_probs / total_probs
    draw_probs = draw_probs / total_probs
    away_win_probs = away_win_probs / total_probs

    # Function to adjust the rounded percentages so they sum to 100%
    def adjust_probabilities_to_100(home_win, draw, away_win):
        rounded_home = round(home_win * 100)
        rounded_draw = round(draw * 100)
        rounded_away = round(away_win * 100)
        
        total_rounded = rounded_home + rounded_draw + rounded_away
        diff = 100 - total_rounded
        
        # Adjust the largest value to ensure the total is exactly 100%
        if diff != 0:
            if max(rounded_home, rounded_draw, rounded_away) == rounded_home:
                rounded_home += diff
            elif max(rounded_home, rounded_draw, rounded_away) == rounded_draw:
                rounded_draw += diff
            else:
                rounded_away += diff
        
        return rounded_home, rounded_draw, rounded_away

    # Normalize the probabilities to range [0, 1] for gradient
    norm = plt.Normalize(0, 1)

    def get_text_color_based_on_bg(value, cmap, norm):
        rgba = cmap(norm(value))  # Get RGBA color from colormap
        # Convert RGBA to luminance (perceived brightness)
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        # Return white if dark background, black if light background
        return 'white' if luminance < 0.5 else 'black'

    # Create a colormap for gradient (from green to red)
    cmap = plt.cm.Reds

    for row in range(rows):
        d = df[-(row + 1)] 
        
        home_win_val = home_win_probs[-(row + 1)]
        draw_val = draw_probs[-(row + 1)]
        away_win_val = away_win_probs[-(row + 1)]

        # Adjust the rounded percentages to sum to 100%
        rounded_home_win, rounded_draw, rounded_away_win = adjust_probabilities_to_100(home_win_val, draw_val, away_win_val)
        
        ax.imshow([[home_win_probs[-(row + 1)]]], aspect='auto', cmap=cmap, norm=norm, extent=[6.25, 7.75, row-0.5, row+0.49])
        ax.imshow([[draw_probs[-(row + 1)]]], aspect='auto', cmap=cmap, norm=norm, extent=[8.25, 9.75, row-0.5, row+0.49])
        ax.imshow([[away_win_probs[-(row + 1)]]], aspect='auto', cmap=cmap, norm=norm, extent=[10.25, 11.75, row-0.5, row+0.49])

        home_text_color = get_text_color_based_on_bg(home_win_val, cmap, norm)
        draw_text_color = get_text_color_based_on_bg(draw_val, cmap, norm)
        away_text_color = get_text_color_based_on_bg(away_win_val, cmap, norm)

        # Add team logos for home and away teams
        if d['home_team'] in logos_dict:
            home_logo_path = logos_dict[d['home_team']]
            home_logo_img = mpimg.imread(home_logo_path)
            ax.imshow(home_logo_img, extent=[3, 3.75, row-0.35, row+0.35], aspect='auto')
        
        if d['away_team'] in logos_dict:
            away_logo_path = logos_dict[d['away_team']]
            away_logo_img = mpimg.imread(away_logo_path)
            ax.imshow(away_logo_img, extent=[14.25, 15, row-0.35, row+0.35], aspect='auto')

        ax.text(x=.25, y=row, s=f"{d['home_team']}", va="center", ha="left", color=text_color) 
        ax.text(x=5, y=row, s=f"{d['home_goal_expectation']:.2f}", va="center", ha="center", color=text_color)  
        ax.text(x=7, y=row, s=f"{rounded_home_win}%", va="center", ha="center", color=home_text_color)  
        ax.text(x=9, y=row, s=f"{rounded_draw}%", va="center", ha="center", color=draw_text_color)  
        ax.text(x=11, y=row, s=f"{rounded_away_win}%", va="center", ha="center", color=away_text_color)  
        ax.text(x=13, y=row, s=f"{d['away_goal_expectation']:.2f}", va="center", ha="center", color=text_color)  
        ax.text(x=17.75, y=row, s=f"{d['away_team']}", va="center", ha="right", color=text_color)

    # Column headers in white and bold
    ax.text(.25, 9.75, 'Home', weight='bold', ha='left', color=text_color)
    ax.text(5, 9.75, 'Goal', weight='bold', ha='center', color=text_color)
    ax.text(7, 9.75, 'Home Win', weight='bold', ha='center', color=text_color)
    ax.text(9, 9.75, 'Draw', weight='bold', ha='center', color=text_color)
    ax.text(11, 9.75, 'Away Win', weight='bold', ha='center', color=text_color)
    ax.text(13, 9.75, 'Goal', weight='bold', ha='center', color=text_color)
    ax.text(17.75, 9.75, 'Away ', weight='bold', ha='right', color=text_color)

    # Add row separator lines in a lighter gray color
    for row in range(rows):
        ax.plot(
            [0, cols + 1],
            [row - .5, row - .5],
            ls=':',
            lw='.5',
            c='lightgray'
        )

    ax.plot([0, cols + 1], [9.5, 9.5], lw='.5', c='white')

    ax.axis('off')

    # Get current gameweek
    current_gw = get_current_gameweek(gw_dict)
    
    ax.set_title(
        f'{current_gw}: Predictions',
        loc='left',
        fontsize=18,
        weight='bold',
        color=text_color
    )

    plt.show()
