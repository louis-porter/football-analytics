import requests
from bs4 import BeautifulSoup
import pandas as pd
import html
import re
import random
import time

def get_match_data(url):
    """
    Scrape match data from fbref.com and return a DataFrame with shots and red cards data.
    
    Parameters:
    url (str): URL of the fbref.com match page
    
    Returns:
    pandas.DataFrame: Combined data of shots and red cards with timing and team information
    """
    # Fetch and parse the page
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'lxml')
    
    # Get shots data - using ID instead of index
    shots_table = soup.find('table', id='shots_all')
    if not shots_table:
        raise ValueError("Could not find shots table")
    
    # Rest of your code remains the same
    headers = [th.get_text(strip=True) for th in shots_table.find_all('tr')[1].find_all('th')]
    
    # Extract shot data rows with blank value handling
    rows_data = []
    for tr in shots_table.find_all('tbody')[0].find_all('tr'):
        cols = tr.find_all(['th', 'td'])
        row_data = []
        for col in cols:
            value = col.get_text(strip=True)
            # Convert blank to "0" for numeric columns
            if value == '':
                value = "0"
            row_data.append(value)
        rows_data.append(row_data)
    
    # Create shots DataFrame
    shots_df = pd.DataFrame(rows_data, columns=headers)
    
    # Get red cards data
    events = soup.find_all('div', class_=re.compile(r'^event\s'))
    
    # Initialize lists for event data
    times, scores, players, event_types, teams = [], [], [], [], []
    
    # Extract event data
    for event in events:
        # Time and score
        time_score_div = event.find('div')
        if time_score_div:
            time_score_text = html.unescape(time_score_div.get_text(strip=True))
            time = time_score_text.split("'")[0] + "'" if "'" in time_score_text else time_score_text
            score = time_score_text.split("'")[1] if "'" in time_score_text else ''
            times.append(time)
            scores.append(score)
            
        # Player name
        player = event.find('a').get_text(strip=True) if event.find('a') else ''
        players.append(player)
        
        # Event type
        event_type = 'Unknown'
        for div in event.find_all('div'):
            if '—' in div.get_text():
                event_type = div.get_text(strip=True).split('—')[-1].strip()
                break
        event_types.append(event_type)
        
        # Team
        team_logo = event.find('img', class_='teamlogo')
        if team_logo:
            team_name = team_logo.get('alt').replace(' Club Crest', '')
            teams.append(team_name)
        else:
            teams.append('Unknown')
    
    # Create red cards DataFrame
    red_cards_df = pd.DataFrame({
        'Time': times,
        'Score': scores,
        'Player': players,
        'Event Type': event_types,
        'Team': teams
    })
    
    # Filter for red cards
    red_cards_df = red_cards_df[red_cards_df['Event Type'].isin(['Red Card', 'Second Yellow Card'])]
    red_cards_df = red_cards_df.reset_index(drop=True)
    
    # Process minute information
    red_cards_df["Minute"] = red_cards_df["Time"].str[:2]
    red_cards_df["Outcome"] = "Red Card"
    
    # Process shots DataFrame
    shots_df["Minute"] = shots_df["Minute"].str[:2]
    shots_df["Team"] = shots_df["Squad"]
    shots_df = shots_df[["Minute", "Team", "xG", "PSxG", "Outcome"]]
    shots_df["Event Type"] = "Shot"
    
    # Clean up data
    shots_df.drop(shots_df[shots_df['Minute'] == ''].index, inplace=True)
    
    # Prepare red cards DataFrame for merging
    red_cards_df = red_cards_df[["Minute", "Team", "Event Type", "Outcome"]]
    
    # Combine DataFrames
    df = pd.concat([shots_df, red_cards_df], ignore_index=True)
    
    # Clean and convert data types
    df["Minute"] = df["Minute"].astype(int)
    df["xG"] = df["xG"].astype(float)
    df["PSxG"] = df["PSxG"].astype(float)
    df.fillna(0.00, inplace=True)
    df.sort_values(by=["Minute"], inplace=True)
    
    # Add match URL column
    df["match_url"] = url
    
    return df

# Example usage:
red_card_matches_df = pd.read_csv(r"data-scraping\fbref\scrape-red-card-games\matches_with_red_cards_2021.csv")

all_matches_data = []

for url in red_card_matches_df["match_url"]:
    try:
        # Random delay between 5-10 seconds
        delay = random.uniform(5, 10)
        time.sleep(delay)
        
        match_df = get_match_data(url)
        all_matches_data.append(match_df)
        print(f"Successfully processed: {url} (waited {delay:.1f}s)")
        
    except Exception as e:
        print(f"Error processing URL: {url}")
        print(f"Error message: {str(e)}")
        continue


# Combine all DataFrames
combined_df = pd.concat(all_matches_data, ignore_index=True)

combined_df.to_csv(r"data-scraping\fbref\scrape-red-card-games\red_card_data_2021.csv", index=False)