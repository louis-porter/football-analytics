from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
from viz.model_pred_viz.gw_dict import gw_dict

def clean_date(date_str):
    # List of all days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Remove day of week if present
    for day in days:
        date_str = date_str.replace(day, '').strip()
    
    # Remove ordinal suffixes
    for suffix in ['st', 'nd', 'rd', 'th']:
        date_str = date_str.replace(suffix, '')
    
    # Clean up any extra whitespace
    date_str = ' '.join(date_str.split())
    
    return date_str

def scrape_match_fixtures(url):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode (no GUI)
    
    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    print("Loading page...")
    
    try:
        driver.get(url)
        time.sleep(5)  # Give JavaScript time to load content
        
        # Find all dates and fixtures
        date_elements = driver.find_elements(By.CLASS_NAME, "fixture-date")
        fixture_elements = driver.find_elements(By.CLASS_NAME, "fixture")
        
        print(f"Found {len(date_elements)} dates and {len(fixture_elements)} fixtures")
        
        fixtures = []
        current_date = None
        gw_dates = {gw: pd.to_datetime(date) for gw, date in gw_dict.items()}
        next_gw = next(gw for gw, date in gw_dates.items() if pd.Timestamp.today() <= date)
        print(next_gw)
        next_gw_date = gw_dates[next_gw]
        print(next_gw_date)

        team_names = {
            'Manchester City': 'Man City',
            'AFC Bournemouth': 'Bournemouth',
            'West Ham United': 'West Ham',
            'Nottingham Forest': "Nott'm Forest",
            'Leicester City': 'Leicester',
            'Newcastle United': 'Newcastle',
            'Tottenham Hotspur': 'Tottenham',
            'Ipswich Town': 'Ipswich',
            'Brighton & Hove Albion': 'Brighton',
            'Manchester United': 'Man United'
        }
        
        # Iterate through all elements in order
        all_elements = driver.find_elements(By.CSS_SELECTOR, ".fixture-date, .fixture")
        
        for element in all_elements:
            if "fixture-date" in element.get_attribute("class"):
                current_date = element.text
                current_date = pd.to_datetime(clean_date(current_date), format='%d %B %Y')
            elif "fixture" in element.get_attribute("class"):
                if current_date < next_gw_date:  # Only add if we have a date
                    try:
                        teams_element = element.find_element(By.CLASS_NAME, "fixture__teams")
                        teams_text = teams_element.text
                        home_team, away_team = teams_text.split(' v ')

                        # Map team names using dictionary, if not in dictionary keep original name
                        home_team = team_names.get(home_team.strip(), home_team.strip())
                        away_team = team_names.get(away_team.strip(), away_team.strip())
                        
                        fixtures.append({
                            'home_team': home_team.strip(),
                            'away_team': away_team.strip()
                        })
                    except Exception as e:
                        print(f"Error processing fixture: {e}")
        

        return fixtures

    finally:
        driver.quit()

# Run the scraper
url = 'https://www.live-footballontv.com/live-premier-league-football-on-tv.html'
matches = scrape_match_fixtures(url)
print(matches)