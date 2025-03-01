import pandas as pd
import time
import random
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
from bs4 import BeautifulSoup
import html
import re

class MatchDataScraper:
    def __init__(self, season, headless=True):
        self.season = season
        self.base_url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        self.match_data = []
        self.setup_driver(headless)
        
    def setup_driver(self, headless):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.maximize_window()

    def random_delay(self, min_seconds=3, max_seconds=5):
        time.sleep(random.uniform(min_seconds, max_seconds))

    def get_match_data(self, url):
        self.random_delay()
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            # Extract match date
            venue_time = soup.find('span', class_='venuetime')
            match_date = venue_time['data-venue-date'] if venue_time else None
            
            # Extract teams
            team_stats = soup.find('div', id='team_stats_extra')
            if team_stats:
                teams = team_stats.find_all('div', class_='th')
                teams = [t.text.strip() for t in teams if t.text.strip() != '']
                teams = list(dict.fromkeys(teams))  # Remove duplicates while preserving order
                home_team = teams[0] if len(teams) > 0 else None
                away_team = teams[1] if len(teams) > 1 else None
            else:
                home_team, away_team = None, None
                
            # Extract division
            division_link = soup.find('a', href=lambda x: x and '/comps/' in x and '-Stats' in x)
            division = division_link.text.strip() if division_link else None
            
            # Get shots data
            shots_table = soup.find('table', id='shots_all')
            if not shots_table:
                print(f"No shots table found for {url}")
                return None
            
            # First, get all header rows
            header_rows = shots_table.find_all('tr')[:2]  # Get both header rows
            if len(header_rows) < 2:
                print(f"Invalid shots table structure for {url}")
                return None
                
            # Extract headers from the second row (contains actual column names)
            headers = []
            for th in header_rows[1].find_all(['th', 'td']):
                header_text = th.get_text(strip=True)
                # If header is empty, use a placeholder
                headers.append(header_text if header_text else f"Column_{len(headers)}")
            
            # Make headers unique
            unique_headers = []
            header_counts = {}
            for header in headers:
                if header in header_counts:
                    header_counts[header] += 1
                    unique_headers.append(f"{header}_{header_counts[header]}")
                else:
                    header_counts[header] = 1
                    unique_headers.append(header)
            
            # Extract shot data rows
            rows_data = []
            for tr in shots_table.find_all('tbody')[0].find_all('tr'):
                cols = tr.find_all(['th', 'td'])
                row_data = []
                for col in cols:
                    value = col.get_text(strip=True)
                    if value == '':
                        value = "0"
                    row_data.append(value)
                if len(row_data) == len(unique_headers):  # Only add rows that match header count
                    rows_data.append(row_data)
            
            # Create shots DataFrame with unique headers
            shots_df = pd.DataFrame(rows_data, columns=unique_headers)
            
            # Check for penalties in player names and set event type accordingly
            player_col = [col for col in shots_df.columns if 'Player' in col][0]
            shots_df["Event Type"] = shots_df.apply(
                lambda row: "Penalty" if "(pen)" in str(row[player_col]).lower() else "Shot", 
                axis=1
            )
            
            # Get red cards data
            events = soup.find_all('div', class_=re.compile(r'^event\s'))
            
            times, scores, players, event_types, teams = [], [], [], [], []
            
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
            
            # Create events DataFrame
            events_df = pd.DataFrame({
                'Time': times,
                'Score': scores,
                'Player': players,
                'Event Type': event_types,
                'Team': teams
            })
            
            # Filter for red cards
            red_cards_df = events_df[events_df['Event Type'].isin(['Red Card', 'Second Yellow Card'])]
            red_cards_df = red_cards_df.reset_index(drop=True)
            
            # Process minute information
            red_cards_df["Minute"] = red_cards_df["Time"].str.extract(r'(\d+)').fillna('0')
            red_cards_df["Outcome"] = "Red Card"
            
            # Process shots DataFrame
            minute_col = [col for col in shots_df.columns if 'Minute' in col][0]
            squad_col = [col for col in shots_df.columns if 'Squad' in col][0]
            outcome_col = [col for col in shots_df.columns if 'Outcome' in col][0]
            xg_col = [col for col in shots_df.columns if 'xG' in col][0]
            psxg_col = [col for col in shots_df.columns if 'PSxG' in col][0]
            
            shots_df["Minute"] = shots_df[minute_col].str.extract(r'(\d+)').fillna('0')
            shots_df["Team"] = shots_df[squad_col]
            shots_df["Outcome"] = shots_df[outcome_col]
            shots_df["xG"] = shots_df[xg_col]
            shots_df["PSxG"] = shots_df[psxg_col]
            
            shots_df = shots_df[["Minute", "Team", "Player", "Event Type", "Outcome", "xG", "PSxG"]]
            
            # Clean up data
            shots_df.drop(shots_df[shots_df['Minute'] == ''].index, inplace=True)
            
            # Prepare red cards DataFrame for merging
            red_cards_df = red_cards_df[["Minute", "Team", "Player", "Event Type", "Outcome"]]
            red_cards_df["xG"] = 0
            red_cards_df["PSxG"] = 0
            
            # Combine DataFrames
            df = pd.concat([shots_df, red_cards_df], ignore_index=True)
            
            # Clean and convert data types
            df["Minute"] = pd.to_numeric(df["Minute"], errors='coerce')
            df["xG"] = pd.to_numeric(df["xG"], errors='coerce')
            df["PSxG"] = pd.to_numeric(df["PSxG"], errors='coerce')
            df.fillna(0.00, inplace=True)
            df.sort_values(by=["Minute"], inplace=True)
            
            # Add match metadata
            df["match_url"] = url
            df["match_date"] = match_date
            df["home_team"] = home_team
            df["away_team"] = away_team
            df["division"] = division
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error processing match data for {url}: {str(e)}")
            return None

    def find_fixtures_table(self):
        try:
            table_selectors = [
                f"table#sched_{self.season}_9_1",
                "table.stats_table.sortable",
                "//table[contains(@class, 'stats_table')]"
            ]
            
            for selector in table_selectors:
                try:
                    if selector.startswith("//"):
                        table = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                    else:
                        table = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                    return table
                except TimeoutException:
                    continue
                    
            raise Exception("Could not find fixtures table")
            
        except Exception as e:
            print(f"Error finding fixtures table: {e}")
            return None

    def scrape_matches(self):
        try:
            print(f"\nStarting scrape for season {self.season}")
            self.driver.get(self.base_url)
            self.random_delay()
            
            table = self.find_fixtures_table()
            if not table:
                raise Exception("Could not find fixtures table")
            
            rows = table.find_elements(By.TAG_NAME, "tr")
            total_matches = len(rows) - 1  # Subtract header row
            
            print(f"\nFound {total_matches} matches to process")
            
            test_rows = rows[1:6]
            for index, row in enumerate(rows[1:], 1): #rows[1:]
                try:
                    match_report = row.find_element(By.XPATH, ".//td/a[text()='Match Report']")
                    
                    if match_report:
                        match_url = match_report.get_attribute('href')
                        print(f"\nProcessing match {index}/{total_matches}: {match_url}")
                        
                        match_df = self.get_match_data(match_url)
                        if match_df is not None:
                            self.match_data.append(match_df)
                            print(f"Successfully processed match data")
                        else:
                            print(f"No data retrieved for match")
                    
                except NoSuchElementException:
                    print("No Match Report link found in this row")
                    continue
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue
                
                # Add delay between matches
                self.random_delay(2, 4)
            
            return True
            
        except Exception as e:
            print(f"An error occurred during scraping: {e}")
            return False

    def save_results(self):
        if self.match_data:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'match_data_{self.season}_{timestamp}.csv'
            
            os.makedirs('data', exist_ok=True)
            filepath = os.path.join('data', filename)
            
            combined_df = pd.concat(self.match_data, ignore_index=True)
            
            combined_df.to_csv(filepath, index=False)
            print(f"\nResults saved to {filepath}")
            print(f"\nTotal events collected: {len(combined_df)}")
            return True
        else:
            print("\nNo match data collected")
            return False

    def cleanup(self):
        if hasattr(self, 'driver'):
            self.driver.quit()
            
    def run(self):
        try:
            if self.scrape_matches():
                self.save_results()
        finally:
            self.cleanup()
            print("\nScript completed")

if __name__ == "__main__":
    season = "2022-2023"
    scraper = MatchDataScraper(season, headless=True)
    scraper.run()