from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scrape_match_fixtures(url):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode (no GUI)
    
    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    print("Loading page...")
    
    try:
        driver.get(url)
        # Wait for the matches to load
        time.sleep(5)  # Give JavaScript time to load content
        
        # Wait for and find all match fixtures
        matches = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "match-fixture__teams"))
        )
        
        print(f"Found {len(matches)} matches")
        
        fixtures_data = []
        for match in matches:
            try:
                # Get team names and time
                home_team = match.find_element(By.CLASS_NAME, "match-fixture__short-name").text
                away_team = match.find_elements(By.CLASS_NAME, "match-fixture__short-name")[1].text
                time_element = match.find_element(By.TAG_NAME, "time")
                match_time = time_element.text
                
                print(f"{home_team} vs {away_team} at {match_time}")
                
                fixtures_data.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'time': match_time
                })
                
            except Exception as e:
                print(f"Error processing match: {e}")
                continue
                
        return fixtures_data
        
    finally:
        driver.quit()

# Run the scraper
url = 'https://www.premierleague.com/fixtures'
matches = scrape_match_fixtures(url)