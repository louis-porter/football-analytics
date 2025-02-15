from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import random

def random_delay():
    """Add a random delay between 3-7 seconds"""
    time.sleep(random.uniform(3, 7))

def check_rate_limit(driver):
    """Check if we're being rate limited"""
    return "Rate Limited Request" in driver.page_source

def wait_out_rate_limit(driver, url, max_retries=3):
    """Handle rate limiting with exponential backoff"""
    for attempt in range(max_retries):
        if check_rate_limit(driver):
            wait_time = (2 ** attempt) * 30  # 30s, 60s, 120s backoff
            print(f"Rate limited. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            driver.get(url)
            random_delay()
        else:
            return True
    return False

# Set up Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# Initialize the driver
driver = webdriver.Chrome(options=options)
driver.maximize_window()

def find_table_direct():
    try:
        # Try to find table directly using JavaScript
        table = driver.execute_script('return document.querySelector("table#sched_2021-2022_9_1")')
        if table:
            return table
            
        # If that fails, wait a bit and try again
        random_delay()
        table = driver.execute_script('return document.querySelector("table#sched_2021-2022_9_1")')
        
        # One more fallback - try to find by class if ID fails
        if not table:
            table = driver.execute_script('return document.querySelector("table.stats_table.sortable")')
            
        return table
    except Exception as e:
        print(f"Error finding table: {e}")
        return None

# Rest of your functions remain the same...
[handle_cookie_popup and check_for_red_cards functions here]

try:
    # Store results
    matches_with_reds = []
    
    # Navigate to the FBref Premier League fixtures page
    url = "https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures"
    year = "2021-2022"
    
    driver.get(url)
    random_delay()
    
    # Check for rate limiting and handle it
    if not wait_out_rate_limit(driver, url):
        raise Exception("Could not overcome rate limiting after maximum retries")
    
    # Handle cookie popup
    handle_cookie_popup()
    random_delay()
    
    # Try to find the table
    table = find_table_direct()

    if not table:
        print("Could not find table. Current page source:")
        print(driver.page_source[:500])
        raise Exception("Table not found")
    
    # Find all rows in the table
    rows = table.find_elements(By.TAG_NAME, "tr")
    
    # Process each row
    for row in rows[1:]:  # Skip header row
        try:
            # Find the Match Report link
            match_report = row.find_element(By.XPATH, ".//td/a[text()='Match Report']")
            
            if match_report:
                match_url = match_report.get_attribute('href')
                print(f"\nChecking match: {match_url}")
                
                # Use JavaScript to click
                driver.execute_script("arguments[0].click();", match_report)
                random_delay()  # Use random delay instead of fixed
                
                # Check for rate limiting on match page
                if check_rate_limit(driver):
                    if not wait_out_rate_limit(driver, match_url):
                        print(f"Skipping match {match_url} due to persistent rate limiting")
                        driver.get(url)  # Go back to main page
                        random_delay()
                        continue
                
                # Check for red cards
                if check_for_red_cards():
                    print(f"Found red/yellow-red card in match: {match_url}")
                    matches_with_reds.append({
                        'match_url': match_url
                    })
                
                # Go back to the main page
                driver.back()
                random_delay()
                
                # Check for rate limiting after navigation
                if check_rate_limit(driver):
                    if not wait_out_rate_limit(driver, url):
                        raise Exception("Persistent rate limiting after navigation")
                
                # Re-find the table after going back
                table = find_table_direct()
                if not table:
                    raise Exception("Lost table after navigation")
        
        except NoSuchElementException:
            print("No Match Report link found in this row")
            continue
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
            
        # Add a longer delay between matches to avoid rate limiting
        time.sleep(random.uniform(5, 10))

    # Create DataFrame and save results
    if matches_with_reds:
        df = pd.DataFrame(matches_with_reds)
        df.to_csv(r'C:\Users\Owner\dev\football-analytics\data-scraping\fbref\scrape-red-card-games\matches_with_red_cards_2021.csv', index=False)
        print("\nResults saved to matches_with_red_cards.csv")
        print("\nMatches with red cards:")
        print(df)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()
    print("\nScript completed")