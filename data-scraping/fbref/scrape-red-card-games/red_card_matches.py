from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time

# Set up Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Initialize the driver
driver = webdriver.Chrome(options=options)
driver.maximize_window()

def wait_for_element(by, value, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout waiting for element: {value}")
        return None

def handle_cookie_popup():
    try:
        selectors = [
            "osano-accept-all-button",
            "//button[contains(text(), 'Accept')]",
            "//button[contains(text(), 'Allow')]",
        ]
        
        for selector in selectors:
            try:
                if selector.startswith("//"):
                    button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                else:
                    button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.ID, selector))
                    )
                button.click()
                return True
            except:
                continue
        return False
    except Exception as e:
        print(f"Error handling cookie popup: {e}")
        return False

def check_for_red_cards():
    try:
        # Find all elements with class 'cards'
        cards_sections = driver.find_elements(By.CLASS_NAME, "cards")
        
        for cards_section in cards_sections:
            # Get all spans within this cards section
            spans = cards_section.find_elements(By.TAG_NAME, "span")
            
            for span in spans:
                classes = span.get_attribute("class")
                if classes and ("red_card" in classes or "yellow_red_card" in classes):
                    return True
        
        return False
    
    except Exception as e:
        print(f"Error checking for red cards: {e}")
        return False

try:
    # Store results
    matches_with_reds = []
    
    # Navigate to the FBref Premier League fixtures page
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    driver.get(url)
    time.sleep(2)
    
    # Handle cookie popup
    handle_cookie_popup()
    time.sleep(2)
    
    # Wait for the table to load
    table = wait_for_element(By.ID, "sched_2024-2025_9_1")
    
    if table:
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
                    time.sleep(2)
                    
                    # Check for red cards
                    if check_for_red_cards():
                        print(f"Found red/yellow-red card in match: {match_url}")
                        # Get match date and teams from the URL
                        matches_with_reds.append({
                            'match_url': match_url
                        })
                    
                    # Go back to the main page
                    driver.back()
                    time.sleep(2)
                    
                    # Re-find the table after going back
                    table = wait_for_element(By.ID, "sched_2024-2025_9_1")
            
            except NoSuchElementException:
                print("No Match Report link found in this row")
                continue
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

    # Create DataFrame and save results
    if matches_with_reds:
        df = pd.DataFrame(matches_with_reds)
        df.to_csv(r'C:\Users\Owner\dev\football-analytics\data-scraping\fbref\scrape-red-card-games\matches_with_red_cards.csv', index=False)
        print("\nResults saved to matches_with_red_cards.csv")
        print("\nMatches with red cards:")
        print(df)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()
    print("\nScript completed")