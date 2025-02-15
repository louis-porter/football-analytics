from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import random
import os
from datetime import datetime

class FBrefScraper:
    def __init__(self, season, headless=True):
        """Initialize the scraper with configuration"""
        self.season = season
        self.base_url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        self.matches_with_reds = []
        self.setup_driver(headless)
        
    def setup_driver(self, headless):
        """Configure and initialize the Chrome WebDriver"""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.maximize_window()
        
    def random_delay(self, min_seconds=0.5, max_seconds=1.5):
        """Add a short random delay between actions"""
        time.sleep(random.uniform(min_seconds, max_seconds))
        
    def check_rate_limit(self):
        """Check if the page is showing rate limiting"""
        rate_limit_indicators = [
            "Rate Limited Request",
            "Too many requests",
            "Please try again later"
        ]
        return any(indicator in self.driver.page_source for indicator in rate_limit_indicators)
        
    def wait_out_rate_limit(self, url, max_retries=3):
        """Handle rate limiting with exponential backoff"""
        for attempt in range(max_retries):
            if self.check_rate_limit():
                wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s backoff
                print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                self.driver.get(url)
                self.random_delay()
            else:
                return True
        return False
        
    def handle_cookie_popup(self):
        """Handle cookie consent popup if present"""
        try:
            cookie_selectors = [
                "button#CybotCookiebotDialogBodyButtonAccept",
                "button#onetrust-accept-btn-handler",
                "button.accept-cookies",
                "button[aria-label='Accept Cookies']"
            ]
            
            for selector in cookie_selectors:
                try:
                    cookie_button = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    self.driver.execute_script("arguments[0].click();", cookie_button)
                    print("Cookie popup handled")
                    self.random_delay()
                    return True
                except TimeoutException:
                    continue
            
            print("No cookie popup found or already accepted")
            return False
            
        except Exception as e:
            print(f"Error handling cookie popup: {e}")
            return False
            
    def find_fixtures_table(self):
        """Locate the fixtures table on the page"""
        try:
            # Try multiple ways to find the table
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
            
    def check_for_red_cards(self):
        """Check if the match report contains any red or yellow-red cards"""
        try:
            # Wait for page load
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            red_card_indicators = [
                "//div[@class='cards']//span[@class='red_card']",
                "//div[@class='cards']//span[@class='yellow_red_card']",
                "//td[@class='cards']//span[@class='red_card']",
                "//td[@class='cards']//span[@class='yellow_red_card']",
                "//div[@class='event']//span[contains(text(), 'Red Card')]",
                "//div[@class='event']//span[contains(text(), 'Second Yellow')]"
            ]
            
            match_details = {
                'date': None,
                'home_team': None,
                'away_team': None,
                'score': None,
                'red_cards': 0
            }
            
            # Try to get match details
            try:
                match_details['date'] = self.driver.find_element(By.XPATH, "//h1").text.split(" ")[0]
                teams = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'team')]//a")
                if len(teams) >= 2:
                    match_details['home_team'] = teams[0].text
                    match_details['away_team'] = teams[1].text
                score = self.driver.find_element(By.XPATH, "//div[contains(@class, 'score')]").text
                match_details['score'] = score
            except:
                print("Could not get complete match details")
            
            total_reds = 0
            for indicator in red_card_indicators:
                try:
                    red_cards = self.driver.find_elements(By.XPATH, indicator)
                    total_reds += len(red_cards)
                except NoSuchElementException:
                    continue
            
            if total_reds > 0:
                match_details['red_cards'] = total_reds
                print(f"Found {total_reds} red/yellow-red card(s)")
                return match_details
            
            print("No red cards found in this match")
            return None
            
        except TimeoutException:
            print("Timeout while checking for red cards")
            return None
        except Exception as e:
            print(f"Error checking for red cards: {e}")
            return None
            
    def scrape_matches(self):
        """Main function to scrape all matches for red cards"""
        try:
            print(f"\nStarting scrape for season {self.season}")
            self.driver.get(self.base_url)
            self.random_delay()
            
            if not self.wait_out_rate_limit(self.base_url):
                raise Exception("Could not overcome rate limiting after maximum retries")
            
            self.handle_cookie_popup()
            
            table = self.find_fixtures_table()
            if not table:
                raise Exception("Could not find fixtures table")
            
            rows = table.find_elements(By.TAG_NAME, "tr")
            total_matches = len(rows) - 1  # Subtract header row
            
            print(f"\nFound {total_matches} matches to check")
            
            for index, row in enumerate(rows[1:], 1):  # Skip header row
                try:
                    match_report = row.find_element(By.XPATH, ".//td/a[text()='Match Report']")
                    
                    if match_report:
                        match_url = match_report.get_attribute('href')
                        print(f"\nChecking match {index}/{total_matches}: {match_url}")
                        
                        self.driver.execute_script("arguments[0].click();", match_report)
                        self.random_delay(0.5, 1)  # Shorter delay after click
                        
                        if self.check_rate_limit():
                            if not self.wait_out_rate_limit(match_url):
                                print(f"Skipping match {match_url} due to persistent rate limiting")
                                self.driver.get(self.base_url)
                                self.random_delay()
                                continue
                        
                        match_details = self.check_for_red_cards()
                        if match_details:
                            match_details['match_url'] = match_url
                            self.matches_with_reds.append(match_details)
                        
                        self.driver.back()
                        self.random_delay(0.5, 1)  # Shorter delay after navigation
                        
                        if self.check_rate_limit():
                            if not self.wait_out_rate_limit(self.base_url):
                                raise Exception("Persistent rate limiting after navigation")
                        
                        table = self.find_fixtures_table()
                        if not table:
                            raise Exception("Lost fixtures table after navigation")
                
                except NoSuchElementException:
                    print("No Match Report link found in this row")
                    continue
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
                
                # Shorter delay between matches
                time.sleep(random.uniform(1, 2))
            
            return True
            
        except Exception as e:
            print(f"An error occurred during scraping: {e}")
            return False
            
    def save_results(self):
        """Save the scraped results to CSV"""
        if self.matches_with_reds:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'red_card_matches_{self.season}_{timestamp}.csv'
            
            # Create 'data' directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            filepath = os.path.join('data', filename)
            
            df = pd.DataFrame(self.matches_with_reds)
            df.to_csv(filepath, index=False)
            print(f"\nResults saved to {filepath}")
            print("\nMatches with red cards:")
            print(df)
            return True
        else:
            print("\nNo matches with red cards found")
            return False
            
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            
    def run(self):
        """Run the complete scraping process"""
        try:
            if self.scrape_matches():
                self.save_results()
        finally:
            self.cleanup()
            print("\nScript completed")

if __name__ == "__main__":
    # Example usage
    season = "2021-2022"
    scraper = FBrefScraper(season, headless=True)
    scraper.run()