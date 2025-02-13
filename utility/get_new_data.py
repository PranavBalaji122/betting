from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import undetected_chromedriver as uc  #  Undetected ChromeDriver for stealth mode

import csv
import os
from dotenv import load_dotenv
load_dotenv()



def get_new_Data():

    username = os.getenv("STATHEAD_USERNAME")
    password = os.getenv("STATHEAD_PASSWORD")

    #  Setup Undetected ChromeDriver
    chrome_driver_path = "/opt/homebrew/bin/chromedriver"  # Ensure correct path
    options = webdriver.ChromeOptions()

    #  Make Selenium Less Detectable
    options.add_argument("--disable-blink-features=AutomationControlled")  # Stealth mode
    # options.add_argument("start-maximized")  # Open browser in full screen
    options.add_argument("disable-infobars")  # Remove Selenium alert bar
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    #  Set User-Agent (Pretend to be a Real User)
    # options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

    #  Use Undetected ChromeDriver
    driver = uc.Chrome(options=options, headless=False)  # Keep browser visible to reduce bot detection

    #  Open login page
    login_url = "https://www.stathead.com/users/login.cgi"
    driver.get(login_url)

    #  Simulate Real User Behavior
    time.sleep(5)  # Wait for Cloudflare to process

    #  Wait for username field to appear
    try:
        username_field = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        password_field = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        
        #  Simulate Human Typing
        for char in username:
            username_field.send_keys(char)
            time.sleep(0.05)  # Small delay between key presses

        for char in password:
            password_field.send_keys(char)
            time.sleep(0.05)  # Small delay between key presses

        #  Simulate a Real Mouse Click
        time.sleep(1)
        password_field.send_keys(Keys.RETURN)

        print(" Login submitted successfully!")
        basketball_stats_url = "https://stathead.com/basketball/player-game-finder.cgi?request=1&order_by=name_display_csk&timeframe=last_n_days&previous_days=1"
        driver.get(basketball_stats_url)

        #  Wait for the page to load completely
        time.sleep(5)

        #  Print page title to confirm navigation
        print(" Navigated to:", driver.title)

        table = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "stats_table"))
        )

        #  Extract table rows
        rows = table.find_elements(By.TAG_NAME, "tr")

        #  Extract Data
        data = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) > 0:
                data.append([
                    1,
                    cols[0].text.strip(),  # Rank
                    cols[1].text.strip(),  # Player
                    cols[2].text.strip(),  # Date
                    cols[3].text.strip(),  # Age
                    cols[4].text.strip(),  # Team
                    cols[5].text.strip(),  # Opp
                    cols[6].text.strip(),  # Result
                    cols[7].text.strip(),  # GS
                    cols[8].text.strip(),  # MP
                    cols[9].text.strip(),  # FG
                    cols[10].text.strip(),  # FGA
                    cols[11].text.strip(),  # FG%
                    cols[12].text.strip(),  # 2P
                    cols[13].text.strip(),  # 2PA
                    cols[14].text.strip(),  # 2P%
                    cols[15].text.strip(),  # 3P
                    cols[16].text.strip(),  # 3PA
                    cols[17].text.strip(),  # 3P%
                    cols[18].text.strip(),  # FT
                    cols[19].text.strip(),  # FTA
                    cols[20].text.strip(),  # FT%
                    cols[21].text.strip(),  # TS%
                    cols[22].text.strip(),  # ORB
                    cols[23].text.strip(),  # DRB
                    cols[24].text.strip(),  # TRB
                    cols[25].text.strip(),  # AST
                    cols[26].text.strip(),  # STL
                    cols[27].text.strip(),  # BLK
                    cols[28].text.strip(),  # TOV
                    cols[29].text.strip(),  # PF
                    cols[30].text.strip(),  # PTS
                    cols[31].text.strip(),  # GmSc
                    cols[32].text.strip(),  # BPM
                    cols[33].text.strip(),  # +/-
                    cols[34].text.strip(),  # Pos
                    "N/A"  # Player-additional
                ])

        #  Save Data as CSV
        filename = "csv/data.csv"
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

        print(f" Data successfully saved to {filename}")

    except Exception as e:
        print(" Element not found! Error:", e)

    #  Give time to login
    time.sleep(8)

    #  Close browser
    driver.quit()


    # install all dependencies needed
    # pip install selenium undetected-chromedriver
    # brew install --cask chromedriver
    # brew install --cask google-chrome
    # brew install --cask firefox
    # put this in termainal:
    # /Applications/Python\ 3.12/Install\ Certificates.command


if __name__ == "__main__":
    get_new_Data()