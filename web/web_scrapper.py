from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import undetected_chromedriver as uc  # ✅ Undetected ChromeDriver for stealth mode
import buetifulsoup4 as bs4






# ✅ Setup Undetected ChromeDriver
chrome_driver_path = "/opt/homebrew/bin/chromedriver"  # Ensure correct path
options = webdriver.ChromeOptions()

# ✅ Make Selenium Less Detectable
options.add_argument("--disable-blink-features=AutomationControlled")  # Stealth mode
options.add_argument("start-maximized")  # Open browser in full screen
options.add_argument("disable-infobars")  # Remove Selenium alert bar
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")

# ✅ Set User-Agent (Pretend to be a Real User)
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

# ✅ Use Undetected ChromeDriver
driver = uc.Chrome(options=options, headless=False)  # Keep browser visible to reduce bot detection

# ✅ Open login page
login_url = "https://www.stathead.com/users/login.cgi"
driver.get(login_url)

# ✅ Simulate Real User Behavior
time.sleep(5)  # Wait for Cloudflare to process

# ✅ Wait for username field to appear
try:
    username_field = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.NAME, "username"))
    )
    password_field = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.NAME, "password"))
    )
    
    # ✅ Simulate Human Typing
    for char in "singhsumair":
        username_field.send_keys(char)
        time.sleep(0.2)  # Small delay between key presses

    for char in "Derp1234(*":
        password_field.send_keys(char)
        time.sleep(0.2)  # Small delay between key presses

    # ✅ Simulate a Real Mouse Click
    time.sleep(1)
    password_field.send_keys(Keys.RETURN)

    print("✅ Login submitted successfully!")
    basketball_stats_url = "https://stathead.com/basketball/player-game-finder.cgi?request=1&timeframe=last_n_days&previous_days=2"
    driver.get(basketball_stats_url)

    # ✅ Wait for the page to load completely
    time.sleep(5)

    # ✅ Print page title to confirm navigation
    print("✅ Navigated to:", driver.title)


except Exception as e:
    print("❌ Element not found! Error:", e)

# ✅ Give time to login
time.sleep(8)

# ✅ Close browser
driver.quit()






# install all dependencies needed
# pip install selenium undetected-chromedriver
# brew install --cask chromedriver
# brew install --cask google-chrome
# brew install --cask firefox
# put this in termainal:
# /Applications/Python\ 3.12/Install\ Certificates.command