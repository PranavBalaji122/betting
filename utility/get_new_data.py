from curl_cffi import requests  # <-- The curl_cffi "requests"-like API
from bs4 import BeautifulSoup
import csv

def login_to_stathead(session, email, password):
    """
    Logs in to Stathead using the given credentials via curl_cffi session.
    """
    login_url = "https://stathead.com/users/login.cgi"
    
    # For best results, impersonate a modern browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/110.0.0.0 Safari/537.36"
        )
    }

    # Stathead expects 'email' and 'password' fields
    payload = {
        "email": email,
        "password": password
    }

    resp = session.post(login_url, data=payload, headers=headers)
    resp.raise_for_status()

    if "Sign out" not in resp.text and "My Account" not in resp.text:
        print("Warning: Login may not have been successful.")
    else:
        print("Logged in to Stathead successfully.")

def append_table_rows_to_csv(session, url, csv_filename='data.csv'):
    """
    Uses the provided curl_cffi session (with cookies) to fetch the page at `url`,
    finds the table by its ID/class, and appends rows (skipping the header) to CSV.
    """
    # Impersonate a modern Chrome. Some sites block default user-agents.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/110.0.0.0 Safari/537.36"
        )
    }

    response = session.get(url, headers=headers, allow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find(
        "table",
        {
            "id": "stats",
            "class": "stats_table sortable suppress_embed suppress_youtube "
                     "suppress_about_sharing suppress_link now_sortable"
        }
    )
    if table is None:
        print("Could not find the table with the specified ID/class.")
        return

    rows = table.find_all("tr")

    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Skip header row (assumed to be rows[0])
        for row in rows[1:]:
            cells = row.find_all(['th', 'td'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            writer.writerow(row_data)

    print(f"Table rows appended to {csv_filename} successfully!")

def main():
    # Replace with your actual credentials
    my_email = "manraaj06@outlook.com"
    my_password = ""
    
    # Create a session that pretends to be Chrome 110 on Windows
    # (curl_cffi has a built-in impersonate parameter, or you can set headers manually)
    session = requests.Session(impersonate="chrome110")

    # 1. Login to Stathead (this should set the session's auth cookies, if successful)
    login_to_stathead(session, my_email, my_password)

    # 2. URL that requires login
    url_to_scrape = (
        "https://stathead.com/basketball/player-game-finder.cgi"
        "?request=1&order_by=name_display_csk&timeframe=last_n_days&previous_days=2"
    )

    # 3. Scrape the table and save to CSV
    append_table_rows_to_csv(session, url_to_scrape, csv_filename='data.csv')

if __name__ == "__main__":
    main()
