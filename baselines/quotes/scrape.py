import requests
from bs4 import BeautifulSoup
import json
import time

def get_author_id(author_name):
    author_ids = {
        "nelson mandela": "367338",
        "mahatma gandhi": "5810891"
    }
    return author_ids.get(author_name.lower())

def scrape_author_quotes(author_name, max_pages=5):
    author_id = get_author_id(author_name)
    if not author_id:
        print(f"Author '{author_name}' not supported. Please choose 'Nelson Mandela' or 'Mahatma Gandhi'.")
        return []

    base_url = f"https://www.goodreads.com/author/quotes/{author_id}"
    quotes = []
    
    for page in range(1, max_pages+1):
        url = f"{base_url}?page={page}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            quote_divs = soup.find_all('div', class_='quoteText')
            
            if not quote_divs:
                break

            for div in quote_divs:
                text = div.get_text(strip=True).split('―')[0].strip()
                text = text.replace('"', '').replace('"', '')
                
                source = div.find_next('a', class_='authorOrTitle')
                source = source.get_text(strip=True) if source else ""
                
                likes = div.find_next('a', class_='smallText')
                likes = likes.get_text(strip=True) if likes else "0 likes"
                
                quotes.append({
                    'quote': text,
                    'source': source,
                    'likes': likes
                })
            
            print(f"Scraped page {page}")
            time.sleep(1)  # Respectful delay between requests

        except Exception as e:
            print(f"Error scraping page {page}: {str(e)}")
    
    # Save to JSON
    filename = f"{author_name.lower().replace(' ', '_')}_quotes.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(quotes, f, indent=2, ensure_ascii=False)
    
    print(f"Scraped {len(quotes)} quotes from {author_name}")
    return quotes

# Usage example:
author = input("Enter author name (Nelson Mandela or Mahatma Gandhi): ")
quotes = scrape_author_quotes(author, max_pages=39)
