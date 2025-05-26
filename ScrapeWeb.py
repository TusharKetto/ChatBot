import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Global driver and options
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--window-size=1920,1080')

def create_driver():
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver = create_driver()

# Seeds
seed_urls = [
    "https://www.ketto.org/how_it_works/fundraising-ideas-for-ngos.php",
    "https://customerhappiness.ketto.org/portal/en/kb/donor",
    "https://customerhappiness.ketto.org/portal/en/kb/fundraiser-queries",
    "https://customerhappiness.ketto.org/portal/en/kb/ngo-queries",
    "https://customerhappiness.ketto.org/portal/en/kb/trust-safety-queries",
    "https://customerhappiness.ketto.org/portal/en/kb/general-queries",
    "https://customerhappiness.ketto.org/portal/en/kb/funds-withdrawal",
    "https://customerhappiness.ketto.org/portal/en/kb/social-impact-plan",
    "https://customerhappiness.ketto.org/portal/en/kb/healthfirst",
    "https://customerhappiness.ketto.org/portal/en/kb/k-bank",
    "https://customerhappiness.ketto.org/portal/en/kb/ketto-one"
]

visited_urls = set()
structured_data = []

MAX_DEPTH = 3
RETRY_LIMIT = 3
CSV_PATH = "ketto_faq_data.csv"

def is_ketto_internal(url):
    parsed = urlparse(url)
    return parsed.netloc.endswith("ketto.org")

def remove_boilerplate(soup):
    for tag in soup(["header", "footer", "nav", "script", "style", "noscript", "iframe"]):
        tag.decompose()
    return soup

def is_kb_article_page(soup):
    return bool(soup.select_one("div.kb-article__title") or soup.select_one(".article-title"))

def extract_kb_article(url, soup):
    title_tag = soup.select_one("div.kb-article__title") or soup.select_one(".article-title")
    question = title_tag.get_text(strip=True) if title_tag else "Untitled Article"
    content_blocks = soup.select("div.kb-article__content, .article-content")
    paragraphs = []
    for block in content_blocks:
        for p in block.find_all(['p', 'ul', 'ol', 'li']):
            text = p.get_text(separator="\n", strip=True)
            if text:
                paragraphs.append(text)
    answer = "\n".join(paragraphs).strip()
    if answer:
        return {
            "page_url": url,
            "page_title": soup.title.string.strip() if soup.title else "",
            "question": question,
            "answer": answer
        }
    return None

def extract_structured_content(url, soup):
    page_title = soup.title.string.strip() if soup.title else "No Title"
    records = []
    for section in soup.find_all(['h2', 'h3']):
        section_title = section.get_text(strip=True)
        content = []
        for sibling in section.find_next_siblings():
            if sibling.name in ['h2', 'h3']:
                break
            if sibling.name in ['p', 'ul', 'ol', 'div', 'span']:
                text = sibling.get_text(separator="\n", strip=True)
                if text:
                    content.append(text)
        full_content = "\n".join(content).strip()
        if full_content:
            records.append({
                "page_url": url,
                "page_title": page_title,
                "question": section_title,
                "answer": full_content
            })
    return records

def extract_internal_links(soup, base_url):
    links = set()
    for a in soup.find_all('a', href=True):
        full_url = urljoin(base_url, a['href']).split('#')[0]
        if is_ketto_internal(full_url):
            links.add(full_url)
    return links

def save_progress():
    if structured_data:
        df = pd.DataFrame(structured_data)
        df.to_csv(CSV_PATH, index=False)
        print(f"💾 Saved {len(structured_data)} records to {CSV_PATH}")

def restart_driver():
    global driver
    try:
        driver.quit()
    except Exception:
        pass
    print("♻️ Restarting Chrome driver...")
    driver = create_driver()

def safe_get(url):
    """Try loading the page with retries and driver restart on failure."""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            driver.get(url)
            time.sleep(2)  # can add smarter wait here if needed
            return BeautifulSoup(driver.page_source, 'html.parser')
        except (WebDriverException, TimeoutException) as e:
            print(f"⚠️ Attempt {attempt} failed for {url}: {e}")
            restart_driver()
            if attempt == RETRY_LIMIT:
                print(f"❌ Giving up on {url} after {RETRY_LIMIT} attempts.")
                return None
    return None

def scrape_page(url, depth=0):
    if url in visited_urls or depth > MAX_DEPTH:
        return
    print(f"{'  '*depth}🔍 Scraping depth={depth}: {url}")
    visited_urls.add(url)

    soup = safe_get(url)
    if not soup:
        return

    soup = remove_boilerplate(soup)

    if is_kb_article_page(soup):
        record = extract_kb_article(url, soup)
        if record:
            structured_data.append(record)
            save_progress()
    else:
        records = extract_structured_content(url, soup)
        if records:
            structured_data.extend(records)
            save_progress()
        # Crawl internal links recursively
        internal_links = extract_internal_links(soup, url)
        for link in internal_links:
            scrape_page(link, depth + 1)

if __name__ == "__main__":
    try:
        for url in seed_urls:
            scrape_page(url)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user, saving progress...")
    finally:
        save_progress()
        driver.quit()
        print("✅ Scraping finished and driver closed.")
