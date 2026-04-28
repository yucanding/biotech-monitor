import feedparser
import re
import time
import calendar
import cloudscraper
import random
import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from openai import OpenAI

# 环境变量
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 配置
HOURS_WINDOW = 96
RSS_URLS = [
    "https://www.stocktitan.net/rss-clinical-trials",
    "https://www.stocktitan.net/rss-fda-approvals"
]
SENT_DB_FILE = "sent_urls.txt"
PATTERN_ACTION = r"to (?:report|announce|discuss)"
PATTERN_SUBJECT = r"data|phase"
PATTERN_EXCLUDE = r"financial|quarter|Q1"

scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload, timeout=10)

def get_article_body(url):
    try:
        time.sleep(random.uniform(2, 4))
        response = scraper.get(url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            body = soup.find('div', class_='article-body') or soup.find('article')
            return body.get_text(separator=' ', strip=True)[:2000]
        return None
    except:
        return None

def analyze_event_time(title, body):
    prompt = f"Extract only the clinical data release date and time from this text. If not mentioned, reply 'NONE'. Title: {title}. Body: {body}."
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=50
        )
        res = response.choices[0].message.content.strip()
        return None if "NONE" in res.upper() else res
    except:
        return None

def run_monitor():
    current_utc_ts = time.time()
    cutoff_ts = current_utc_ts - (HOURS_WINDOW * 3600)
    
    if not os.path.exists(SENT_DB_FILE): open(SENT_DB_FILE, "w").close()
    with open(SENT_DB_FILE, "r") as f: sent_urls = set(line.strip() for line in f)

    new_urls = []
    for rss_url in RSS_URLS:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if entry.link in sent_urls: continue
            pub_ts = calendar.timegm(entry.published_parsed) if hasattr(entry, 'published_parsed') else 0
            if pub_ts < cutoff_ts: continue

            if re.search(PATTERN_ACTION, entry.title.lower()) and re.search(PATTERN_SUBJECT, entry.title.lower()) and not re.search(PATTERN_EXCLUDE, entry.title.lower()):
                ticker = re.search(r'\|\s*([A-Z]+)\s+Stock News', entry.title)
                ticker = ticker.group(1) if ticker else "N/A"
                
                dt_et = datetime.fromtimestamp(pub_ts, tz=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
                pub_date_et = dt_et.strftime('%Y-%m-%d %H:%M:%S %Z')

                body_text = get_article_body(entry.link)
                event_time = analyze_event_time(entry.title, body_text) if body_text else None

                # 格式化消息
                msg = f"🚀 股票代码: {ticker}\n"
                msg += f"📅 发布时间: {pub_date_et}\n"
                if event_time:
                    msg += f"⏰ 数据公布: {event_time}\n"
                msg += f"📰 内容标题: {entry.title}\n"
                msg += f"🔗 <a href='{entry.link}'>点击查看公告</a>"
                
                send_telegram(msg)
                new_urls.append(entry.link)
                sent_urls.add(entry.link)

    with open(SENT_DB_FILE, "a") as f:
        for url in new_urls: f.write(url + "\n")

if __name__ == "__main__":
    run_monitor()
