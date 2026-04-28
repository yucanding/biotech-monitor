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

# --- 环境变量读取 ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- 配置区 ---
HOURS_WINDOW = 96
RSS_URLS = [
    "https://www.stocktitan.net/rss-clinical-trials",
    "https://www.stocktitan.net/rss-fda-approvals"
]
SENT_DB_FILE = "sent_urls.txt"

# 动作词/内容词：严格锁定
PATTERN_ACTION = r"to (?:report|announce|discuss)"
PATTERN_SUBJECT = r"data|phase"
PATTERN_EXCLUDE = r"financial|quarter|Q1"

scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

def send_telegram(message):
    """发送 Telegram 消息"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True # 汇总信息建议禁用预览，避免太乱
    }
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"发送失败: {e}")

def get_article_body(url):
    """抓取正文"""
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
    """AI 分析具体数据发布日期"""
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

def clean_title(title):
    """过滤掉标题末尾的股票代码后缀"""
    return re.sub(r'\s*\|\s*[A-Z]+\s+Stock News', '', title)

def run_monitor():
    current_utc_ts = time.time()
    cutoff_ts = current_utc_ts - (HOURS_WINDOW * 3600)
    
    # 加载已发送数据库
    if not os.path.exists(SENT_DB_FILE): open(SENT_DB_FILE, "w").close()
    with open(SENT_DB_FILE, "r") as f: sent_urls = set(line.strip() for line in f)

    collected_items = [] # 用于暂存所有新发现的新闻
    new_urls = []

    for rss_url in RSS_URLS:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if entry.link in sent_urls or entry.link in new_urls:
                continue
            
            pub_ts = calendar.timegm(entry.published_parsed) if hasattr(entry, 'published_parsed') else 0
            if pub_ts < cutoff_ts: continue

            title = entry.title
            title_lower = title.lower()

            if re.search(PATTERN_ACTION, title_lower) and \
               re.search(PATTERN_SUBJECT, title_lower) and \
               not re.search(PATTERN_EXCLUDE, title_lower):
                
                # 提取代码
                ticker_match = re.search(r'\|\s*([A-Z]+)\s+Stock News', title)
                ticker = ticker_match.group(1) if ticker_match else "N/A"
                
                # 过滤标题
                display_title = clean_title(title)
                
                # 时间转换 (ET)
                dt_et = datetime.fromtimestamp(pub_ts, tz=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
                pub_date_et = dt_et.strftime('%Y-%m-%d %H:%M:%S %Z')

                # AI 分析
                body_text = get_article_body(entry.link)
                event_time = analyze_event_time(title, body_text) if body_text else None

                # 存入列表
                collected_items.append({
                    "ticker": ticker,
                    "pub_date": pub_date_et,
                    "event_time": event_time,
                    "title": display_title,
                    "link": entry.link
                })
                new_urls.append(entry.link)

    # --- 批量构造并发送 ---
    if collected_items:
        # 获取当前美东日期用于 Header
        now_et = datetime.now(ZoneInfo("America/New_York"))
        header = f"🚨 {now_et.month}月{now_et.day}日医药股数据发布预警（共{len(collected_items)}条）\n\n"
        
        full_msg = header
        for i, item in enumerate(collected_items, 1):
            item_str = f"{i}. 🚀股票代码: {item['ticker']}\n"
            item_str += f"   📅新闻发布: {item['pub_date']}\n"
            if item['event_time']:
                item_str += f"   ⏰公布时间: {item['event_time']}\n"
            item_str += f"   📰内容标题: {item['title']}\n"
            item_str += f"   🔗<a href='{item['link']}'>点击查看公告</a>\n"
            item_str += "--------------------------------\n"
            
            # Telegram 限制一条消息约 4096 字符，若单条太长则分批发送
            if len(full_msg) + len(item_str) > 3800:
                send_telegram(full_msg)
                full_msg = "接上条续：\n\n" + item_str
            else:
                full_msg += item_str
        
        send_telegram(full_msg)

        # 更新已发送列表
        with open(SENT_DB_FILE, "a") as f:
            for url in new_urls: f.write(url + "\n")
        print(f"成功推送 {len(collected_items)} 条新闻至频道。")
    else:
        print("未发现满足条件的新条目。")

if __name__ == "__main__":
    run_monitor()
