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
from google import genai
from google.genai import types

# --- 环境变量 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not GEMINI_API_KEY:
    raise RuntimeError("请先设置环境变量 GEMINI_API_KEY")

# --- 配置 ---
HOURS_WINDOW = 48
RSS_URLS = [
    "https://www.stocktitan.net/rss-clinical-trials",
    "https://www.stocktitan.net/rss-fda-approvals",
]
SENT_DB_FILE = "sent_urls.txt"

PATTERN_ACTION = r"to (?:report|announce|discuss|showcase )"
PATTERN_SUBJECT = r"data|phase|result|results|topline"
PATTERN_EXCLUDE = r"financial|quarter|Q1|Q2|Q3|Q4"

# Flash 够快、便宜，抽取/翻译足够；抽不准再换成 gemini-2.5-pro / gemini-3.8-flash
GEMINI_MODEL = "gemini-2.5-flash"

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "desktop": True}
)
client = genai.Client(api_key=GEMINI_API_KEY)


def gemini_text(prompt: str, max_tokens: int = 200) -> str:
    """统一封装 Gemini 调用，失败返回空字符串。"""
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
            ),
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"Gemini 调用失败: {e}")
        return ""


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"发送失败: {e}")


def get_article_body(url):
    try:
        time.sleep(random.uniform(2, 4))
        response = scraper.get(url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            body = soup.find("div", class_="article-body") or soup.find("article")
            if body:
                return body.get_text(separator=" ", strip=True)[:3000]
        return None
    except Exception as e:
        print(f"抓取正文失败: {e}")
        return None


def analyze_event_time(title, body):
    """抽取即将公布临床数据的具体日期/时间。"""
    prompt = f"""你是医药新闻信息抽取器。
从标题和正文中提取「临床数据 / 试验结果 / topline / readout」即将公布或计划公布的日期和时间。

规则：
1. 只提取未来或即将发生的数据公布时间，不要提取新闻发布日期本身。
2. 有明确日期就输出，例如：2026-10-15 或 October 15, 2026 before market 或 Q4 2026。
3. 若只有季度/会议名（如 ASCO 2026、Q4 2026），原样输出该时间窗口。
4. 完全没有公布时间则只回复 NONE，不要解释。
5. 不要翻译，不要加引号，不要加任何前后缀。

Title: {title}
Body: {body or ""}
"""
    res = gemini_text(prompt, max_tokens=80)
    if not res or "NONE" in res.upper():
        return None
    return res


def translate_title(title):
    """医药财经标题译成简洁专业中文。"""
    prompt = f"""将下面这条英文医药/生物科技财经新闻标题译成简洁、专业的中文。
要求：
- 只返回中文译文，不要解释、不要引号、不要拼音。
- 保留公司名、药名、试验代号、Phase 1/2/3、FDA、topline 等专业词的惯用译法或原文。
- 不要把股票代码后缀译出来。

标题：{title}
"""
    res = gemini_text(prompt, max_tokens=200)
    return res if res else title


def clean_title(title):
    return re.sub(r"\s*\|\s*[A-Z]+\s+Stock News", "", title)


def run_monitor():
    current_utc_ts = time.time()
    cutoff_ts = current_utc_ts - (HOURS_WINDOW * 3600)

    if not os.path.exists(SENT_DB_FILE):
        open(SENT_DB_FILE, "w").close()
    with open(SENT_DB_FILE, "r") as f:
        sent_urls = set(line.strip() for line in f)

    collected_items = []
    new_urls = []

    for rss_url in RSS_URLS:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if entry.link in sent_urls or entry.link in new_urls:
                continue

            pub_ts = (
                calendar.timegm(entry.published_parsed)
                if hasattr(entry, "published_parsed")
                else 0
            )
            if pub_ts < cutoff_ts:
                continue

            title = entry.title
            title_lower = title.lower()
            if (
                re.search(PATTERN_ACTION, title_lower)
                and re.search(PATTERN_SUBJECT, title_lower)
                and not re.search(PATTERN_EXCLUDE, title_lower)
            ):
                ticker_match = re.search(r"\|\s*([A-Z]+)\s+Stock News", title)
                ticker = ticker_match.group(1) if ticker_match else "N/A"

                english_title = clean_title(title)
                chinese_title = translate_title(english_title)

                dt_et = datetime.fromtimestamp(pub_ts, tz=ZoneInfo("UTC")).astimezone(
                    ZoneInfo("America/New_York")
                )
                pub_date_et = dt_et.strftime("%Y-%m-%d %H:%M:%S %Z")

                body_text = get_article_body(entry.link)
                event_time = (
                    analyze_event_time(english_title, body_text) if body_text else None
                )

                collected_items.append(
                    {
                        "ticker": ticker,
                        "pub_date": pub_date_et,
                        "event_time": event_time,
                        "title": chinese_title,
                        "link": entry.link,
                    }
                )
                new_urls.append(entry.link)

    if collected_items:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        header = f"🚨<b>{now_et.month}月{now_et.day}日医药股数据发布预警（共{len(collected_items)}条）</b>\n\n"
        footer = "\n#ClinicalData"

        full_msg = header
        for i, item in enumerate(collected_items, 1):
            item_str = f"{i}. 🚀股票代码: ${item['ticker']}\n"
            item_str += f"   📅新闻时间: {item['pub_date']}\n"
            if item["event_time"]:
                item_str += f"   ⏰公布时间: {item['event_time']}\n"
            item_str += f"   📰内容标题: {item['title']}\n"
            item_str += f"   🔗<a href='{item['link']}'>点击查看公告</a>\n"
            if i < len(collected_items):
                item_str += "--------------------------------\n"

            if len(full_msg) + len(item_str) + len(footer) > 3900:
                send_telegram(full_msg + footer)
                full_msg = "接上条续：\n\n" + item_str
            else:
                full_msg += item_str

        send_telegram(full_msg + footer)
        with open(SENT_DB_FILE, "a") as f:
            for url in new_urls:
                f.write(url + "\n")
        print(f"成功推送 {len(collected_items)} 条新闻至频道。")
    else:
        print("未发现满足条件的新条目。")


if __name__ == "__main__":
    run_monitor()
