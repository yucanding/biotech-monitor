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
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")

TELEGRAM_TARGETS = [
    x.strip()
    for x in (TELEGRAM_CHAT_ID, TELEGRAM_USER_ID)
    if x and x.strip()
]

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

# 优先更空闲的 lite / latest，把容易 503 的 3.8 放后面
MODEL_PREFER = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "desktop": True}
)
client = genai.Client(api_key=GEMINI_API_KEY)
_ACTIVE_MODEL = None
_SKIP_MODELS = set()


def _model_id(name: str) -> str:
    return name.split("/")[-1] if name else ""


def list_generate_models():
    ids = []
    try:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions:
                mid = _model_id(getattr(m, "name", "") or "")
                if mid:
                    ids.append(mid)
    except Exception as e:
        print(f"列出模型失败，改用本地候选: {e}")
    return ids


def pick_model():
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL and _ACTIVE_MODEL not in _SKIP_MODELS:
        return _ACTIVE_MODEL

    listed = list_generate_models()
    print("账号可见 generateContent 模型:", listed or "(空，用本地候选)")

    flash_listed = [
        m
        for m in listed
        if "flash" in m.lower()
        and "embed" not in m.lower()
        and "image" not in m.lower()
        and "tts" not in m.lower()
        and "omni" not in m.lower()
        and "transcribe" not in m.lower()
    ]

    candidates = []
    for m in MODEL_PREFER + flash_listed:
        if m not in candidates and m not in _SKIP_MODELS:
            candidates.append(m)

    for mid in candidates:
        try:
            resp = client.models.generate_content(
                model=mid,
                contents="ok",
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            if getattr(resp, "text", None) is not None:
                _ACTIVE_MODEL = mid
                print(f"选用模型: {mid}")
                return mid
        except Exception as e:
            msg = str(e)
            if "404" in msg or "NOT_FOUND" in msg:
                print(f"跳过不可用模型 {mid}")
                _SKIP_MODELS.add(mid)
                continue
            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                print(f"{mid} 限流/高峰，换下一个")
                _SKIP_MODELS.add(mid)
                continue
            print(f"{mid} 探测失败: {e}")
            _SKIP_MODELS.add(mid)

    raise RuntimeError("当前没有可用的 Gemini 文本模型，请稍后重跑")


def gemini_text(prompt: str, max_tokens: int = 300) -> str:
    global _ACTIVE_MODEL
    last_err = None
    for attempt in range(6):
        try:
            mid = pick_model()
        except RuntimeError as e:
            last_err = e
            break
        try:
            resp = client.models.generate_content(
                model=mid,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=max_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            text = (resp.text or "").strip()
            if text:
                return text
            last_err = "empty text"
        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"Gemini 调用失败({mid}): {e}")
            if (
                "404" in msg
                or "NOT_FOUND" in msg
                or "503" in msg
                or "UNAVAILABLE" in msg
                or "429" in msg
            ):
                _SKIP_MODELS.add(mid)
                _ACTIVE_MODEL = None
                time.sleep(1.2 * (attempt + 1))
                continue
            break
    print(f"Gemini 最终失败: {last_err}")
    return ""


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    if not TELEGRAM_TARGETS:
        print("未配置任何 Telegram 目标")
        return
    for chat_id in TELEGRAM_TARGETS:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code != 200:
                print(f"发送到 {chat_id} 失败: {r.text}")
        except Exception as e:
            print(f"发送到 {chat_id} 失败: {e}")


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
    prompt = f"""你是医药新闻信息抽取器。
从标题和正文中提取「临床数据 / 试验结果 / topline / readout」即将公布或计划召开电话会的日期和时间。

规则：
1. 只提取未来或即将发生的数据公布/电话会时间，不要提取新闻发布日期本身。
2. 输出必须用中文，例如：2026年9月15日 上午8:30；或 2026年四季度；或 2026年ASCO年会。
3. 有钟点就写上午/下午+点分，并注明时区，如ET/PT。
4. 只有季度或会议名时，用中文写时间窗口。
5. 完全没有公布时间则只回复 NONE，不要解释。
6. 只返回这一行时间，不要引号，不要前后缀。

Title: {title}
Body: {body or ""}
"""
    res = gemini_text(prompt, max_tokens=80)
    if not res or "NONE" in res.upper():
        return None
    return res


def _looks_truncated_zh(text: str, source_en: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if t.endswith(("讨论", "宣布", "报告", "召开", "关于", "以及", "与", "的")):
        return True
    if t.endswith((",", "，", ":", "：", "...", "…")):
        return True
    if len(t) < max(8, int(len(source_en) * 0.25)):
        return True
    return False


def translate_title(title):
    prompt = f"""将下面英文医药财经新闻标题完整译成简洁专业中文。
要求：
- 只返回完整中文译文，不要解释、不要引号、不要拼音。
- 必须译完整句，禁止截断后半句。
- 保留公司名、药名、试验代号、Phase 1/2/3、FDA、topline 等专业词的惯用译法或原文。
- 不要把股票代码后缀译出来。

标题：{title}
"""
    res = gemini_text(prompt, max_tokens=300)
    if not res or _looks_truncated_zh(res, title):
        print(f"译文不可用，回退英文标题: {title}")
        return title
    return res


def clean_title(title):
    return re.sub(r"\s*\|\s*[A-Z]+\s+Stock News", "", title)


def format_et_cn(dt):
    return f"{dt.year}年{dt.month}月{dt.day}日 {dt.strftime('%H:%M')} ET"


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
                pub_date_et = format_et_cn(dt_et)

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
        header = (
            f"🚨<b>{now_et.month}月{now_et.day}日医药股数据发布预警"
            f"（共{len(collected_items)}条）</b>\n\n"
        )
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
        print(
            f"成功推送 {len(collected_items)} 条新闻至 {len(TELEGRAM_TARGETS)} 个目标。"
        )
    else:
        print("未发现满足条件的新条目。")


if __name__ == "__main__":
    run_monitor()
