import feedparser
import re
import time
import calendar
import cloudscraper
import random
import os
import requests
import asyncio
import subprocess
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Tuple

# ==================== 环境变量 ====================
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

# ==================== 配置 ====================
HOURS_WINDOW = 48
RSS_URLS = [
    "https://www.stocktitan.net/rss-clinical-trials",
    "https://www.stocktitan.net/rss-fda-approvals",
]
SENT_DB_FILE = "sent_urls.txt"
BPIQ_SENT_DB_FILE = "sent_bpiq.txt"

ENABLE_BPIQ = True
DAYS_BEFORE = 0
DAYS_AFTER = 1

PATTERN_ACTION = r"to (?:report|announce|discuss|showcase|present )"
PATTERN_SUBJECT = r"data|phase|result|results|topline"
PATTERN_EXCLUDE = r"financial|quarter|Q1|Q2|Q3|Q4|annual|meeting|congress|conference|symposium"

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

# ==================== 工具函数 ====================
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
        m for m in listed
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
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            if getattr(resp, "text", None) is not None:
                _ACTIVE_MODEL = mid
                print(f"选用模型: {mid}")
                return mid
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["404", "NOT_FOUND", "503", "UNAVAILABLE", "429"]):
                _SKIP_MODELS.add(mid)
                continue
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
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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
            if any(x in msg for x in ["404", "NOT_FOUND", "503", "UNAVAILABLE", "429"]):
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

def kill_browser_processes():
    try:
        subprocess.run(["pkill", "-f", "chrome"], capture_output=True)
        subprocess.run(["pkill", "-f", "chromium"], capture_output=True)
    except Exception:
        pass

# ==================== BPIQ ====================
def _month_to_num(name: str) -> Optional[int]:
    mapping = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
        'mar': 3, 'march': 3, 'apr': 4, 'april': 4, 'may': 5,
        'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }
    return mapping.get(name.lower())

def parse_catalyst_date(date_str: str) -> Optional[Tuple[str, date]]:
    if not date_str or not date_str.strip():
        return None
    original = date_str.strip()

    m = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})',
        original, re.I
    )
    if m and not re.search(r'\d{1,2}\s*-\s*\d{1,2}', original):
        month_name, day, year = m.groups()
        month = _month_to_num(month_name)
        if month:
            try:
                return original, date(int(year), month, int(day))
            except ValueError:
                pass

    m = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s*-\s*(\d{1,2}),?\s+(\d{4})',
        original, re.I
    )
    if m:
        month_name, _, day2, year = m.groups()
        month = _month_to_num(month_name)
        if month:
            try:
                return original, date(int(year), month, int(day2))
            except ValueError:
                pass

    m = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
        original, re.I
    )
    if m:
        month_name, year = m.groups()
        month = _month_to_num(month_name)
        if month:
            if month == 12:
                d = date(int(year), 12, 31)
            else:
                d = date(int(year), month + 1, 1) - timedelta(days=1)
            return original, d

    m = re.search(r'Q([1-4])\s*\'?(\d{2,4})', original, re.I)
    if m:
        q, y = m.groups()
        year = int(y) if len(y) == 4 else 2000 + int(y)
        end_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        month, day = end_day[int(q)]
        return original, date(year, month, day)

    m = re.search(r'(Summer|Mid|Late|Early|H[12])\s+(\d{4})', original, re.I)
    if m:
        season, year = m.groups()
        year = int(year)
        season = season.lower()
        mapping = {
            'summer': (8, 31), 'mid': (6, 30), 'late': (12, 31),
            'early': (3, 31), 'h1': (6, 30), 'h2': (12, 31)
        }
        month, day = mapping.get(season, (12, 31))
        return original, date(year, month, day)

    m = re.search(r'\b(20\d{2})\b', original)
    if m:
        return original, date(int(m.group(1)), 12, 31)
    return None

def translate_to_chinese(text: str) -> str:
    if not text or not text.strip():
        return text
    prompt = f"""将下面英文医药术语/适应症完整译成简洁专业中文。
只返回中文译文，不要解释、不要引号。
原文：{text}
"""
    res = gemini_text(prompt, max_tokens=120)
    return res if res else text

async def scrape_bpiq_catalysts(exclude_tickers: set) -> List[Dict]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("[BPIQ] 未安装 playwright，跳过")
        return []

    items = []
    today = date.today()
    start_date = today - timedelta(days=DAYS_BEFORE)
    end_date = today + timedelta(days=DAYS_AFTER)
    print(f"[BPIQ] 时间窗口：{start_date} ~ {end_date}")

    if not os.path.exists(BPIQ_SENT_DB_FILE):
        open(BPIQ_SENT_DB_FILE, "w").close()
    with open(BPIQ_SENT_DB_FILE, "r") as f:
        bpiq_sent = set(line.strip() for line in f)

    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            )
            page = await context.new_page()

            print("[BPIQ] 打开页面...")
            try:
                await page.goto(
                    "https://app.bpiq.com/catalyst-calendar",
                    wait_until="domcontentloaded",
                    timeout=45000
                )
            except Exception as e:
                print(f"[BPIQ] 页面加载失败: {e}")
                return []

            try:
                await page.wait_for_selector("table", timeout=20000)
                print("[BPIQ] 检测到 table")
            except Exception:
                print("[BPIQ] 未检测到 table，跳过")
                return []

            await asyncio.sleep(3)
            rows = await page.query_selector_all("table tbody tr") or await page.query_selector_all("tr")
            print(f"[BPIQ] 找到 {len(rows)} 行")

            now_et = datetime.now(ZoneInfo("America/New_York"))
            pub_date_str = format_et_cn(now_et)

            for row in rows:
                try:
                    cells = await row.query_selector_all("td")
                    if len(cells) < 5:
                        continue

                    company_text = (await cells[0].inner_text()).strip()
                    ticker = company_text.split()[0] if company_text else ""
                    if not ticker or ticker in exclude_tickers:
                        continue

                    company_link = ""
                    a_tag = await cells[0].query_selector("a")
                    if a_tag:
                        href = await a_tag.get_attribute("href")
                        if href:
                            company_link = href if href.startswith("http") else f"https://app.bpiq.com{href}"

                    drug_en = re.sub(r'\s+', ' ', (await cells[2].inner_text()).strip())
                    stage_en = re.sub(r'\s+', ' ', (await cells[3].inner_text()).strip())
                    drug_zh = translate_to_chinese(drug_en)
                    stage_zh = translate_to_chinese(stage_en)
                    title = f"{drug_zh} {stage_zh}".strip()

                    date_cell = cells[4]
                    full_text = (await date_cell.inner_text()).strip()
                    date_match = re.search(
                        r'((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:\s*-\s*\d{1,2})?,?\s+\d{4}|'
                        r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|'
                        r'Q[1-4]\s*\'?\d{2,4}|'
                        r'(?:Summer|Mid|Late|Early|H[12])\s+\d{4})',
                        full_text, re.I
                    )
                    if not date_match:
                        continue

                    date_raw = date_match.group(1)
                    parsed = parse_catalyst_date(date_raw)
                    if not parsed:
                        continue
                    original_str, compare_date = parsed

                    if not (start_date <= compare_date <= end_date):
                        continue

                    dedup_key = f"{ticker}|{original_str}"
                    if dedup_key in bpiq_sent:
                        continue

                    source_link = company_link
                    links = await date_cell.query_selector_all("a")
                    for link in links:
                        href = await link.get_attribute("href")
                        text = ((await link.inner_text()) or "").lower()
                        if href and ("view source" in text or any(x in href for x in ["globenewswire", "prnewswire", "sec.gov", "businesswire"])):
                            source_link = href
                            break
                        if href and href.startswith("http") and not source_link:
                            source_link = href

                    # 统一成 StockTitan 字段结构
                    items.append({
                        "ticker": ticker,
                        "pub_date": pub_date_str,
                        "event_time": original_str,
                        "title": title,
                        "link": source_link or company_link or "https://app.bpiq.com/catalyst-calendar",
                        "dedup_key": dedup_key,
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"[BPIQ] 异常: {e}")
        return []
    finally:
        try:
            if browser:
                await browser.close()
        except Exception:
            pass
        kill_browser_processes()

    return items

def push_combined(items: List[Dict], bpiq_keys: List[str]):
    """合并推送，格式与 StockTitan 一致"""
    if not items:
        print("无符合条件的事件")
        return

    now_et = datetime.now(ZoneInfo("America/New_York"))
    header = (
        f"🚨<b>{now_et.month}月{now_et.day}日医药股数据发布预警"
        f"（共{len(items)}条）</b>\n\n"
    )
    footer = "\n#ClinicalData"
    full_msg = header

    for i, item in enumerate(items, 1):
        item_str = f"{i}. 🚀股票代码: ${item['ticker']}\n"
        item_str += f"   📅新闻时间: {item['pub_date']}\n"
        if item.get("event_time"):
            item_str += f"   ⏰公布时间: {item['event_time']}\n"
        item_str += f"   📰内容标题: {item['title']}\n"
        item_str += f"   🔗<a href='{item['link']}'>点击查看公告</a>\n"
        if i < len(items):
            item_str += "--------------------------------\n"

        if len(full_msg) + len(item_str) + len(footer) > 3900:
            send_telegram(full_msg + footer)
            full_msg = "接上条续：\n\n" + item_str
        else:
            full_msg += item_str

    send_telegram(full_msg + footer)

    if bpiq_keys:
        with open(BPIQ_SENT_DB_FILE, "a") as f:
            for key in bpiq_keys:
                f.write(key + "\n")

    print(f"成功推送 {len(items)} 条（其中 BPIQ {len(bpiq_keys)} 条）")

# ==================== 主流程 ====================
def run_monitor():
    current_utc_ts = time.time()
    cutoff_ts = current_utc_ts - (HOURS_WINDOW * 3600)

    if not os.path.exists(SENT_DB_FILE):
        open(SENT_DB_FILE, "w").close()
    with open(SENT_DB_FILE, "r") as f:
        sent_urls = set(line.strip() for line in f)

    collected_items = []
    new_urls = []
    rss_tickers = set()

    # ---- 1. StockTitan ----
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
                rss_tickers.add(ticker)

                english_title = clean_title(title)
                chinese_title = translate_title(english_title)
                dt_et = datetime.fromtimestamp(pub_ts, tz=ZoneInfo("UTC")).astimezone(
                    ZoneInfo("America/New_York")
                )
                pub_date_et = format_et_cn(dt_et)
                body_text = get_article_body(entry.link)
                event_time = analyze_event_time(english_title, body_text) if body_text else None

                collected_items.append({
                    "ticker": ticker,
                    "pub_date": pub_date_et,
                    "event_time": event_time,
                    "title": chinese_title,
                    "link": entry.link,
                })
                new_urls.append(entry.link)

    # ---- 2. BPIQ（带超时） ----
    bpiq_keys = []
    if ENABLE_BPIQ:
        print("\n[BPIQ] 开始抓取...")
        try:
            bpiq_items = asyncio.run(
                asyncio.wait_for(
                    scrape_bpiq_catalysts(exclude_tickers=rss_tickers),
                    timeout=120
                )
            )
            for it in bpiq_items:
                collected_items.append(it)
                bpiq_keys.append(it["dedup_key"])
        except asyncio.TimeoutError:
            print("[BPIQ] 超时，跳过")
        except Exception as e:
            print(f"[BPIQ] 失败: {e}")
        finally:
            kill_browser_processes()
    else:
        print("[BPIQ] 已关闭")

    # ---- 3. 合并推送 ----
    if collected_items:
        push_combined(collected_items, bpiq_keys)
        with open(SENT_DB_FILE, "a") as f:
            for url in new_urls:
                f.write(url + "\n")
    else:
        print("未发现满足条件的新条目（RSS 与 BPIQ 均为空）", flush=True)
        print(f"RSS 条数相关: new_urls={len(new_urls)}, rss_tickers={rss_tickers}", flush=True)
        print(f"BPIQ keys 数: {len(bpiq_keys)}", flush=True)

if __name__ == "__main__":
    try:
        run_monitor()
    except Exception as e:
        print(f"主流程异常: {e}", flush=True)
    finally:
        kill_browser_processes()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        # 给 Telegram / 日志一点收尾时间，再退出
        time.sleep(2)
        os._exit(0)
