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
from urllib.parse import urljoin
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Tuple

try:
    import translators as ts
except ImportError:
    ts = None

# ==================== 环境变量 ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")
TELEGRAM_TARGETS = [x.strip() for x in (TELEGRAM_CHAT_ID, TELEGRAM_USER_ID) if x and x.strip()]
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
RTTNEWS_SENT_DB_FILE = "sent_rttnews.txt"
ENABLE_BPIQ = True
ENABLE_RTTNEWS = True
RTTNEWS_DAYS_BEFORE = 0
RTTNEWS_DAYS_AFTER = 1
RTTNEWS_URLS = [
    "https://www.rttnews.com/corpinfo/fdacalendar.aspx?PageNum=5",
    "https://www.rttnews.com/corpinfo/fdacalendar.aspx?PageNum=6",
    "https://www.rttnews.com/corpinfo/fdacalendar.aspx?PageNum=7",
]
DAYS_BEFORE = 0
DAYS_AFTER = 1
PATTERN_ACTION = r"to (?:report|announce|discuss|showcase|present )"
PATTERN_SUBJECT = r"data|phase|result|results|topline"
PATTERN_EXCLUDE = r"financial|quarter|Q1|Q2|Q3|Q4|annual|meeting|congress|conference|symposium"
MODEL_PREFER = [
    "gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.5-flash",
    "gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.8-flash",
]
scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "desktop": True})
client = genai.Client(api_key=GEMINI_API_KEY)
_ACTIVE_MODEL = None
_SKIP_MODELS = set()

# ==================== 通用工具 ====================
def ensure_file(path: str):
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()

def read_set(path: str) -> set:
    ensure_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def append_lines(path: str, lines: List[str]):
    if not lines:
        return
    existing = read_set(path)
    new_lines = []
    for line in lines:
        line = str(line).strip()
        if line and line not in existing:
            existing.add(line)
            new_lines.append(line)
    if new_lines:
        with open(path, "a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")

# ==================== Gemini ====================
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
        print(f"列出模型失败，改用本地候选: {e}", flush=True)
    return ids

def pick_model():
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL and _ACTIVE_MODEL not in _SKIP_MODELS:
        return _ACTIVE_MODEL
    listed = list_generate_models()
    print("账号可见 generateContent 模型:", listed or "(空，用本地候选)", flush=True)
    flash_listed = [
        m for m in listed
        if "flash" in m.lower() and "embed" not in m.lower() and "image" not in m.lower()
        and "tts" not in m.lower() and "omni" not in m.lower() and "transcribe" not in m.lower()
    ]
    candidates = []
    for m in MODEL_PREFER + flash_listed:
        if m not in candidates and m not in _SKIP_MODELS:
            candidates.append(m)
    for mid in candidates:
        try:
            resp = client.models.generate_content(
                model=mid, contents="ok",
                config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=8,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            if getattr(resp, "text", None) is not None:
                _ACTIVE_MODEL = mid
                print(f"选用模型: {mid}", flush=True)
                return mid
        except Exception:
            _SKIP_MODELS.add(mid)
            continue
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
                model=mid, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1, max_output_tokens=max_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            result = (resp.text or "").strip()
            if result:
                return result
            last_err = "empty text"
        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"Gemini 调用失败({mid}): {e}", flush=True)
            if any(x in msg for x in ["404", "NOT_FOUND", "503", "UNAVAILABLE", "429"]):
                _SKIP_MODELS.add(mid)
                _ACTIVE_MODEL = None
                time.sleep(1.2 * (attempt + 1))
                continue
            break
    print(f"Gemini 最终失败: {last_err}", flush=True)
    return ""

# ==================== Telegram ====================
def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    if not TELEGRAM_TARGETS:
        print("未配置任何 Telegram 目标", flush=True)
        return False
    all_success = True
    for chat_id in TELEGRAM_TARGETS:
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code != 200:
                all_success = False
                print(f"发送到 {chat_id} 失败: {r.text}", flush=True)
                continue
            try:
                response_json = r.json()
                if not response_json.get("ok", False):
                    all_success = False
                    print(f"发送到 {chat_id} 失败: {response_json}", flush=True)
            except Exception:
                all_success = False
                print(f"发送到 {chat_id} 返回无法解析的响应", flush=True)
        except Exception as e:
            all_success = False
            print(f"发送到 {chat_id} 失败: {e}", flush=True)
    return all_success

# ==================== StockTitan ====================
def get_article_body(url: str) -> Optional[str]:
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
        print(f"抓取正文失败: {e}", flush=True)
        return None

def analyze_event_time(title: str, body: Optional[str]) -> Optional[str]:
    prompt = f"""你是医药新闻信息抽取器。
从标题和正文中提取「临床数据 / 试验结果 / topline / readout」即将公布或计划召开电话会的日期和时间。
规则：
1. 只提取未来或即将发生的数据公布/电话会时间，不要提取新闻发布日期本身。
2. 输出必须用中文，例如：2026年9月15日 上午8:30；或 2026年四季度；或 2026年ASCO年会。
3. 有钟点就写上午/下午+点分，并注明时区，如ET/PT。
4. 只有季度或会议名时，用中文写时间窗口。
5. 完全没有公布时间则只回复 NONE，不要解释。
6. 只返回这一行时间，不要引号、不要前后缀。
Title: {title}
Body: {body or ""}"""
    result = gemini_text(prompt, max_tokens=80)
    if not result or "NONE" in result.upper():
        return None
    return result

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

def translate_title(title: str) -> str:
    prompt = f"""将下面英文医药财经新闻标题完整译成简洁专业中文。
要求：
- 只返回完整中文译文，不要解释、不要引号、不要拼音。
- 必须译完整句，禁止截断后半句。
- 保留公司名、药名、试验代号、Phase 1/2/3、FDA、topline 等专业词的惯用译法或原文。
- 不要把股票代码后缀译出来。
标题：{title}"""
    result = gemini_text(prompt, max_tokens=300)
    if not result or _looks_truncated_zh(result, title):
        print(f"译文不可用，回退英文标题: {title}", flush=True)
        return title
    return result

def clean_title(title: str) -> str:
    return re.sub(r"\s*\|\s*[A-Z]+\s+Stock News", "", title)

def parse_rss_event_date(event_time: str) -> Optional[date]:
    if not event_time or not str(event_time).strip():
        return None
    text = str(event_time).strip()
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        y, mo = map(int, m.groups())
        try:
            if mo == 12:
                return date(y, 12, 31)
            return date(y, mo + 1, 1) - timedelta(days=1)
        except ValueError:
            return None
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(20\d{2})",
        text, re.I,
    )
    if m:
        month_name, day_num, year = m.groups()
        month = _month_to_num(month_name)
        if month:
            try:
                return date(int(year), month, int(day_num))
            except ValueError:
                return None
    m = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", text)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    return None

# ==================== BPIQ 清洗 / 翻译 ====================
def clean_scraped_cell(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"Log in or subscribe to Elite to view.*$", "", text, flags=re.I)
    text = re.sub(r"登录或订阅\s*Elite.*$", "", text, flags=re.I)
    junk_patterns = [
        r"Mover Analysis.*$", r"Full Analysis.*$", r"Simplified metrics.*$",
        r"Odds lean.*$", r"its position is.*$", r"Center-right means.*$",
        r"Center-left means.*$", r"Negative\s+\d+.*Positive\s+\d+.*$",
        r"\d{1,3}[–\-]\d{1,3}%\s*(Negative|Mixed|Positive).*stock move.*$",
        r"stock (move|fluctuation).*$",
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"移动者分析.*$", "", text)
    text = re.sub(r"完整分析.*$", "", text)
    text = re.sub(r"精简指标.*$", "", text)
    text = re.sub(r"赔率倾向.*$", "", text)
    text = re.sub(r"负面\s*\d+.*正面\s*\d+.*$", "", text)
    text = re.sub(r"股票波动.*$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—,:;/|&")
    return text

def is_junk_title(title: str) -> bool:
    if not title:
        return True
    t = title.lower()
    markers = [
        "elite", "mover analysis", "股票波动", "赔率", "移动者分析",
        "精简指标", "完整分析", "log in or subscribe", "登录或订阅",
    ]
    return any(marker in t for marker in markers)

def translate_bpiq_text(text: str) -> str:
    if not text or not text.strip():
        return text
    if ts is None:
        print("[BPIQ] 未安装 translators，返回英文原文", flush=True)
        return text
    engines = ["bing", "alibaba", "baidu", "sogou", "tencent"]
    for engine in engines:
        try:
            result = ts.translate_text(text, translator=engine, from_language="en", to_language="zh")
            if result and result.strip():
                return result.strip()
        except Exception:
            continue
    print(f"[BPIQ] 翻译失败，返回原文: {text[:60]}...", flush=True)
    return text

def clean_stage_event(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\bdata\s+readout\b", "", text, flags=re.I)
    text = re.sub(r"\bapproval\s+decision\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\s*[-–—,:;/|&]+\s*", "", text)
    text = re.sub(r"\s*[-–—,:;/|&]+\s*$", "", text)
    return text.strip()

# ==================== BPIQ 日期解析 / 标准化 ====================
def _month_to_num(name: str) -> Optional[int]:
    mapping = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    return mapping.get(name.lower())

def parse_catalyst_date_range(date_str: str) -> Optional[Tuple[date, date, str]]:
    if not date_str or not date_str.strip():
        return None
    original = re.sub(r"\s+", " ", date_str.strip())
    month_pattern = (
        r"January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    )
    # 跨月份日期范围
    m = re.search(
        rf"({month_pattern})\s+(\d{{1,2}})\s*-\s*({month_pattern})\s+(\d{{1,2}}),?\s+(\d{{4}})",
        original, re.I,
    )
    if m:
        mon1, day1, mon2, day2, year = m.groups()
        try:
            start = date(int(year), _month_to_num(mon1), int(day1))
            end = date(int(year), _month_to_num(mon2), int(day2))
            if end < start:
                return None
            display = f"{start.year}年{start.month}月{start.day}日–{end.month}月{end.day}日"
            return start, end, display
        except (ValueError, TypeError):
            return None
    # 同月份日期范围
    m = re.search(rf"({month_pattern})\s+(\d{{1,2}})\s*-\s*(\d{{1,2}}),?\s+(\d{{4}})", original, re.I)
    if m:
        mon, day1, day2, year = m.groups()
        try:
            month = _month_to_num(mon)
            start = date(int(year), month, int(day1))
            end = date(int(year), month, int(day2))
            if end < start:
                return None
            display = f"{start.year}年{start.month}月{start.day}日–{end.day}日"
            return start, end, display
        except (ValueError, TypeError):
            return None
    # 精确日期
    m = re.search(rf"({month_pattern})\s+(\d{{1,2}}),?\s+(\d{{4}})", original, re.I)
    if m:
        mon, day_num, year = m.groups()
        try:
            d = date(int(year), _month_to_num(mon), int(day_num))
            return d, d, f"{d.year}年{d.month}月{d.day}日"
        except (ValueError, TypeError):
            return None
    # 月份
    m = re.search(rf"({month_pattern})\s+(\d{{4}})", original, re.I)
    if m:
        mon, year = m.groups()
        try:
            month = _month_to_num(mon)
            year = int(year)
            start = date(year, month, 1)
            end = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
            return start, end, f"{year}年{month}月"
        except (ValueError, TypeError):
            return None
    # 季度
    m = re.search(r"\bQ([1-4])\s*'?(\d{2,4})\b", original, re.I)
    if m:
        q, y = m.groups()
        q = int(q)
        year = int(y) if len(y) == 4 else 2000 + int(y)
        periods = {1: (1, 3, 31, "第一季度"), 2: (4, 6, 30, "第二季度"), 3: (7, 9, 30, "第三季度"), 4: (10, 12, 31, "第四季度")}
        start_month, end_month, end_day, label = periods[q]
        return date(year, start_month, 1), date(year, end_month, end_day), f"{year}年{label}"
    # 半年、季节及模糊时间
    m = re.search(r"\b(Summer|Mid|Late|Early|H[12])\s+(\d{4})\b", original, re.I)
    if m:
        period, y = m.groups()
        year = int(y)
        period = period.lower()
        mapping = {
            "h1": (date(year, 1, 1), date(year, 6, 30), "上半年"),
            "h2": (date(year, 7, 1), date(year, 12, 31), "下半年"),
            "summer": (date(year, 6, 1), date(year, 8, 31), "夏季"),
            "early": (date(year, 1, 1), date(year, 3, 31), "年初"),
            "mid": (date(year, 4, 1), date(year, 8, 31), "年中"),
            "late": (date(year, 10, 1), date(year, 12, 31), "年末"),
        }
        start, end, label = mapping[period]
        return start, end, f"{year}年{label}"
    return None

def normalize_event_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def make_bpiq_dedup_key(ticker: str, drug: str, stage: str, start_date: date, end_date: date) -> str:
    return "|".join([
        ticker.upper().strip(), normalize_event_text(drug), normalize_event_text(stage),
        start_date.isoformat(), end_date.isoformat(),
    ])

# ==================== BPIQ 抓取 ====================
async def scrape_bpiq_catalysts() -> List[Dict]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("[BPIQ] 未安装 playwright，跳过", flush=True)
        return []
    items = []
    today = date.today()
    start_date = today - timedelta(days=DAYS_BEFORE)
    end_date = today + timedelta(days=DAYS_AFTER)
    print(f"[BPIQ] 时间窗口：{start_date} ~ {end_date}", flush=True)
    bpiq_sent = read_set(BPIQ_SENT_DB_FILE)
    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            )
            page = await context.new_page()
            print("[BPIQ] 打开页面...", flush=True)
            try:
                await page.goto("https://app.bpiq.com/catalyst-calendar", wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                print(f"[BPIQ] 页面加载失败: {e}", flush=True)
                return []
            try:
                await page.wait_for_selector("table", timeout=20000)
                print("[BPIQ] 检测到 table", flush=True)
            except Exception:
                print("[BPIQ] 未检测到 table，跳过", flush=True)
                return []
            await asyncio.sleep(3)
            rows = await page.query_selector_all("table tbody tr") or await page.query_selector_all("tr")
            print(f"[BPIQ] 找到 {len(rows)} 行", flush=True)
            candidates = []
            month_pattern = (
                r"January|February|March|April|May|June|July|August|September|October|November|December|"
                r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
            )
            date_pattern = (
                rf"({month_pattern}\s+\d{{1,2}}\s*-\s*(?:(?:{month_pattern})\s+)?\d{{1,2}},?\s+\d{{4}}|"
                rf"{month_pattern}\s+\d{{1,2}},?\s+\d{{4}}|{month_pattern}\s+\d{{4}}|"
                rf"Q[1-4]\s*'?\d{{2,4}}|(?:Summer|Mid|Late|Early|H[12])\s+\d{{4}})"
            )
            for row in rows:
                try:
                    cells = await row.query_selector_all("td")
                    if len(cells) < 5:
                        continue
                    company_text = (await cells[0].inner_text()).strip()
                    ticker = company_text.split()[0] if company_text else ""
                    if not ticker or ticker == "N/A" or len(ticker) > 4:
                        continue
                    company_link = ""
                    a_tag = await cells[0].query_selector("a")
                    if a_tag:
                        href = await a_tag.get_attribute("href")
                        if href:
                            company_link = href if href.startswith("http") else f"https://app.bpiq.com{href}"
                    drug_en = clean_scraped_cell(re.sub(r"\s+", " ", (await cells[2].inner_text()).strip()))
                    stage_en = clean_scraped_cell(re.sub(r"\s+", " ", (await cells[3].inner_text()).strip()))
                    if not drug_en:
                        continue
                    date_cell = cells[4]
                    full_text = (await date_cell.inner_text()).strip()
                    date_match = re.search(date_pattern, full_text, re.I)
                    if not date_match:
                        continue
                    date_raw = date_match.group(1)
                    parsed = parse_catalyst_date_range(date_raw)
                    if not parsed:
                        continue
                    date_start, date_end, date_zh = parsed
                    if date_end < start_date or date_start > end_date:
                        continue
                    dedup_key = make_bpiq_dedup_key(ticker, drug_en, stage_en, date_start, date_end)
                    legacy_key = f"{ticker}|{date_raw}"
                    if dedup_key in bpiq_sent or legacy_key in bpiq_sent:
                        continue
                    source_link = company_link
                    links = await date_cell.query_selector_all("a")
                    for link in links:
                        href = await link.get_attribute("href")
                        link_text = ((await link.inner_text()) or "").lower()
                        if href and ("view source" in link_text or any(d in href for d in ["globenewswire", "prnewswire", "sec.gov", "businesswire"])):
                            source_link = href
                            break
                        if href and href.startswith("http") and not source_link:
                            source_link = href
                    candidates.append({
                        "ticker": ticker, "drug_en": drug_en, "stage_en": stage_en,
                        "date_raw": date_raw, "date_zh": date_zh, "date_start": date_start,
                        "date_end": date_end, "link": source_link or company_link or "https://app.bpiq.com/catalyst-calendar",
                        "dedup_key": dedup_key,
                    })
                except Exception:
                    continue
            print(f"[BPIQ] 日期过滤后剩余 {len(candidates)} 条，开始翻译（非AI）...", flush=True)
            for c in candidates:
                drug_zh = translate_bpiq_text(c["drug_en"])
                stage_en = clean_stage_event(c["stage_en"])
                title = f"{drug_zh}{stage_en}".strip()
                if is_junk_title(title):
                    print(f"[BPIQ] 跳过废话标题: {c['ticker']} | {title[:80]}...", flush=True)
                    continue
                items.append({
                    "ticker": c["ticker"], "event_time": c["date_zh"], "title": title, "link": c["link"],
                    "dedup_key": c["dedup_key"], "_source": "BPIQ", "_date_start": c["date_start"],
                    "_date_end": c["date_end"], "_match_text": c["drug_en"],
                })
            print(f"[BPIQ] 本轮最终 {len(items)} 条", flush=True)
    except Exception as e:
        print(f"[BPIQ] 异常: {e}", flush=True)
        return []
    finally:
        try:
            if browser:
                await browser.close()
        except Exception:
            pass
        kill_browser_processes()
    return items

# ==================== RTTNews FDA Calendar ====================
# 页面结构（实测，非 <table>）：div.dvCalendar 内每行是 div.grid-row，
# 单元格是带 data-th 属性的 div（Company Name / Drug / Event / Outcome / Details），
# 日期在 Event 单元格的 span.bg-purple 里；行间夹杂无 data-th 单元格的广告占位行，
# 解析时跳过。Details 单元格里取 "Drug Status" 链接。
RTTNEWS_DATE_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{4})")   # 11/25/2026
RTTNEWS_MONTH_RE = re.compile(r"^\s*([A-Za-z]{3,9})\s+(\d{4})")  # Nov 2026（仅到月）
RTTNEWS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36"
}

def parse_rttnews_event_date(text: str):
    """返回 (date, 是否仅到月)，解析失败返回 None。仅到月的用当月 1 日作起点。"""
    m = RTTNEWS_DATE_RE.match(text)
    if m:
        return datetime.strptime(m.group(1), "%m/%d/%Y").date(), False
    m = RTTNEWS_MONTH_RE.match(text)
    if m:
        for fmt in ("%b %Y", "%B %Y"):
            try:
                d = datetime.strptime(f"{m.group(1)} {m.group(2)}", fmt).date()
                return d.replace(day=1), True
            except ValueError:
                continue
    return None

def is_rttnews_pending(outcome_text: str) -> bool:
    # 待定项字面就是 "Pending-" / "pending-"（带尾部连字符、大小写不一）；
    # 已出结果的是完整句子。去掉尾部连字符后严格等于 pending 才算。
    return outcome_text.strip().rstrip("-").strip().lower() == "pending"

def extract_rttnews_ticker(company_text: str) -> str:
    # Company 单元格格式：公司名 + 括号内代码，如 "Sanofi SA ( SNY, SNYNF, SAN.PA)"，
    # 取括号内的第一个代码（主代码）。不设 4 字母限制，ADR（如 RHHBY）也会保留。
    groups = re.findall(r"\(([^)]*)\)", company_text)
    if not groups:
        return company_text.strip()
    codes = [c.strip() for c in groups[-1].split(",") if c.strip()]
    return codes[0] if codes else company_text.strip()

def extract_rttnews_title(event_cell) -> str:
    # Event 单元格全文 = 日期 + 描述，去掉开头的日期部分，剩下的是内容标题英文原文
    # （如 "FDA decision on Gazyva/Gazyvaro for the treatment of ..."），翻译在去重后做。
    full = event_cell.get_text(separator=" ", strip=True)
    for rx in (RTTNEWS_DATE_RE, RTTNEWS_MONTH_RE):
        m = rx.match(full)
        if m:
            return full[m.end():].strip()
    return full

def extract_rttnews_link(details_cell, base_url: str) -> str:
    links = [a for a in details_cell.find_all("a", href=True)
             if not a["href"].strip().lower().startswith("javascript:")]
    for a in links:
        if "drug status" in a.get_text(strip=True).lower():
            return urljoin(base_url, a["href"])
    if links:
        return urljoin(base_url, links[0]["href"])
    return ""

def format_rttnews_event_time(d: date, month_only: bool) -> str:
    # 与 BPIQ 同样的中文显示：精确日期 "2026年11月25日"，仅到月 "2026年11月"。
    if month_only:
        return f"{d.year}年{d.month}月"
    return f"{d.year}年{d.month}月{d.day}日"

def make_rttnews_dedup_key(ticker: str, title_en: str, start_date: date, end_date: date) -> str:
    return "|".join([
        ticker.upper().strip(), normalize_event_text(title_en),
        start_date.isoformat(), end_date.isoformat(),
    ])

# NYSE 休市日（2026–2027），用于算每月最后一个交易日；超出此范围的年份退化为最后一个工作日。
US_MARKET_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}

def last_trading_day_of_month(year: int, month: int) -> date:
    """该月最后一个交易日：从月末往前找，跳过周末与 NYSE 休市日。"""
    d = date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() >= 5 or d in US_MARKET_HOLIDAYS:
        d -= timedelta(days=1)
    return d

def parse_rttnews_page(html: str, page_url: str, window_start: date, window_end: date, today: date) -> List[Dict]:
    """解析一页 HTML，返回窗口内且 Pending 的原始条目（英文标题，未翻译）。
    仅到月的条目（如 Nov 2026）只有在 today 为该月最后一个交易日时才返回。"""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for row in soup.find_all("div", class_="grid-row"):
        cells = {}
        for cell in row.find_all("div", attrs={"data-th": True}):
            cells[cell["data-th"]] = cell
        # 广告占位行没有 data-th 数据单元格，直接跳过
        if not {"Company Name", "Drug", "Event", "Outcome"}.issubset(cells):
            continue
        outcome = cells["Outcome"].get_text(separator=" ", strip=True)
        if not is_rttnews_pending(outcome):
            continue
        event_cell = cells["Event"]
        date_span = event_cell.find("span", class_="bg-purple")
        parsed = parse_rttnews_event_date(date_span.get_text(strip=True)) if date_span else None
        if parsed is None:
            # 兜底：span 不存在或解析失败时，从单元格全文开头匹配日期
            parsed = parse_rttnews_event_date(event_cell.get_text(separator=" ", strip=True))
        if parsed is None:
            continue
        d, month_only = parsed
        if month_only:
            # 仅到月的条目：只有到了该月最后一个交易日才推，平时一律不推
            if today != last_trading_day_of_month(d.year, d.month):
                continue
            last_day = calendar.monthrange(d.year, d.month)[1]
            date_start, date_end = d, date(d.year, d.month, last_day)
        else:
            date_start = date_end = d
            if not (window_start <= d <= window_end):
                continue
        company = cells["Company Name"].get_text(separator=" ", strip=True)
        ticker = extract_rttnews_ticker(company)
        if not ticker:
            continue
        title_en = extract_rttnews_title(event_cell)
        link = ""
        if "Details" in cells:
            link = extract_rttnews_link(cells["Details"], page_url)
        rows.append({
            "ticker": ticker, "title_en": title_en,
            "event_time": format_rttnews_event_time(d, month_only),
            "link": link or page_url,
            "_source": "RTTNews", "_date_start": date_start, "_date_end": date_end,
            "_match_text": title_en,
        })
    return rows

def scrape_rttnews_catalysts() -> List[Dict]:
    """抓取 RTTNews FDA Calendar 各页，返回窗口内未推送过的条目（英文标题，翻译在跨源去重后做）。"""
    today = date.today()
    window_start = today - timedelta(days=RTTNEWS_DAYS_BEFORE)
    window_end = today + timedelta(days=RTTNEWS_DAYS_AFTER)
    print(f"[RTTNews] 时间窗口：{window_start} ~ {window_end}", flush=True)
    sent = read_set(RTTNEWS_SENT_DB_FILE)
    items = []
    for url in RTTNEWS_URLS:
        try:
            resp = requests.get(url, headers=RTTNEWS_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[RTTNews] 抓取失败 {url}: {e}", flush=True)
            continue
        for row in parse_rttnews_page(resp.text, url, window_start, window_end, today):
            dedup_key = make_rttnews_dedup_key(row["ticker"], row["title_en"], row["_date_start"], row["_date_end"])
            if dedup_key in sent:
                continue
            row["dedup_key"] = dedup_key
            items.append(row)
    print(f"[RTTNews] 本轮新条目 {len(items)} 条（未翻译、未跨源去重）", flush=True)
    return items

# ==================== 跨源事件级去重 ====================
def _item_date_range(item: Dict):
    """统一取条目的日期范围：BPIQ/RTTNews 用 _date_start/_date_end，StockTitan 用 _event_date。"""
    if item.get("_date_start") and item.get("_date_end"):
        return item["_date_start"], item["_date_end"]
    if item.get("_event_date"):
        return item["_event_date"], item["_event_date"]
    return None

_DEDUP_STOP_WORDS = {
    "phase", "trial", "study", "data", "result", "results", "readout", "clinical",
    "program", "treatment", "patients", "patient", "drug", "therapy", "and", "for",
    "with", "the", "of", "in", "to", "a", "an",
    "fda", "decision", "approval", "topline", "announce", "announced", "announces",
    "report", "reports", "reported", "update", "endpoint", "endpoints",
}

def _event_texts_match(text_a: str, text_b: str) -> bool:
    """两段事件英文文本是否指向同一事件：任一包含另一，或较短一方的 70% 关键词重合。"""
    a = normalize_event_text(text_a)
    b = normalize_event_text(text_b)
    if not a or not b:
        return False
    if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
        return True
    a_tokens = {t for t in a.split() if len(t) >= 3 and t not in _DEDUP_STOP_WORDS}
    b_tokens = {t for t in b.split() if len(t) >= 3 and t not in _DEDUP_STOP_WORDS}
    if not a_tokens or not b_tokens:
        return False
    matched = a_tokens & b_tokens
    return len(matched) / min(len(a_tokens), len(b_tokens)) >= 0.7

def is_cross_source_duplicate(candidate: Dict, existing_items: List[Dict]) -> bool:
    """同一 ticker + 日期范围重叠 + 事件文本相符，即判与已收集条目重复（不分来源）。"""
    ticker = candidate["ticker"].upper().strip()
    cand_range = _item_date_range(candidate)
    cand_text = candidate.get("_match_text", "")
    if not cand_range or not cand_text:
        return False
    for other in existing_items:
        if other.get("ticker", "").upper().strip() != ticker:
            continue
        other_range = _item_date_range(other)
        if not other_range:
            continue
        if cand_range[1] < other_range[0] or other_range[1] < cand_range[0]:
            continue
        if _event_texts_match(cand_text, other.get("_match_text", "")):
            return True
    return False

# ==================== 合并推送 ====================
def push_combined(items: List[Dict], bpiq_keys: List[str], rttnews_keys: Optional[List[str]] = None) -> bool:
    if not items:
        print("无符合条件的事件", flush=True)
        return False
    now_et = datetime.now(ZoneInfo("America/New_York"))
    header = f"🚨<b>{now_et.month}月{now_et.day}日医药股数据发布预警（共{len(items)}条）</b>\n\n"
    footer = "\n#ClinicalData"
    messages = []
    current_msg = header
    for i, item in enumerate(items, 1):
        item_str = f"{i}. 🚀股票代码: ${item['ticker']}\n"
        if item.get("event_time"):
            item_str += f"   ⏰公布时间: {item['event_time']}\n"
        item_str += f"   📰内容标题: {item['title']}\n   🔗<a href='{item['link']}'>点击查看公告</a>\n\n"
        if len(current_msg) + len(item_str) + len(footer) > 3900:
            messages.append(current_msg + footer)
            current_msg = "接上条续：\n\n" + item_str
        else:
            current_msg += item_str
    if current_msg.strip():
        messages.append(current_msg + footer)
    all_success = True
    for message in messages:
        if not send_telegram(message):
            all_success = False
    if not all_success:
        print("[Telegram] 至少一条消息发送失败，本轮不写入去重数据库", flush=True)
        return False
    if bpiq_keys:
        append_lines(BPIQ_SENT_DB_FILE, bpiq_keys)
    if rttnews_keys:
        append_lines(RTTNEWS_SENT_DB_FILE, rttnews_keys)
    print(f"成功推送 {len(items)} 条（其中 BPIQ {len(bpiq_keys)} 条，RTTNews {len(rttnews_keys or [])} 条）", flush=True)
    return True

# ==================== 主流程 ====================
def run_monitor():
    current_utc_ts = time.time()
    cutoff_ts = current_utc_ts - (HOURS_WINDOW * 3600)
    rss_event_start = date.today()
    rss_event_end = date.today() + timedelta(days=3)
    sent_urls = read_set(SENT_DB_FILE)
    collected_items = []
    new_urls = []
    rss_tickers = set()
    # ---- 1. StockTitan ----
    for rss_url in RSS_URLS:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if entry.link in sent_urls or entry.link in new_urls:
                continue
            pub_ts = calendar.timegm(entry.published_parsed) if hasattr(entry, "published_parsed") and entry.published_parsed else 0
            if pub_ts < cutoff_ts:
                continue
            title = entry.title
            title_lower = title.lower()
            if not (re.search(PATTERN_ACTION, title_lower) and re.search(PATTERN_SUBJECT, title_lower) and not re.search(PATTERN_EXCLUDE, title_lower)):
                continue
            ticker_match = re.search(r"\|\s*([A-Z]+)\s+Stock News", title)
            ticker = ticker_match.group(1) if ticker_match else "N/A"
            if ticker == "N/A" or len(ticker) > 4:
                print(f"[RSS] 跳过无效 ticker: {ticker}", flush=True)
                continue
            english_title = clean_title(title)
            chinese_title = translate_title(english_title)
            body_text = get_article_body(entry.link)
            event_time = analyze_event_time(english_title, body_text) if body_text else None
            event_date = parse_rss_event_date(event_time) if event_time else None
            if event_date is None:
                print(f"[RSS] 跳过（无明确近日期）: {ticker} | {event_time}", flush=True)
                continue
            if not (rss_event_start <= event_date <= rss_event_end):
                print(f"[RSS] 跳过（超出3天）: {ticker} | {event_time} -> {event_date}", flush=True)
                continue
            rss_tickers.add(ticker)
            collected_items.append({
                "ticker": ticker, "event_time": event_time, "title": chinese_title, "link": entry.link,
                "_source": "StockTitan", "_event_date": event_date,
                "_match_text": f"{english_title} {body_text or ''}",
            })
            new_urls.append(entry.link)
    print(f"[RSS] 本轮收集 {len(collected_items)} 条，涉及 ticker: {sorted(rss_tickers)}", flush=True)
    # ---- 2. BPIQ ----
    bpiq_keys = []
    if ENABLE_BPIQ:
        print("\n[BPIQ] 开始抓取...", flush=True)
        try:
            bpiq_items = asyncio.run(asyncio.wait_for(scrape_bpiq_catalysts(), timeout=120))
            cross_duplicates = 0
            for item in bpiq_items:
                if is_cross_source_duplicate(item, collected_items):
                    cross_duplicates += 1
                    print(f"[CROSS-DEDUP] 跳过重复事件: {item['ticker']} | {item['title']} | {item['event_time']}", flush=True)
                    continue
                collected_items.append(item)
                bpiq_keys.append(item["dedup_key"])
            print(f"[CROSS-DEDUP] BPIQ 跨源去重: {cross_duplicates} 条，保留 {len(bpiq_keys)} 条", flush=True)
        except asyncio.TimeoutError:
            print("[BPIQ] 超时，跳过", flush=True)
        except Exception as e:
            print(f"[BPIQ] 失败: {e}", flush=True)
        finally:
            kill_browser_processes()
    else:
        print("[BPIQ] 已关闭", flush=True)
    # ---- 3. RTTNews ----
    rttnews_keys = []
    if ENABLE_RTTNEWS:
        print("\n[RTTNews] 开始抓取...", flush=True)
        try:
            rttnews_items = scrape_rttnews_catalysts()
            cross_duplicates = 0
            for item in rttnews_items:
                if is_cross_source_duplicate(item, collected_items):
                    cross_duplicates += 1
                    print(f"[CROSS-DEDUP] 跳过重复事件: {item['ticker']} | {item['title_en']} | {item['event_time']}", flush=True)
                    continue
                # 跨源去重通过后再翻译，省掉重复条目的 Gemini 调用；RTT 标题后统一跟 PDUFA
                item["title"] = (translate_title(item["title_en"]) + " PDUFA").strip()
                collected_items.append(item)
                rttnews_keys.append(item["dedup_key"])
            print(f"[CROSS-DEDUP] RTTNews 跨源去重: {cross_duplicates} 条，保留 {len(rttnews_keys)} 条", flush=True)
        except Exception as e:
            print(f"[RTTNews] 失败: {e}", flush=True)
    else:
        print("[RTTNews] 已关闭", flush=True)
    # ---- 4. 合并推送（按事件日期升序，不分来源同一格式） ----
    if collected_items:
        collected_items.sort(key=lambda it: (_item_date_range(it) or (date.max, date.max))[0])
        success = push_combined(collected_items, bpiq_keys, rttnews_keys)
        if success:
            append_lines(SENT_DB_FILE, new_urls)
    else:
        print("未发现满足条件的新条目（RSS、BPIQ 与 RTTNews 均为空）", flush=True)
        print(f"RSS: new_urls={len(new_urls)}, tickers={rss_tickers}", flush=True)
        print(f"BPIQ keys: {len(bpiq_keys)}, RTTNews keys: {len(rttnews_keys)}", flush=True)

# ==================== 浏览器清理 ====================
def kill_browser_processes():
    try:
        subprocess.run(["pkill", "-f", "chrome"], capture_output=True)
        subprocess.run(["pkill", "-f", "chromium"], capture_output=True)
    except Exception:
        pass

# ==================== 程序入口 ====================
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
        time.sleep(2)
        os._exit(0)
