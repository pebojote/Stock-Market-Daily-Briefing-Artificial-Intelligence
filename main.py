#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import os
import smtplib
import logging
import json
import html
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Iterable, Optional, Any, Dict, List

from openai import OpenAI
import yfinance as yf
import pandas as pd
from email.utils import formatdate, make_msgid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader

# ---------------------------
# 1) Configuration
# ---------------------------

PHT = ZoneInfo("Asia/Manila")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))

# Models for the dual-AI workflow
DATA_MODEL = "gpt-4o"
REASONING_MODEL = "gpt-5-turbo" # Placeholder, adjust if needed

RISK_LOWER = 5.5
RISK_UPPER = 7.0

WATCHLIST_UNIVERSE = ["MSFT", "NVDA", "ETN", "LLY", "NOC", "MA", "ANET", "CRWD"]
OPEN_POSITIONS = {
    "NVDA": 182.72,
    "MSFT": 530.26,
    "ANET": 139.85,
}

# ---------------------------
# 2) Logging
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("daily-briefing")

# ---------------------------
# 3) Validation
# ---------------------------

def _require_env(var_name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value

OPENAI_API_KEY = _require_env("OPENAI_API_KEY", OPENAI_API_KEY)
EMAIL_USER = _require_env("EMAIL_USER", EMAIL_USER)
GMAIL_APP_PASSWORD = _require_env("GMAIL_APP_PASSWORD", GMAIL_APP_PASSWORD)

# ---------------------------
# 4) Client
# ---------------------------

client = OpenAI(api_key=OPENAI_API_KEY, timeout=3600)

# ---------------------------
# 5) AI Workflow
# ---------------------------

def build_gpt4o_prompt(market_data: Dict[str, Any]) -> str:
    market_data_str = json.dumps(market_data, indent=2)
    return f"""
You are a data structuring AI. Your task is to take raw market data and structure it into a specific JSON format.
Calculate RSI, MACD, and identify trends and patterns.

Raw Market Data:
{market_data_str}

Return ONLY valid JSON, no prose, matching this schema exactly:
{{
  "date": "string",
  "market_overview": {{
    "sentiment": "string",
    "indexes": {{"sp500": "string", "nasdaq": "string"}},
    "news": ["string", "string"]
  }},
  "watchlist": [
    {{
      "ticker": "string",
      "rsi": "number (0-100)",
      "macd": "bullish|bearish|neutral",
      "ma_trend": "string",
      "pattern": "string",
      "entry": "number",
      "rank": "Excellent|Very Good|Good|Neutral|Bad|Very Bad|Worse",
      "action": "Buy|Hold|Watch|Sell",
      "notes": ""
    }}
  ],
  "open_positions": [
      {{
          "ticker": "string",
          "entry_price": "number",
          "current_price": "number"
      }}
  ],
  "journal": {{
    "did_right": ["string"],
    "improve": ["string"],
    "traps": ["string"]
  }},
  "opportunities": [
    {{
      "ticker": "string",
      "setup": "string",
      "entry_hint": "string"
    }}
  ],
  "reminders": ["string"]
}}
"""

def build_gpt5_prompt(structured_data: str) -> str:
    return f"""
You are a senior market analyst. Review the following structured market data.
For each item in the watchlist, provide a detailed reasoning in the 'notes' field.
The reasoning should explain WHY the stock is a good candidate (based on its technicals like RSI, MACD, pattern) and WHEN a potential entry might be considered (e.g., "on a breakout above $550" or "on a pullback to the 50-day MA around $520").

{structured_data}

Return the complete JSON object with the 'notes' fields populated. Do not add any other text.
"""

def call_openai_api(prompt: str, model: str) -> str:
    logger.info(f"Requesting briefing JSON with model={model}")
    # This is a placeholder for the actual API call.
    # In a real implementation, you would use the OpenAI client:
    # response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    # return response.choices[0].message.content

    # Mock responses for testing
    if model == DATA_MODEL:
        return json.dumps({
            "date": datetime.now(PHT).strftime("%A, %B %d, %Y"),
            "market_overview": {"sentiment": "Cautiously Optimistic", "indexes": {"sp500": "+0.2%", "nasdaq": "+0.5%"}, "news": ["Chip stocks rally.", "Inflation data comes in cooler than expected."]},
            "watchlist": [{"ticker": "MSFT", "rsi": 65, "macd": "bullish", "ma_trend": "uptrend", "pattern": "bull flag", "entry": 550, "rank": "Excellent", "action": "Buy", "notes": ""}],
            "open_positions": [{"ticker": "NVDA", "entry_price": 182.72, "current_price": 185.00}],
            "journal": {"did_right": ["Stuck to trading plan."], "improve": ["Avoided chasing FOMO trades."], "traps": ["Over-leveraging."]},
            "opportunities": [{"ticker": "ANET", "setup": "breakout", "entry_hint": "Entry on a close above $145"}],
            "reminders": ["Max risk per trade: 1.5%-2.0%", "Stay disciplined."]
        })
    elif model == REASONING_MODEL:
        start = prompt.find('{')
        end = prompt.rfind('}')
        if start != -1 and end != -1:
            json_part = prompt[start:end+1]
            data = json.loads(json_part)
            for item in data.get("watchlist", []):
                item["notes"] = f"Reasoning for {item['ticker']}: Strong bullish signals from MACD and MA trend. A buy signal would be a breakout above the recent high of ${item['entry']:.2f}."
            return json.dumps(data)
    return ""

def get_market_briefing_data() -> Dict[str, Any]:
    market_data = fetch_market_data()
    prompt_4o = build_gpt4o_prompt(market_data)
    structured_data_json = call_openai_api(prompt_4o, model=DATA_MODEL)
    prompt_5 = build_gpt5_prompt(structured_data_json)
    final_data_json = call_openai_api(prompt_5, model=REASONING_MODEL)
    return json.loads(final_data_json)

# ---------------------------
# 6) Data Fetching
# ---------------------------

def fetch_market_data() -> Dict[str, Any]:
    logger.info("Fetching market data...")
    all_tickers = set(WATCHLIST_UNIVERSE) | set(OPEN_POSITIONS.keys())
    ticker_list = list(all_tickers)
    if not ticker_list:
        return {}
    try:
        ticker_data = yf.download(ticker_list, period="1d", progress=False)
        if ticker_data.empty or 'Close' not in ticker_data.columns:
            logger.warning(f"yfinance returned no data for tickers: {ticker_list}")
            return {}
        last_prices = ticker_data['Close'].tail(1).to_dict('records')[0]
        ticker_data_dict = {t: {'price': p} for t, p in last_prices.items() if pd.notna(p)}
        logger.info(f"Successfully fetched prices for: {list(ticker_data_dict.keys())}")
        return ticker_data_dict
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data from yfinance: {e}", exc_info=True)
        return {}

# ---------------------------
# 7) Email Generation
# ---------------------------

def pct(a: float, b: float) -> float:
    return (b - a) / a * 100.0 if a != 0 else 0.0

def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def rank_color(rank: str) -> str:
    return {"Excellent": "#16a34a", "Very Good": "#22c55e", "Good": "#2dd4bf", "Neutral": "#6b7280", "Bad": "#f59e0b", "Very Bad": "#ef4444", "Worse": "#991b1b"}.get(rank, "#6b7280")

def action_color(action: str) -> str:
    return {"Buy": "#16a34a", "Hold": "#2563eb", "Watch": "#6b7280", "Sell": "#dc2626"}.get(action, "#6b7280")

def macd_color(macd: str) -> str:
    m = (macd or "").lower()
    return "#16a34a" if m == "bullish" else "#dc2626" if m == "bearish" else "#6b7280"

def rsi_color(rsi: float) -> str:
    try:
        r = float(rsi)
        if r >= 70: return "#dc2626"
        if r <= 30: return "#2563eb"
        return "#374151"
    except (ValueError, TypeError):
        return "#6b7280"

def risk_row_style(loss_pct: float) -> str:
    lp = abs(loss_pct)
    if lp >= RISK_UPPER: return "background:#fee2e2;color:#991b1b;"
    if lp >= RISK_LOWER: return "background:#fff7ed;color:#9a3412;"
    return ""

# Create a global Jinja2 environment
env = Environment(loader=FileSystemLoader('templates'), autoescape=True)

# Define and register custom filters
def format_percent(value):
    return f"{value:+.2f}%"
env.filters['format_percent'] = format_percent

# Register global functions
env.globals.update({
    'pct': pct,
    'safe_num': safe_num,
    'rank_color': rank_color,
    'action_color': action_color,
    'macd_color': macd_color,
    'rsi_color': rsi_color,
    'risk_row_style': risk_row_style
})

def build_html_email(data: Dict[str, Any], now_pht: datetime) -> str:
    template = env.get_template('email.html')

    template_data = data.copy()
    template_data['date'] = data.get("date") or now_pht.strftime("%A, %B %d, %Y")
    if 'verified_at' in data and data['verified_at']:
        template_data['verified_at'] = datetime.fromisoformat(data['verified_at'])

    # Sort watchlist by rank for the template
    template_data['watchlist'] = sorted(data.get('watchlist', []), key=lambda x: ["Worse", "Very Bad", "Bad", "Neutral", "Good", "Very Good", "Excellent"].index(x.get("rank", "Neutral")))

    return template.render(template_data)

def build_plain_text(data: Dict[str, Any], now_pht: datetime) -> str:
    # ... (simplified for brevity, but would be updated in a real scenario)
    return "Plain text version of the email."

# ---------------------------
# 8) Core Actions
# ---------------------------

def send_email(subject: str, html_body: str, plain_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"], msg["Date"] = subject, EMAIL_USER, EMAIL_TO, formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(EMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        raise

def daily_job() -> None:
    now_pht = datetime.now(PHT)
    logger.info("Starting daily job...")
    data = get_market_briefing_data()
    data['verified_at'] = now_pht.isoformat()
    html_report = build_html_email(data, now_pht)
    plain_report = build_plain_text(data, now_pht)
    subject = f"📊 Daily Market Briefing – {now_pht.strftime('%B %d, %Y at %I:%M %p')}"
    send_email(subject, html_report, plain_report)

# ---------------------------
# 9) Main Entry Point
# ---------------------------
if __name__ == "__main__":
    logger.info("Starting Cloud Run Job.")
    try:
        daily_job()
        logger.info("Cloud Run Job finished successfully.")
    except Exception as e:
        logger.error(f"Cloud Run Job failed: {e}", exc_info=True)
        raise
