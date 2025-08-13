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

def build_gpt4o_prompt(market_data: Dict[str, Any], now_pht: datetime) -> str:
    market_data_str = json.dumps(market_data, indent=2)
    session_time = now_pht.strftime("%H:%M")

    # Determine market session
    session = "Pre-Market"
    if "09:30" <= session_time < "12:00":
        session = "Morning Session"
    elif "12:00" <= session_time < "14:00":
        session = "Lunchtime"
    elif "14:00" <= session_time < "16:00":
        session = "Afternoon Session"
    elif "16:00" <= session_time < "17:00":
        session = "Closing Auction"
    elif "17:00" <= session_time:
        session = "After-hours Session"

    return f"""
You are a data structuring AI. Your task is to take raw market data and structure it into a specific JSON format.
The current time is {now_pht.strftime('%Y-%m-%d %H:%M:%S %Z')}. The market session is {session}.

Raw Market Data (prices and news):
{market_data_str}

Structure this data into the following JSON format. For each stock in the watchlist, calculate the technical indicators.
For open positions, use the provided prices.

Return ONLY valid JSON, no prose.

{{
  "session": "{session}",
  "market_overview": {{
    "sentiment": "string",
    "indexes": {{"sp500": "string", "nasdaq": "string"}},
    "economic_events": ["string"]
  }},
  "watchlist": [
    {{
      "ticker": "string",
      "news": ["string"],
      "support": "number",
      "resistance": "number",
      "rsi": "number",
      "macd": "string",
      "moving_averages": "string",
      "patterns": ["string"],
      "suggested_entry": "string",
      "rank": "Excellent|Good|Neutral|Bad|Very Bad",
      "action": "Buy|Hold|Watch|Sell"
    }}
  ],
  "open_positions": [
    {{
      "ticker": "string",
      "entry_price": "number",
      "current_price": "number",
      "outlook": "",
      "action_suggestion": "",
      "target_sell_zone": ""
    }}
  ],
  "trade_journal": {{
    "did_right": [""],
    "improve": [""],
    "traps": [""]
  }},
  "new_opportunities": [
    {{
      "ticker": "string",
      "setup": "High volume breakout|Oversold bounce|Trend continuation",
      "reason": ""
    }}
  ],
  "reminders": {{
    "max_risk_per_trade": "string",
    "stop_loss_discipline": "string",
    "emotional_check_in": "string"
  }}
}}
"""

def build_gpt5_prompt(structured_data: str) -> str:
    return f"""
You are a senior market analyst providing a detailed trading briefing.
Based on the structured data below, provide deep, insightful analysis for the empty fields.

- For `open_positions`, fill in `outlook`, `action_suggestion`, and `target_sell_zone`.
- For `trade_journal`, provide a detailed summary.
- For `new_opportunities`, provide a reason for each setup.
- For `reminders`, provide personalized advice.

{structured_data}

Return the complete JSON object with the analytical fields populated. Do not add any other text.
"""

def call_openai_api(prompt: str, model: str) -> str:
    logger.info(f"Requesting briefing JSON with model={model}")

    # This is a placeholder for the actual API call.
    if model == DATA_MODEL:
        # Parse the raw market data from the prompt
        raw_market_data_str = prompt.split("Raw Market Data (prices and news):\n")[1].split("\n\nStructure this data")[0]
        market_data = json.loads(raw_market_data_str)

        # In a real scenario, GPT-4o would generate this. For the mock, we build it.
        structured_data = {
            "session": "Morning Session",
            "market_overview": {
                "sentiment": "Mixed",
                "indexes": {"sp500": "+0.1%", "nasdaq": "-0.2%"},
                "economic_events": ["CPI data release at 8:30 AM EST.", "Fed chair speaks at 10:00 AM EST."]
            },
            "watchlist": [
                {
                    "ticker": ticker,
                    "news": market_data.get(ticker, {}).get('news', []),
                    "support": 100, "resistance": 120, "rsi": 55, "macd": "bullish",
                    "moving_averages": "Above 50-day MA", "patterns": ["cup and handle"],
                    "suggested_entry": f"Consider entry above 120",
                    "rank": "Good", "action": "Watch"
                } for ticker in WATCHLIST_UNIVERSE
            ],
            "open_positions": [
                {
                    "ticker": ticker, "entry_price": price, "current_price": market_data.get(ticker, {}).get('price', price),
                    "outlook": "", "action_suggestion": "", "target_sell_zone": ""
                } for ticker, price in OPEN_POSITIONS.items()
            ],
            "trade_journal": {"did_right": [""], "improve": [""], "traps": [""]},
            "new_opportunities": [{"ticker": "GOOGL", "setup": "Oversold bounce", "reason": ""}],
            "reminders": {"max_risk_per_trade": "", "stop_loss_discipline": "", "emotional_check_in": ""}
        }
        return json.dumps(structured_data)

    elif model == REASONING_MODEL:
        # GPT-5 would fill in the analytical fields.
        start = prompt.find('{')
        end = prompt.rfind('}')
        if start != -1 and end != -1:
            json_part = prompt[start:end+1]
            data = json.loads(json_part)
            # Add detailed reasoning
            for pos in data["open_positions"]:
                pos["outlook"] = f"Positive outlook for {pos['ticker']} due to market conditions."
                pos["action_suggestion"] = "Hold position, consider tightening stop-loss."
                pos["target_sell_zone"] = f"Between ${pos['current_price'] * 1.1:.2f} and ${pos['current_price'] * 1.2:.2f}"
            data["trade_journal"]["did_right"] = ["Followed trading plan for NVDA."]
            data["trade_journal"]["improve"] = ["Could have taken profits on MSFT sooner."]
            data["trade_journal"]["traps"] = ["Avoided FOMO on meme stocks."]
            data["new_opportunities"][0]["reason"] = "GOOGL is showing signs of bouncing from a key support level."
            data["reminders"]["max_risk_per_trade"] = "1.5% of portfolio"
            data["reminders"]["stop_loss_discipline"] = "On track"
            data["reminders"]["emotional_check_in"] = "Feeling calm and objective. Predicted mood: Focused."
            return json.dumps(data)

    return ""

def get_market_briefing_data(now_pht: datetime) -> Dict[str, Any]:
    market_data = fetch_market_data()
    prompt_4o = build_gpt4o_prompt(market_data, now_pht)
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

        for ticker in WATCHLIST_UNIVERSE:
            if ticker in ticker_data_dict:
                try:
                    news = yf.Ticker(ticker).news
                    if news:
                        ticker_data_dict[ticker]['news'] = [n['title'] for n in news[:3]] # Get top 3 news titles
                except Exception as e:
                    logger.warning(f"Could not fetch news for {ticker}: {e}")

        logger.info(f"Successfully fetched data for: {list(ticker_data_dict.keys())}")
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
    data = get_market_briefing_data(now_pht)
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
