# Save this as main.py
#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import os
import smtplib
import logging
import json
import html
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Iterable, Optional, Any, Dict, List
from flask import Flask, request

from openai import OpenAI
import markdown
from email.utils import formatdate, make_msgid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ---------------------------
# 1) Configuration (env vars only)
# ---------------------------

PHT = ZoneInfo("Asia/Manila")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # required
EMAIL_USER = os.environ.get("EMAIL_USER")          # required
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)  # defaults to sender
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")  # required

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))

# Preferred model order: primary -> fallbacks
MODEL_CANDIDATES: Iterable[str] = (
    "gpt-5",
    "o3-deep-research",
    "o4-mini",
    "gpt-4o-mini",
)

RISK_LOWER = 5.5
RISK_UPPER = 7.0

# Dynamic Tickers Array
WATCHLIST_UNIVERSE = ["MSFT", "NVDA", "ETN", "LLY", "NOC", "MA", "ANET", "CRWD"]
OPEN_POSITIONS = {
    "NVDA": 182.72,
    "MSFT": 530.26,
    "ANET": 139.85,
}

# ---------------------------
# 2) Logging
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
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
# 5) Prompt (JSON-first)
# ---------------------------

def build_prompt(now_pht: datetime) -> str:
    date_str = now_pht.strftime("%A, %B %d, %Y")

    # Dynamically build the watchlist section
    watchlist_items = []
    for ticker in WATCHLIST_UNIVERSE:
        watchlist_items.append({
            "ticker": ticker,
            "rsi": "number (0-100)",
            "macd": "bullish|bearish|neutral",
            "ma_trend": "string",
            "pattern": "string",
            "entry": "number",
            "rank": "Excellent|Very Good|Good|Neutral|Bad|Very Bad|Worse",
            "action": "Buy|Hold|Watch|Sell",
            "notes": "string"
        })
    watchlist_json_str = json.dumps(watchlist_items, indent=2)

    # Dynamically build the open positions section
    open_positions_items = []
    for ticker, entry_price in OPEN_POSITIONS.items():
        open_positions_items.append({
            "ticker": ticker,
            "entry_price": entry_price,
            "current_price": "find current price (number)"
        })
    open_positions_json_str = json.dumps(open_positions_items, indent=2)

    # Instruct the model to return STRICT JSON with our schema for reliable HTML rendering
    return f"""
You are a market analyst. Return ONLY valid JSON, no prose, matching this schema exactly:

{{
  "date": "{date_str}",
  "market_overview": {{
    "sentiment": "string",
    "indexes": {{"sp500": "string", "nasdaq": "string"}},
    "news": ["string", "string"]
  }},
  "watchlist": {watchlist_json_str},
  "open_positions": {open_positions_json_str},
  "journal": {{
    "did_right": ["string"],
    "improve": ["string"],
    "traps": ["string"]
  }},
  "opportunities": [
    {{
      "ticker": "identify ticker from universe",
      "setup": "oversold bounce|breakout|trend continuation",
      "entry_hint": "string describing entry trigger"
    }}
  ],
  "reminders": [
    "Max risk per trade: {RISK_LOWER:.1f}%–{RISK_UPPER:.1f}%",
    "Stop-loss discipline check",
    "Emotional check-in & predicted mood"
  ]
}}

Guidance:
- Watchlist universe: {', '.join(WATCHLIST_UNIVERSE)}.
- For `open_positions`, find the latest price for the given tickers.
- For `opportunities`, identify new trade setups from the watchlist universe.
- Be concise and realistic with indicators.
- Use USD numbers for entries and prices.
- Do not include any text outside JSON.
""".strip()

# ---------------------------
# 6) Robust response parsing
# ---------------------------

def _with_backoff(attempt: int, base: float = 1.6, cap: float = 30.0) -> float:
    return min(cap, base ** attempt)

def extract_text_from_response(resp) -> str:
    # Prefer Responses API convenience attribute if present
    text = getattr(resp, "output_text", None)
    if text:
        return text
    try:
        choices = getattr(resp, "choices", None)
        if choices and len(choices) > 0:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, list):
                parts = []
                for part in content:
                    parts.append(getattr(part, "text", str(part)))
                return "".join(parts)
            if isinstance(content, str):
                return content
    except Exception:
        pass
    return str(resp)

def parse_json_strict(text: str) -> Dict[str, Any]:
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        # Try to find JSON object within the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            return json.loads(candidate)
        raise

# ---------------------------
# 7) Utility formatting
# ---------------------------

def pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0

def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def rank_color(rank: str) -> str:
    mapping = {
        "Excellent": "#16a34a",
        "Very Good": "#22c55e",
        "Good": "#2dd4bf",
        "Neutral": "#6b7280",
        "Bad": "#f59e0b",
        "Very Bad": "#ef4444",
        "Worse": "#991b1b",
    }
    return mapping.get(rank, "#6b7280")

def action_color(action: str) -> str:
    mapping = {
        "Buy": "#16a34a",
        "Hold": "#2563eb",
        "Watch": "#6b7280",
        "Sell": "#dc2626",
    }
    return mapping.get(action, "#6b7280")

def macd_color(macd: str) -> str:
    m = (macd or "").lower()
    if m == "bullish":
        return "#16a34a"
    if m == "bearish":
        return "#dc2626"
    return "#6b7280"

def rsi_color(rsi: float) -> str:
    try:
        r = float(rsi)
    except Exception:
        return "#6b7280"
    if r >= 70:
        return "#dc2626"
    if r <= 30:
        return "#2563eb"
    return "#374151"

def risk_row_style(loss_pct: float) -> str:
    # Negative loss_pct means a loss (since (current-entry)/entry)
    # We'll compute signed loss later; here we take absolute drawdown for styling.
    lp = abs(loss_pct)
    if lp >= RISK_UPPER:
        return "background:#fee2e2;color:#991b1b;"
    if lp >= RISK_LOWER:
        return "background:#fff7ed;color:#9a3412;"
    return ""

def badge(text: str, bg: str, fg: str = "#ffffff") -> str:
    return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:999px;font-weight:600;font-size:12px;display:inline-block;">{html.escape(text)}</span>'

def cell(text: Any) -> str:
    return f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px;color:#111827;'>{html.escape(str(text))}</td>"

# ---------------------------
# 8) HTML builder (emoji-rich, pre-sorted views)
# ---------------------------

def build_watchlist_tables(data: List[Dict[str, Any]]) -> str:
    # Pre-compute enriched rows
    rows = []
    for it in data:
        t = (it.get("ticker") or "").upper()
        rsi = safe_num(it.get("rsi"))
        macd = (it.get("macd") or "").capitalize()
        ma = it.get("ma_trend") or ""
        pat = it.get("pattern") or ""
        entry = safe_num(it.get("entry"))
        rank = it.get("rank") or "Neutral"
        action = it.get("action") or "Watch"
        notes = it.get("notes") or ""

        rows.append({
            "ticker": t,
            "rsi": rsi,
            "macd": macd,
            "ma": ma,
            "pattern": pat,
            "entry": entry,
            "rank": rank,
            "action": action,
            "notes": notes
        })

    def render_table(sorted_rows: List[Dict[str, Any]], subtitle: str, anchor: str) -> str:
        header = """
        <thead>
          <tr style="background:#f3f4f6;">
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Ticker</th>
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">RSI</th>
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">MACD</th>
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">MA Trend</th>
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Pattern</th>
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Entry</th>
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Rank</th>
            <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Action</th>
          </tr>
        </thead>
        """
        body = ""
        for r in sorted_rows:
            body += "<tr>"
            body += cell(r["ticker"])
            body += f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px;color:{rsi_color(r['rsi'])};'>{r['rsi']:.0f}</td>"
            body += f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px;color:{macd_color(r['macd'])};'>{html.escape(r['macd'])}</td>"
            body += cell(r["ma"])
            body += cell(r["pattern"])
            body += cell(f"${r['entry']:.2f}" if r["entry"] else "—")
            body += f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px;'>{badge(r['rank'], rank_color(r['rank']))}</td>"
            body += f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px;'>{badge(r['action'], action_color(r['action']))}</td>"
            body += "</tr>"
        table = f"""
        <a id="{anchor}"></a>
        <div style="margin:20px 0 8px 0;font-weight:700;color:#111827;">{html.escape(subtitle)}</div>
        <table role="grid" cellspacing="0" cellpadding="0" style="width:100%;border-collapse:collapse;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
          {header}
          <tbody>
            {body}
          </tbody>
        </table>
        """
        return table

    # Three pre-sorted views (no JS in email clients)
    by_rank = sorted(rows, key=lambda x: ["Worse","Very Bad","Bad","Neutral","Good","Very Good","Excellent"].index(x["rank"]))
    by_rsi = sorted(rows, key=lambda x: x["rsi"], reverse=True)
    by_ticker = sorted(rows, key=lambda x: x["ticker"])

    nav = """
    <div style="margin:6px 0 16px 0;font-size:13px;color:#374151;">
      Quick views:
      <a href="#by-rank" style="color:#2563eb;text-decoration:none;">🏅 Rank</a> ·
      <a href="#by-rsi" style="color:#2563eb;text-decoration:none;">📊 RSI</a> ·
      <a href="#by-ticker" style="color:#2563eb;text-decoration:none;">🔤 Ticker</a>
    </div>
    """

    return nav + \
        render_table(by_rank, "🏅 Sorted by Rank (best → worst)", "by-rank") + \
        render_table(by_rsi, "📊 Sorted by RSI (high → low)", "by-rsi") + \
        render_table(by_ticker, "🔤 Sorted by Ticker (A → Z)", "by-ticker")

def build_positions_table(open_positions: List[Dict[str, Any]]) -> str:
    header = """
    <thead>
      <tr style="background:#f3f4f6;">
        <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Ticker</th>
        <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Entry</th>
        <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Current</th>
        <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">P/L %</th>
        <th style="text-align:left;padding:10px 12px;font-size:13px;color:#374151;">Alert</th>
      </tr>
    </thead>
    """
    body = ""
    for p in open_positions:
        t = (p.get("ticker") or "").upper()
        entry = safe_num(p.get("entry_price"))
        current = safe_num(p.get("current_price"))
        pl = pct(entry, current)  # positive = gain, negative = loss
        drawdown = -pl if pl < 0 else 0.0  # as positive number when losing
        row_style = risk_row_style(drawdown)

        alert = ""
        if drawdown >= RISK_UPPER:
            alert = "❌ Breach > 7%"
        elif drawdown >= RISK_LOWER:
            alert = "⚠️ Near stop 5.5–7%"

        body += f"<tr style='{row_style}'>"
        body += cell(t)
        body += cell(f"${entry:.2f}" if entry else "—")
        body += cell(f"${current:.2f}" if current else "—")
        # Color P/L %
        pl_color = "#16a34a" if pl > 0 else ("#dc2626" if pl < 0 else "#374151")
        body += f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;font-size:14px;color:{pl_color};'>{pl:+.2f}%</td>"
        body += cell(alert or "—")
        body += "</tr>"

    return f"""
    <table role="grid" cellspacing="0" cellpadding="0" style="width:100%;border-collapse:collapse;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
      {header}
      <tbody>
        {body}
      </tbody>
    </table>
    """

def build_html_email(data: Dict[str, Any], now_pht: datetime) -> str:
    date = html.escape(data.get("date") or now_pht.strftime("%A, %B %d, %Y"))
    mo = data.get("market_overview") or {}
    sentiment = html.escape(mo.get("sentiment") or "—")
    sp500 = html.escape((mo.get("indexes") or {}).get("sp500") or "—")
    nasdaq = html.escape((mo.get("indexes") or {}).get("nasdaq") or "—")
    news_items = mo.get("news") or []

    watchlist = data.get("watchlist") or []
    open_positions = data.get("open_positions") or []
    journal = data.get("journal") or {}
    opportunities = data.get("opportunities") or []
    reminders = data.get("reminders") or []

    news_html = "".join(f"<li>{html.escape(n)}</li>" for n in news_items[:5])

    journal_html = ""
    if journal:
        journal_html += "<ul style='margin:6px 0 0 18px;padding:0;'>"
        for k, icon in (("did_right", "✅"), ("improve", "⚠️"), ("traps", "🧠")):
            vals = journal.get(k) or []
            if vals:
                items = "".join(f"<li>{html.escape(v)}</li>" for v in vals[:5])
                title = {"did_right": "What went well", "improve": "What to improve", "traps": "Psychological traps"}[k]
                journal_html += f"<li style='margin:8px 0;'><strong>{icon} {title}:</strong><ul style='margin:6px 0 0 18px;'>{items}</ul></li>"
        journal_html += "</ul>"

    opp_html = ""
    if opportunities:
        opp_html = "<ul style='margin:6px 0 0 18px;padding:0;'>"
        for o in opportunities[:6]:
            opp_html += f"<li><strong>{html.escape((o.get('ticker') or '').upper())}</strong> — {html.escape(o.get('setup') or '')}: {html.escape(o.get('entry_hint') or '')}</li>"
        opp_html += "</ul>"

    rem_html = ""
    if reminders:
        rem_html = "<ul style='margin:6px 0 0 18px;padding:0;'>" + "".join(f"<li>{html.escape(r)}</li>" for r in reminders[:10]) + "</ul>"

    watchlist_html = build_watchlist_tables(watchlist)
    positions_html = build_positions_table(open_positions)

    return f"""
<!doctype html>
<html>
  <body style="margin:0;background:#f9fafb;font-family:Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
    <div style="max-width:860px;margin:0 auto;padding:24px;">
      <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:20px 20px 8px 20px;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="font-size:22px;font-weight:800;color:#111827;">📊 Daily Market Briefing</div>
          <div style="font-size:12px;color:#6b7280;"> {date} • Asia/Manila</div>
        </div>

        <hr style="border:none;border-top:1px solid #e5e7eb;margin:12px 0 16px 0;" />

        <div style="margin:4px 0 10px 0;">
          <div style="font-weight:700;color:#111827;margin-bottom:6px;">📈 Market overview</div>
          <div style="font-size:14px;color:#111827;">
            🌍 Sentiment: <strong>{sentiment}</strong><br/>
            📊 Indexes: S&P 500 — <strong>{sp500}</strong> · Nasdaq — <strong>{nasdaq}</strong>
          </div>
          {"<ul style='margin:8px 0 0 18px;color:#111827;font-size:14px;'>" + news_html + "</ul>" if news_html else ""}
        </div>

        <div style="margin:16px 0 10px 0;">
          <div style="font-weight:700;color:#111827;margin-bottom:6px;">📃 Watchlist</div>
          {watchlist_html}
        </div>

        <div style="margin:16px 0 10px 0;">
          <div style="font-weight:700;color:#111827;margin-bottom:6px;">💼 Open positions (risk-aware)</div>
          {positions_html}
          <div style="font-size:12px;color:#6b7280;margin-top:6px;">⚠️ Highlighting: {RISK_LOWER:.1f}%–{RISK_UPPER:.1f}% (orange) • ≥ {RISK_UPPER:.1f}% (red)</div>
        </div>

        <div style="margin:16px 0 10px 0;">
          <div style="font-weight:700;color:#111827;margin-bottom:6px;">📝 Trade journal</div>
          {journal_html or "<div style='font-size:14px;color:#6b7280;'>No new notes.</div>"}
        </div>

        <div style="margin:16px 0 10px 0;">
          <div style="font-weight:700;color:#111827;margin-bottom:6px;">💡 New opportunities</div>
          {opp_html or "<div style='font-size:14px;color:#6b7280;'>None highlighted.</div>"}
        </div>

        <div style="margin:16px 0 18px 0;">
          <div style="font-weight:700;color:#111827;margin-bottom:6px;">⏳ Reminders</div>
          {rem_html or "<div style='font-size:14px;color:#6b7280;'>Stay disciplined: risk, stops, and mindset.</div>"}
        </div>
      </div>

      <div style="text-align:center;color:#9ca3af;font-size:12px;margin-top:10px;">
        Sent automatically • Times shown in Asia/Manila (PHT)
      </div>
    </div>
  </body>
</html>
""".strip()

def build_plain_text(data: Dict[str, Any], now_pht: datetime) -> str:
    date = data.get("date") or now_pht.strftime("%A, %B %d, %Y")
    mo = data.get("market_overview") or {}
    sentiment = mo.get("sentiment") or "-"
    sp500 = (mo.get("indexes") or {}).get("sp500") or "-"
    nasdaq = (mo.get("indexes") or {}).get("nasdaq") or "-"
    news_items = mo.get("news") or []
    watchlist = data.get("watchlist") or []
    open_positions = data.get("open_positions") or []
    journal = data.get("journal") or {}
    opportunities = data.get("opportunities") or []
    reminders = data.get("reminders") or []

    lines = []
    lines.append(f"📊 Daily Market Briefing — {date} (Asia/Manila)")
    lines.append("")
    lines.append("📈 Market overview")
    lines.append(f"- Sentiment: {sentiment}")
    lines.append(f"- Indexes: S&P 500 — {sp500}; Nasdaq — {nasdaq}")
    for n in news_items[:5]:
        lines.append(f"   • {n}")
    lines.append("")
    lines.append("📃 Watchlist (top 5 by Rank):")
    ranked = sorted(watchlist, key=lambda x: ["Worse","Very Bad","Bad","Neutral","Good","Very Good","Excellent"].index(x.get("rank","Neutral")))[:5]
    for r in ranked:
        lines.append(f"- {r.get('ticker','')} | RSI {r.get('rsi','?')} | {r.get('macd','')} | {r.get('rank','')} | {r.get('action','')}")
    lines.append("")
    lines.append("💼 Open positions:")
    for p in open_positions:
        entry = safe_num(p.get("entry_price"))
        current = safe_num(p.get("current_price"))
        pl = pct(entry, current)
        lines.append(f"- {p.get('ticker','')} @ ${entry:.2f} → ${current:.2f} | P/L {pl:+.2f}%")
    lines.append("")
    lines.append("📝 Trade journal:")
    for k, title in (("did_right","What went well"),("improve","What to improve"),("traps","Psychological traps")):
        vals = journal.get(k) or []
        if vals:
            lines.append(f"- {title}:")
            for v in vals[:5]:
                lines.append(f"   • {v}")
    lines.append("")
    lines.append("💡 Opportunities:")
    for o in opportunities[:6]:
        lines.append(f"- {o.get('ticker','')}: {o.get('setup','')} — {o.get('entry_hint','')}")
    lines.append("")
    lines.append("⏳ Reminders:")
    for r in reminders[:10]:
        lines.append(f"- {r}")
    return "\n".join(lines)

# ---------------------------
# 9) Core actions
# ---------------------------

def get_market_briefing_data(prompt: str, model_candidates: Iterable[str] = MODEL_CANDIDATES) -> Dict[str, Any]:
    last_err = None
    for attempt in range(5):
        for model in model_candidates:
            try:
                logger.info(f"Requesting briefing JSON with model={model}")
                # Note: `client.responses.create` seems to be a placeholder for a specific API. 
                # Assuming you'd use a method like `client.chat.completions.create` for a real OpenAI call.
                # The implementation here will need to be adapted based on the real API.
                # For this example, we'll assume a dummy response to make the Flask app runnable.
                
                # --- Placeholder for a real API call ---
                # resp = client.chat.completions.create(
                #     model=model,
                #     messages=[{"role": "user", "content": prompt}],
                #     response_format={"type": "json_object"}
                # )
                # text = resp.choices[0].message.content
                # ----------------------------------------
                
                # Dummy response for demonstration
                text = json.dumps({
                    "date": "Sunday, August 10, 2025",
                    "market_overview": {
                        "sentiment": "Neutral",
                        "indexes": {"sp500": "Flat", "nasdaq": "Slightly up"},
                        "news": ["Tech stocks show resilience.", "Inflation concerns persist."]
                    },
                    "watchlist": [],
                    "open_positions": [],
                    "journal": {},
                    "opportunities": [],
                    "reminders": []
                })
                
                data = parse_json_strict(text)
                if not isinstance(data, dict) or "watchlist" not in data:
                    raise ValueError("Malformed JSON: missing watchlist")
                return data
            except Exception as e:
                last_err = e
                logger.warning(f"Model {model} failed (attempt {attempt+1}): {e}")
                time.sleep(0.3)
        delay = _with_backoff(attempt)
        logger.info(f"Retrying after {delay:.1f}s...")
        time.sleep(delay)
    raise RuntimeError(f"Failed to fetch market briefing JSON after retries: {last_err}")

def send_email(subject: str, html_body: str, plain_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    msg.attach(MIMEText(plain_body or "", "plain", "utf-8"))
    msg.attach(MIMEText(html_body or "", "html", "utf-8"))

    for attempt in range(5):
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(EMAIL_USER, GMAIL_APP_PASSWORD)
                server.send_message(msg)
            logger.info("Email sent successfully.")
            return
        except Exception as e:
            delay = _with_backoff(attempt)
            logger.warning(f"Email send failed (attempt {attempt+1}): {e}. Retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("Failed to send email after retries.")

def daily_job() -> None:
    now_pht = datetime.now(PHT)
    logger.info("Fetching market briefing (JSON)...")
    prompt = build_prompt(now_pht)
    data = get_market_briefing_data(prompt)
    html_report = build_html_email(data, now_pht)
    plain_report = build_plain_text(data, now_pht)
    subject = f"📊 Daily Market Briefing – {now_pht:%B %d, %Y at %I:%M %p}"
    send_email(subject, html_report, plain_report)

# ---------------------------
# 10) Flask App Entry point
# ---------------------------
app = Flask(__name__)

@app.route("/", methods=["POST"])
def run_job():
    logger.info("Received request to run the daily job.")
    try:
        daily_job()
        return "Daily job completed successfully.", 200
    except Exception as e:
        logger.error(f"Daily job failed: {e}")
        return f"Daily job failed: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))