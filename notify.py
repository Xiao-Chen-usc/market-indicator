"""
Send monthly/quarterly email report for CC's Market Indicator.

Required environment variables (set as GitHub Actions secrets):
  FRED_API_KEY   — from https://fred.stlouisfed.org/docs/api/api_key.html
  GMAIL_USER     — your Gmail address  e.g. yourname@gmail.com
  GMAIL_APP_PASS — Gmail App Password  (NOT your regular password)
                   Generate at: myaccount.google.com → Security → App passwords
  NOTIFY_EMAIL   — recipient address (can be same as GMAIL_USER)
"""

import os
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from calculate import calculate


def signal_emoji(score, threshold=1.5):
    if score > threshold + 1:   return "🔴"   # strong warning
    if score > threshold:       return "🟠"   # triggered
    if score > threshold - 0.5: return "🟡"   # approaching
    if score < -threshold:      return "🟢"   # historically cheap
    return "⚪"                                # neutral


def build_email(data):
    today  = datetime.date.today().strftime("%Y-%m-%d")
    yr     = data["year"]
    score  = data["score"]
    bi     = data["bi"]
    ce     = data["ce_3yr"]
    z_bi   = data["z_bi"]
    z_ce   = data["z_ce"]
    sig    = "⚠️  SIGNAL TRIGGERED" if data["signal"] else "✅  No signal"
    gap    = data["gap_to_signal"]
    emoji  = signal_emoji(score)

    subject = f"{emoji} CC Market Indicator [{today}]  Score={score}  BI={bi}%"

    body = f"""
CC's Market Indicator — Monthly Report
Generated: {today}  |  Data year: {yr}
{"="*52}

  Score = 0.5 × z(BI) + 0.5 × z(CreditExcess)

  Buffett Indicator (BI)    = {bi}%
  CreditExcess 3yr avg      = {ce}%/yr
  z(BI)                     = {z_bi}
  z(CreditExcess)           = {z_ce}
  ─────────────────────────────────────
  Score                     = {score}
  Threshold                 = 1.5
  Status                    = {sig}
  Gap to signal             = {gap:+.2f}

{"="*52}
INTERPRETATION
{"="*52}

  z(BI)  measures how extreme current valuation is
         relative to the past 15 years.

  z(CE)  measures how extreme credit expansion is
         (TCMDO growth minus Real GDP growth, 3yr avg)
         relative to the past 15 years.

  Score > 1.5  →  bubble warning  (8/8 correct historically)
  Score < -1.5 →  historically cheap

{"="*52}
HISTORICAL SIGNALS (all correct)
{"="*52}

  1976: 2.06  →  Volcker tightening ✅
  1979: 2.49  →  Oil shock / recession ✅
  1995: 2.18  →  Dot-com buildup ✅
  2005: 1.76  →  GFC buildup ✅
  2006: 2.21  →  GFC buildup ✅
  2007: 2.04  →  GFC buildup ✅
  2008: 1.63  →  GFC ✅
  2020: 2.25  →  COVID bubble ✅

{"="*52}
Data sources: FRED (NCBEILQ027S, GDP, GDPC1, TCMDO)
Formula: github.com/[your-repo]/market-indicator
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = os.environ["GMAIL_USER"]
    msg["To"]      = os.environ["NOTIFY_EMAIL"]
    msg.attach(MIMEText(body, "plain"))
    return msg


def send(msg):
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ["GMAIL_USER"],
                     os.environ["GMAIL_APP_PASS"])
        server.send_message(msg)
    print("Email sent.")


if __name__ == "__main__":
    data = calculate()
    print("Latest reading:", data)
    msg  = build_email(data)
    send(msg)
