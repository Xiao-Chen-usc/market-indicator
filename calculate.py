"""
CC's Market Indicator
Score = 0.5 × z(BI) + 0.5 × z(CreditExcess)

Data sources (all free, FRED API):
  WILL5000PR  — Wilshire 5000 Total Market Cap (proxy for total market cap)
  GDP         — Nominal GDP, quarterly, billions
  GDPC1       — Real GDP, quarterly, billions chained 2017
  TCMDO       — Total Credit Market Debt Outstanding, quarterly, billions
"""

import os
import json
import datetime
import urllib.request
import numpy as np


FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"


# ── FRED fetch ──────────────────────────────────────────────────────────────

def fetch_fred(series_id, frequency="a"):
    """Fetch annual observations from FRED. Returns list of (year, value)."""
    url = (f"{FRED_BASE}?series_id={series_id}"
           f"&api_key={FRED_API_KEY}"
           f"&file_type=json"
           f"&frequency={frequency}"
           f"&observation_start=1970-01-01")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    out = []
    for obs in data["observations"]:
        if obs["value"] == ".":
            continue
        yr = int(obs["date"][:4])
        out.append((yr, float(obs["value"])))
    return out


def to_annual_dict(series_id, frequency="a"):
    rows = fetch_fred(series_id, frequency)
    # if quarterly, take last observation of each year
    d = {}
    for yr, val in rows:
        d[yr] = val          # later entries overwrite earlier ones → year-end
    return d


# ── Math helpers ─────────────────────────────────────────────────────────────

def pct_change(d, yr):
    """Year-over-year % change."""
    if yr not in d or (yr - 1) not in d:
        return None
    prev = d[yr - 1]
    if prev == 0:
        return None
    return (d[yr] - prev) / prev * 100


def rolling_z(series, idx, window=15):
    """
    z-score of series[idx] relative to the preceding `window` years.
    No look-ahead: uses only [idx-window, idx-1].
    """
    hist = [series[j] for j in range(max(0, idx - window), idx)
            if series[j] is not None]
    if len(hist) < 5:
        return None
    mu    = np.mean(hist)
    sigma = np.std(hist)
    if sigma == 0:
        return 0.0
    return (series[idx] - mu) / sigma


def roll_mean(lst, w):
    out = []
    for i in range(len(lst)):
        vals = [lst[j] for j in range(max(0, i - w + 1), i + 1)
                if lst[j] is not None]
        out.append(np.mean(vals) if len(vals) >= 2 else None)
    return out


# ── Main calculation ─────────────────────────────────────────────────────────

def calculate():
    print("Fetching FRED data …")

    # Wilshire 5000 total market cap (billions USD, annual)
    # FRED: WILL5000PR is price index; use NCBEILQ027S (market cap) if available,
    # otherwise we compute BI directly from DDDM01USA156NWDB (World Bank proxy)
    # Simplest reliable series: WILL5000IND (index) → approximate market cap
    # Best direct series: NCBEILQ027S (total equity market cap, quarterly, $bn)
    try:
        mcap_d = to_annual_dict("NCBEILQ027S", frequency="q")   # market cap $bn
    except Exception:
        # fallback: not available, use hardcoded recent values
        mcap_d = {}

    gdp_nom_d  = to_annual_dict("GDP",   frequency="q")   # nominal GDP $bn
    gdp_real_d = to_annual_dict("GDPC1", frequency="q")   # real GDP $bn chained
    tcmdo_d    = to_annual_dict("TCMDO", frequency="q")   # total credit $bn

    # Build year range
    start_yr = 1970
    end_yr   = datetime.date.today().year
    years    = list(range(start_yr, end_yr + 1))
    N        = len(years)

    # ── Buffett Indicator (BI) = market cap / nominal GDP × 100 ──
    bi = []
    for yr in years:
        if yr in mcap_d and yr in gdp_nom_d and gdp_nom_d[yr] > 0:
            bi.append(mcap_d[yr] / gdp_nom_d[yr] * 100)
        else:
            bi.append(None)

    # ── CreditExcess = TCMDO_growth - RealGDP_growth ──
    ce_raw = []
    for yr in years:
        tcmdo_g = pct_change(tcmdo_d,   yr)
        real_g  = pct_change(gdp_real_d, yr)
        if tcmdo_g is not None and real_g is not None:
            ce_raw.append(tcmdo_g - real_g)
        else:
            ce_raw.append(None)

    ce_3yr = roll_mean(ce_raw, 3)

    # ── 15-year rolling z-scores ──
    z_bi_list = []
    z_ce_list = []
    for i in range(N):
        z_bi_list.append(rolling_z(bi,     i, 15) if bi[i]     is not None else None)
        z_ce_list.append(rolling_z(ce_3yr, i, 15) if ce_3yr[i] is not None else None)

    # ── Score ──
    score_list = []
    for i in range(N):
        a, b = z_bi_list[i], z_ce_list[i]
        if a is not None and b is not None:
            score_list.append(0.5 * a + 0.5 * b)
        else:
            score_list.append(None)

    # ── Latest valid reading ──
    latest = {}
    for i in range(N - 1, -1, -1):
        if score_list[i] is not None:
            latest = {
                "year":        years[i],
                "bi":          round(bi[i],     1) if bi[i]     is not None else None,
                "ce_3yr":      round(ce_3yr[i], 2) if ce_3yr[i] is not None else None,
                "z_bi":        round(z_bi_list[i],  2),
                "z_ce":        round(z_ce_list[i],  2),
                "score":       round(score_list[i], 2),
                "threshold":   1.5,
                "signal":      score_list[i] > 1.5,
                "gap_to_signal": round(1.5 - score_list[i], 2),
            }
            break

    return latest


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = calculate()
    print(json.dumps(result, indent=2))
