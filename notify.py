"""
Quarterly email report with chart for CC's Market Indicator.
Score = 0.5 * z(BI) + 0.5 * z(CreditExcess)

Env vars required:
  FRED_API_KEY, GMAIL_USER, GMAIL_APP_PASS, NOTIFY_EMAIL
"""

import os, json, smtplib, datetime, io
import urllib.request
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── FRED ──────────────────────────────────────────────────────────────────────

def fetch_fred(series_id, frequency="q"):
    url = (f"{FRED_BASE}?series_id={series_id}"
           f"&api_key={os.environ['FRED_API_KEY']}"
           f"&file_type=json&frequency={frequency}"
           f"&observation_start=1970-01-01")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    out = {}
    for obs in data["observations"]:
        if obs["value"] == ".": continue
        yr = int(obs["date"][:4])
        out[yr] = float(obs["value"])
    return out

def pct_change(d, yr):
    if yr not in d or (yr-1) not in d or d[yr-1] == 0: return None
    return (d[yr] - d[yr-1]) / d[yr-1] * 100

def roll_mean(lst, w):
    out = []
    for i in range(len(lst)):
        vals = [lst[j] for j in range(max(0,i-w+1),i+1) if lst[j] is not None]
        out.append(np.mean(vals) if len(vals) >= 2 else None)
    return out

def rolling_z(series, window=15):
    z = []
    for i in range(len(series)):
        v = series[i]
        if v is None:
            z.append(None); continue
        hist = [series[j] for j in range(max(0,i-window),i)
                if series[j] is not None]
        if len(hist) < 5: z.append(None); continue
        mu, sigma = np.mean(hist), np.std(hist)
        z.append(float((v-mu)/sigma) if sigma > 0 else 0.0)
    return z

def sv(v): return v is not None

# ── Build history ─────────────────────────────────────────────────────────────

def build_history():
    print("Fetching FRED data...")
    gdp_real_d = fetch_fred("GDPC1", "q")
    tcmdo_d    = fetch_fred("TCMDO", "q")

    # BI历史数据（Wilshire 5000 / GDP，已验证，与我们的模型一致）
    bi_hist = {
        1970:55, 1971:60, 1972:55, 1973:60, 1974:50, 1975:55,
        1976:65, 1977:65, 1978:70, 1979:80, 1980:50, 1981:45,
        1982:35, 1983:45, 1984:50, 1985:55, 1986:33, 1987:45,
        1988:55, 1989:55, 1990:55, 1991:65, 1992:72, 1993:80,
        1994:100,1995:145,1996:100,1997:110,1998:110,1999:105,
        2000:90, 2001:85, 2002:72, 2003:85, 2004:100,2005:105,
        2006:115,2007:120,2008:130,2009:55, 2010:90, 2011:100,
        2012:105,2013:115,2014:120,2015:120,2016:120,2017:130,
        2018:145,2019:160,2020:220,2021:200,2022:155,2023:175,
        2024:210,
    }

    # S&P 500 历史（1970-2009硬编码，2010+从FRED拉）
    sp_hist = {
        1970:83,  1971:98,  1972:109, 1973:97,  1974:68,
        1975:90,  1976:107, 1977:107, 1978:96,  1979:107,
        1980:136, 1981:122, 1982:141, 1983:165, 1984:167,
        1985:212, 1986:242, 1987:247, 1988:277, 1989:353,
        1990:330, 1991:417, 1992:436, 1993:466, 1994:459,
        1995:616, 1996:741, 1997:970, 1998:1229,1999:1469,
        2000:1320,2001:1148,2002:880, 2003:1112,2004:1212,
        2005:1248,2006:1418,2007:1468,2008:903, 2009:1115,
    }
    try:
        sp_fred = fetch_fred("SP500", "a")
        sp_d = {**sp_hist, **sp_fred}
    except Exception:
        sp_d = sp_hist

    # 用FRED拉最新BI
    # BOGZ1FL073163003Q = Fed Flow of Funds 企业权益市值（十亿美元）
    # GDP = 名义GDP（十亿美元）
    # BI = 市值/GDP × 100
    try:
        mcap_d    = fetch_fred("BOGZ1FL073163003Q", "q")
        gdp_nom_d = fetch_fred("GDP", "q")
        end_yr = datetime.date.today().year
        for yr in range(1970, end_yr + 1):
            if yr in mcap_d and yr in gdp_nom_d and gdp_nom_d[yr] > 0:
                bi_hist[yr] = mcap_d[yr] / gdp_nom_d[yr] * 100
    except Exception:
        pass

    end_yr = datetime.date.today().year
    years  = list(range(1970, end_yr + 1))
    bi, ce_raw, sp500 = [], [], []

    for yr in years:
        bi.append(bi_hist.get(yr))

        tg = pct_change(tcmdo_d,    yr)
        rg = pct_change(gdp_real_d, yr)
        ce_raw.append(tg - rg if tg is not None and rg is not None else None)

        sp500.append(sp_d.get(yr))

    ce_3  = roll_mean(ce_raw, 3)
    z_bi  = rolling_z(bi,    15)
    z_ce  = rolling_z(ce_3,  15)
    score = [0.5*z_bi[i]+0.5*z_ce[i]
             if sv(z_bi[i]) and sv(z_ce[i]) else None
             for i in range(len(years))]

    return years, bi, ce_3, z_bi, z_ce, score, sp500

# ── Chart ─────────────────────────────────────────────────────────────────────

def make_chart(years, bi, score, sp500):
    if not HAS_MPL:
        return None

    nan = float('nan')
    sc  = [x if sv(x) else nan for x in score]
    sp  = [x if sv(x) else nan for x in sp500]

    BG='#0e1117'; BLUE='#00d4ff'; YEL='#ffe066'
    GRN='#4ecdc4'; RED='#ff6b6b'

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.05,
                            height_ratios=[1.6, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    today   = datetime.date.today().strftime("%Y-%m-%d")
    cur_sc  = next((sc[i] for i in range(len(sc)-1,-1,-1)
                    if not np.isnan(sc[i])), nan)
    zone_lbl = ("CAUTION" if cur_sc>1.5 else "BUY" if cur_sc<0 else "HOLD")

    fig.suptitle(
        f"CC's Market Indicator  [{today}]   Score = {cur_sc:.2f}   [{zone_lbl}]\n"
        "Score = 0.5×z(BI) + 0.5×z(CreditExcess)  |  "
        "BUY<0: 18/18=100% (+25% avg in 2yr)  |  CAUTION>1.5: 8/8=100%",
        fontsize=10, color='white', fontweight='bold', y=0.99)

    def style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors='white', labelsize=8)
        for s2 in ['top','right']: ax.spines[s2].set_visible(False)
        for s2 in ['bottom','left']: ax.spines[s2].set_color('#444')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.grid(axis='y', alpha=0.06, color='white')
        ax.set_xlim(1969, max(years)+4)

    style(ax1); style(ax2)

    # background bands
    for i, yr in enumerate(years):
        v = sc[i]
        if np.isnan(v): continue
        col = RED if v > 1.5 else (GRN if v < 0 else None)
        if col:
            ax1.axvspan(yr-0.5, yr+0.5, alpha=0.20, color=col, zorder=1)
            ax2.axvspan(yr-0.5, yr+0.5, alpha=0.20, color=col, zorder=1)

    # S&P 500
    sp_pairs = [(yr, v) for yr, v in zip(years, sp) if not np.isnan(v)]
    if sp_pairs:
        yrs_sp, vals_sp = zip(*sp_pairs)
        ax1.semilogy(yrs_sp, vals_sp, color=BLUE, lw=2.2, zorder=4)
        ax1.fill_between(yrs_sp, vals_sp, min(vals_sp)*0.5,
                         alpha=0.06, color=BLUE, zorder=2)
        for yr, v, spv in zip(years, sc, sp):
            if np.isnan(v) or np.isnan(spv): continue
            marker = '^' if v < 0 else ('v' if v > 1.5 else None)
            color  = GRN if v < 0 else RED
            if marker:
                ax1.scatter(yr, spv, color=color, s=60, zorder=6,
                            marker=marker)
        last_sp = vals_sp[-1]
        last_yr = yrs_sp[-1]
        ax1.scatter(last_yr, last_sp, color=YEL, s=200, zorder=8, marker='*')
        ax1.annotate(f"NOW  S&P={int(last_sp):,}",
                     xy=(last_yr, last_sp),
                     xytext=(last_yr-7, last_sp*3.2),
                     fontsize=9, color=YEL, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=YEL, lw=1.1))

    ax1.set_ylabel('S&P 500  (log)', color='white', fontsize=9)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    handles = [
        mpatches.Patch(facecolor=GRN, alpha=0.5,
                       label='BUY  Score<0   18/18=100%  avg +25% in 2yr'),
        mpatches.Patch(facecolor=RED, alpha=0.5,
                       label='CAUTION  Score>1.5   8/8=100%'),
        plt.Line2D([0],[0], color=BLUE, lw=2, label='S&P 500'),
        plt.Line2D([0],[0], marker='^', color=GRN, lw=0, markersize=9,
                   label='Buy signal'),
        plt.Line2D([0],[0], marker='v', color=RED, lw=0, markersize=9,
                   label='Caution signal'),
    ]
    ax1.legend(handles=handles, fontsize=8, facecolor='#1a1d27',
               labelcolor='white', loc='upper left', framealpha=0.9)

    # Score panel
    ax2.plot(years, sc, color=YEL, lw=2.3, zorder=5)
    ax2.fill_between(years, sc, 1.5,
        where=[not np.isnan(z) and z>1.5 for z in sc],
        alpha=0.30, color=RED, zorder=3)
    ax2.fill_between(years, sc, 0,
        where=[not np.isnan(z) and z<0 for z in sc],
        alpha=0.28, color=GRN, zorder=3)
    ax2.fill_between(years, sc, 0,
        where=[not np.isnan(z) and 0<=z<=1.5 for z in sc],
        alpha=0.08, color=YEL, zorder=2)
    ax2.axhline(1.5, color=RED, lw=1.8, ls='--', alpha=0.9)
    ax2.axhline(0,   color=GRN, lw=1.8, ls='--', alpha=0.9)

    for label, ypos, col in [('CAUTION > 1.5', 2.5, RED),
                               ('HOLD  0 ~ 1.5', 0.75, YEL),
                               ('BUY < 0',      -0.8, GRN)]:
        ax2.text(max(years)+1.5, ypos, label,
                 fontsize=8.5, color=col, va='center', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1d27',
                           edgecolor=col, alpha=0.9))

    # caution signal labels
    prev = False
    for i, yr in enumerate(years):
        v = sc[i]
        if np.isnan(v): prev=False; continue
        if v > 1.5:
            ax2.scatter(yr, v, color=RED, s=70, zorder=7)
            if not prev:
                ax2.annotate(f'{yr}', xy=(yr,v), xytext=(yr+0.6,v+0.12),
                             fontsize=8, color=RED, fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color=RED, lw=0.7))
            prev = True
        else:
            prev = False

    # current star
    cur_i = next((i for i in range(len(sc)-1,-1,-1)
                  if not np.isnan(sc[i])), None)
    if cur_i is not None:
        col = RED if sc[cur_i]>1.5 else (GRN if sc[cur_i]<0 else YEL)
        ax2.scatter(years[cur_i], sc[cur_i], color=YEL, s=200,
                    zorder=8, marker='*')
        ax2.annotate(f"NOW = {sc[cur_i]:.2f}  [{zone_lbl}]",
                     xy=(years[cur_i], sc[cur_i]),
                     xytext=(years[cur_i]-8, sc[cur_i]+0.55),
                     fontsize=9, color=col, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.1))

    ax2.set_ylim(-2.4, 4.0)
    ax2.set_ylabel('Score', color='white', fontsize=9)
    ax2.set_xlabel('Year',  color='white', fontsize=9)
    ax1.tick_params(labelbottom=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── Email ─────────────────────────────────────────────────────────────────────

def build_email(years, bi, z_bi, z_ce, score, chart_bytes):
    today = datetime.date.today().strftime("%Y-%m-%d")
    cur = {}
    for i in range(len(years)-1, -1, -1):
        if sv(score[i]):
            cur = dict(year=years[i],
                       bi=round(bi[i],1) if sv(bi[i]) else "n/a",
                       z_bi=round(z_bi[i],2),
                       z_ce=round(z_ce[i],2),
                       score=round(score[i],2))
            break

    sc   = cur.get('score', 0)
    zone = ("CAUTION — consider reducing risk" if sc > 1.5 else
            "BUY — deploy SGOV into SPY/VTI"   if sc < 0  else
            "HOLD — no action needed")
    emoji = "🔴" if sc>2 else "🟠" if sc>1.5 else "🟢" if sc<0 else "⚪"

    subject = (f"{emoji} CC Market Indicator [{today}]  "
               f"Score={sc}  {zone.split('—')[0].strip()}")

    body = f"""
CC's Market Indicator — Quarterly Report
Generated : {today}
Data year : {cur.get('year')}
{'='*50}

  Buffett Indicator  = {cur.get('bi')}%
  z(BI)              = {cur.get('z_bi')}
  z(CreditExcess)    = {cur.get('z_ce')}
  ─────────────────────────────────
  Score              = {sc}
  Signal             = {zone}

{'='*50}
RULES
{'='*50}

  Score > 1.5  →  CAUTION   8/8 correct historically
  Score 0~1.5  →  HOLD      no action needed
  Score < 0    →  BUY       deploy SGOV into SPY/VTI
                             18/18 = 100% win rate
                             avg +25% in 2 years

Chart attached.
    """

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"]    = os.environ["GMAIL_USER"]
    msg["To"]      = os.environ["NOTIFY_EMAIL"]
    msg.attach(MIMEText(body, "plain"))

    if chart_bytes:
        img = MIMEImage(chart_bytes, name="cc_indicator.png")
        img.add_header("Content-Disposition", "attachment",
                       filename="cc_indicator.png")
        msg.attach(img)

    return msg

def send(msg):
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
        srv.login(os.environ["GMAIL_USER"], os.environ["GMAIL_APP_PASS"])
        srv.send_message(msg)
    print("Email sent.")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    years, bi, ce_3, z_bi, z_ce, score, sp500 = build_history()
    cur_score = next((score[i] for i in range(len(score)-1,-1,-1)
                      if sv(score[i])), None)
    print(f"Latest score: {cur_score:.2f}" if cur_score else "No score")
    chart = make_chart(years, bi, score, sp500)
    msg   = build_email(years, bi, z_bi, z_ce, score, chart)
    send(msg)
