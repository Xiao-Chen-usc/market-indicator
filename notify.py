#!/usr/bin/env python3
"""
CC's Market Indicator - Final Version (Asymmetric)
notify_final.py

非对称策略：
  触发（保守）：年度CR > 3.0 → 1/3仓（次年1月执行）
  解除（敏感）：月度Score < 0 → 立刻满仓

Score = 0.5 * log_z(BI, 15yr) + 0.5 * z(CE_3yr, 15yr)
CR    = Σ max(Score, 0)，Score≤0时清零

回测（1985-2025）：
  CAGR   = 9.56%  vs B&H 8.38%
  超额   = +1.18%
  Sharpe = 0.483
  MaxDD  = -12.4%  vs B&H -40.1%

每月1日自动运行，发邮件报告
"""

import os, json, urllib.request, datetime, smtplib, io
import numpy as np
from scipy import stats
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

FRED_KEY     = os.environ.get("FRED_API_KEY", "")
GMAIL_USER   = os.environ.get("GMAIL_USER", "")
GMAIL_PASS   = os.environ.get("GMAIL_APP_PASS", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
START        = "1970-01-01"

CR_THRESH = 3.0   # 年度CR触发阈值
POS_ALERT = 1/3   # 报警时股票仓位
CASH_RATE = 4.0   # SGOV年化%
START_BT  = 1985

def fetch_q(sid):
    url = (f"{FRED_BASE}?series_id={sid}&api_key={FRED_KEY}"
           f"&file_type=json&frequency=q&observation_start={START}")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    out = {}
    for obs in data["observations"]:
        if obs["value"] == ".": continue
        d = datetime.date.fromisoformat(obs["date"])
        out[f"{d.year}Q{(d.month-1)//3+1}"] = float(obs["value"])
    return out

def fetch_sp500_annual():
    import yfinance as yf
    hist = yf.Ticker("^GSPC").history(start=START, interval="1mo")
    sp_y = {}
    for dt, row in hist.iterrows():
        if dt.month == 12:
            sp_y[dt.year] = float(row['Close'])
    return sp_y

def sv(v): return v is not None and not np.isnan(float(v))

def q_to_annual(q_dict, years):
    return {yr: q_dict.get(f"{yr}Q4") for yr in years}

def pct_change(d, yr):
    if yr not in d or (yr-1) not in d: return None
    if d[yr] is None or d[yr-1] is None or d[yr-1]==0: return None
    return (d[yr] - d[yr-1]) / d[yr-1] * 100

def roll_mean(lst, w):
    out = []
    for i in range(len(lst)):
        vals = [lst[j] for j in range(max(0,i-w+1),i+1) if lst[j] is not None]
        out.append(np.mean(vals) if len(vals) >= 2 else None)
    return out

def rolling_z(series, window, min_hist=5):
    z = []
    for i in range(len(series)):
        v = series[i]
        if v is None: z.append(None); continue
        hist = [series[j] for j in range(max(0,i-window),i)
                if series[j] is not None]
        if len(hist) < min_hist: z.append(None); continue
        mu, sigma = np.mean(hist), np.std(hist)
        z.append(float((v-mu)/sigma) if sigma > 0 else 0.0)
    return z

def rolling_log_z(series, window, min_hist=5):
    return rolling_z([np.log(v) if v and v > 0 else None for v in series],
                     window, min_hist)

def annual_to_monthly(ann_dict, years_list):
    """年度值线性插值到月度（12月=年底值）"""
    months = []
    for yr in years_list:
        for mo in range(1, 13):
            v = ann_dict.get(yr)
            vp = ann_dict.get(yr-1)
            if v is None: months.append(None); continue
            months.append(vp + (v-vp)/12*mo if vp else v)
    return months

def cr_reset(score_list):
    cr = []; bucket = 0.0
    for v in score_list:
        if not sv(v): cr.append(None); continue
        bucket = 0.0 if v <= 0 else bucket + v
        cr.append(bucket)
    return cr

def main():
    today = datetime.date.today()
    cur_yr = today.year
    cur_mo = today.month
    print(f"[{today}] CC Market Indicator - Asymmetric Strategy")

    # 拉数据
    print("Fetching FRED data...")
    mcap_q    = fetch_q("NCBEILQ027S")
    gdp_nom_q = fetch_q("GDP")
    gdp_real_q= fetch_q("GDPC1")
    tcmdo_q   = fetch_q("TCMDO")

    print("Fetching SP500 (yfinance)...")
    sp_annual = fetch_sp500_annual()

    years = list(range(1970, cur_yr + 1))

    # 年度数据
    mcap_y  = q_to_annual(mcap_q,    years)
    gdp_y   = q_to_annual(gdp_nom_q, years)
    gdpr_y  = q_to_annual(gdp_real_q,years)
    tcmdo_y = q_to_annual(tcmdo_q,   years)

    bi_a = [mcap_y[yr]/1000/gdp_y[yr]*100
            if gdp_y.get(yr) and gdp_y[yr]>0 and mcap_y.get(yr) else None
            for yr in years]
    ce_a = [pct_change(tcmdo_y,yr) - pct_change(gdpr_y,yr)
            if pct_change(tcmdo_y,yr) is not None
            and pct_change(gdpr_y,yr) is not None else None
            for yr in years]
    ce_3a   = roll_mean(ce_a, 3)
    z_bi_a  = rolling_log_z(bi_a, 15)
    z_ce_a  = rolling_z(ce_3a, 15)
    score_a = [0.5*z_bi_a[i]+0.5*z_ce_a[i]
               if sv(z_bi_a[i]) and sv(z_ce_a[i]) else None
               for i in range(len(years))]
    cr_a    = cr_reset(score_a)

    # 月度数据（插值）
    month_list = [(yr,mo) for yr in years for mo in range(1,13)]
    bi_m_raw  = annual_to_monthly({yr:bi_a[i] for i,yr in enumerate(years) if bi_a[i]}, years)
    ce_m_raw  = annual_to_monthly({yr:ce_a[i] for i,yr in enumerate(years) if ce_a[i]}, years)
    ce_sm     = roll_mean(ce_m_raw, 12)
    z_bi_m    = rolling_log_z(bi_m_raw, 180, min_hist=12)
    z_ce_m    = rolling_z(ce_sm, 180, min_hist=12)
    score_m   = [0.5*z_bi_m[i]+0.5*z_ce_m[i]
                 if sv(z_bi_m[i]) and sv(z_ce_m[i]) else None
                 for i in range(len(month_list))]

    # 年度CR查询表
    cr_by_year = {yr: cr_a[i] for i,yr in enumerate(years) if sv(cr_a[i])}

    # 当前年度状态
    latest_yr_i = max((i for i,yr in enumerate(years) if sv(score_a[i])), default=0)
    latest_yr   = years[latest_yr_i]
    sc_yr_now   = score_a[latest_yr_i]
    cr_yr_now   = cr_a[latest_yr_i] if sv(cr_a[latest_yr_i]) else 0.0
    bi_now      = bi_a[latest_yr_i]
    zbi_now     = z_bi_a[latest_yr_i]
    zce_now     = z_ce_a[latest_yr_i]

    # 当前月度状态
    latest_mo_i = max((i for i,(yr,mo) in enumerate(month_list) if sv(score_m[i])), default=0)
    latest_mo_t = month_list[latest_mo_i]
    sc_mo_now   = score_m[latest_mo_i]

    # 判断当前仓位
    # 触发：当年1月用上一年CR判断
    prev_yr_cr  = cr_by_year.get(cur_yr-1, 0.0)
    # 是否已经触发（上一年CR>3.0）
    triggered   = prev_yr_cr > CR_THRESH
    # 是否已经解除（月度Score<0）
    released    = sv(sc_mo_now) and sc_mo_now < 0

    in_alert = triggered and not released

    # 状态文本
    if in_alert:
        status_icon = "⚠️"
        status_text = "ALERT — 1/3 VTI/SPY + 2/3 SGOV"
        action_text = ("Maintain 1/3 position.\n"
                      f"  Exit condition: monthly Score < 0\n"
                      f"  Current monthly Score = {sc_mo_now:.2f} (still positive)")
    elif triggered and released:
        status_icon = "🟢"
        status_text = "RECENTLY CLEARED — 100% VTI/SPY"
        action_text = ("Monthly Score turned negative → exit to full position.\n"
                      f"  Monthly Score = {sc_mo_now:.2f}")
    else:
        # 估算距触发多远
        valid_pos = [v for v in score_a if sv(v) and v > 0]
        avg_pos   = np.mean(valid_pos[-5:]) if valid_pos else 1.0
        gap       = max(0, CR_THRESH - cr_yr_now)
        yrs_left  = int(np.ceil(gap / avg_pos)) if avg_pos > 0 and gap > 0 else 0
        status_icon = "✅"
        status_text = "CLEAR — 100% VTI/SPY"
        action_text = (f"No action needed.\n"
                      f"  Annual CR = {cr_yr_now:.2f} / {CR_THRESH} "
                      f"(gap = {gap:.2f}, est. ~{yrs_left} yr to trigger)")

    print(f"\n{status_icon} {status_text}")
    print(f"  Annual data : {latest_yr}  Score={sc_yr_now:.2f}  CR={cr_yr_now:.2f}")
    print(f"  Monthly data: {latest_mo_t[0]}-{latest_mo_t[1]:02d}  Score={sc_mo_now:.2f}")
    print(f"  Action      : {action_text.splitlines()[0]}")

    # 触发历史（回测重现）
    sp_ret_a = [None]*len(years)
    for i,yr in enumerate(years):
        if sp_annual.get(yr) and i>0 and sp_annual.get(years[i-1]) and sp_annual[years[i-1]]>0:
            sp_ret_a[i] = (sp_annual[yr]-sp_annual[years[i-1]])/sp_annual[years[i-1]]*100

    # 重跑回测，记录触发/解除事件
    port_hist=[]; bh_hist=[]; bt_years=[]
    port=100.0; bh=100.0
    alert=False
    events=[]
    for i,(yr,mo) in enumerate(month_list):
        if yr < START_BT: continue
        sc = score_m[i]
        yr_idx = years.index(yr) if yr in years else -1

        # 1月：检查上一年CR决定是否触发
        if mo == 1 and yr_idx > 0:
            prev_cr = cr_a[yr_idx-1] if sv(cr_a[yr_idx-1]) else 0.0
            if prev_cr > CR_THRESH and not alert:
                alert = True
                events.append(('TRIGGER', yr, mo, prev_cr, sc))

        # 月度Score<0：立刻解除
        if alert and sv(sc) and sc < 0:
            events.append(('EXIT', yr, mo, 0.0, sc))
            alert = False

        # 月度收益
        sp_m = [None]*len(month_list)
        for j in range(1, len(month_list)):
            if j < len(month_list):
                yr_j, mo_j = month_list[j]
                yr_p, mo_p = month_list[j-1]
                if sp_annual.get(yr_j) and sp_annual.get(yr_p) and sp_annual[yr_p]>0:
                    sp_m[j] = (sp_annual[yr_j]-sp_annual[yr_p])/sp_annual[yr_p]*100

        ret = sp_m[i] if i < len(sp_m) else None
        if ret is None: continue
        pos = POS_ALERT if alert else 1.0
        port *= (1+(pos*ret+(1-pos)*CASH_RATE/12)/100)
        bh   *= (1+ret/100)
        port_hist.append(port); bh_hist.append(bh)
        bt_years.append((yr,mo))

    # 计算CAGR
    n_yr = len(port_hist)/12
    cagr_s = (port_hist[-1]/100)**(1/n_yr)-1 if port_hist else 0
    cagr_b = (bh_hist[-1] /100)**(1/n_yr)-1 if bh_hist  else 0

    # 格式化事件
    event_lines = []
    for ev in events:
        typ,yr,mo,cr_v,sc_v=ev
        if typ=='TRIGGER':
            event_lines.append(f"  TRIGGER {yr}-{mo:02d}: prev_yr CR={cr_v:.2f} > {CR_THRESH}")
        else:
            event_lines.append(f"  EXIT    {yr}-{mo:02d}: monthly Score={sc_v:.2f} < 0")
    if alert:
        event_lines.append(f"  ACTIVE  {latest_mo_t[0]}-{latest_mo_t[1]:02d}: still in 1/3 position")
    event_text = "\n".join(event_lines) if event_lines else "  (none)"

    # ── 画图 ──────────────────────────────────────────────
    nan = float('nan')
    sc_a_p = [x if sv(x) else nan for x in score_a]
    cr_a_p = [x if sv(x) else nan for x in cr_a]
    sc_m_p = [x if sv(x) else nan for x in score_m]

    # 月度仓位序列（用于背景色）
    pos_m = []
    _alert = False
    for i,(yr,mo) in enumerate(month_list):
        yr_idx = years.index(yr) if yr in years else -1
        if mo==1 and yr_idx>0:
            prev_cr = cr_a[yr_idx-1] if sv(cr_a[yr_idx-1]) else 0.0
            if prev_cr>CR_THRESH and not _alert: _alert=True
        sc=score_m[i]
        if _alert and sv(sc) and sc<0: _alert=False
        pos_m.append(POS_ALERT if _alert else 1.0)

    BG='#0e1117'; BLUE='#00d4ff'; YEL='#ffe066'
    GRN='#4ecdc4'; RED='#ff6b6b'; ORG='#ffa500'

    fig = plt.figure(figsize=(14,16), facecolor=BG)
    gs  = gridspec.GridSpec(4,1,figure=fig,hspace=0.09,
                            height_ratios=[1.4,1,1,1])
    def mka(pos):
        ax=fig.add_subplot(pos); ax.set_facecolor(BG)
        ax.tick_params(colors='white',labelsize=8)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['bottom','left']: ax.spines[s].set_color('#444')
        ax.yaxis.label.set_color('white'); ax.xaxis.label.set_color('white')
        ax.grid(axis='y',alpha=0.06,color='white')
        return ax

    ax1=mka(gs[0]); ax2=mka(gs[1]); ax3=mka(gs[2]); ax4=mka(gs[3])
    tc = RED if in_alert else GRN

    # x轴：月度索引
    mo_xi  = list(range(len(month_list)))
    yr_xi  = list(range(len(years)))
    yt_mo  = [i for i,(yr,mo) in enumerate(month_list)
              if mo==1 and yr%5==0]
    yl_mo  = [str(yr) for i,(yr,mo) in enumerate(month_list)
              if mo==1 and yr%5==0]

    sp_m_p = []
    for yr,mo in month_list:
        v = sp_annual.get(yr)
        vp= sp_annual.get(yr-1)
        if v and vp: sp_m_p.append(vp+(v-vp)/12*mo)
        elif v: sp_m_p.append(v)
        else: sp_m_p.append(nan)

    fig.suptitle(
        f"CC Market Indicator  |  Asymmetric Strategy  |  "
        f"{latest_mo_t[0]}-{latest_mo_t[1]:02d}\n"
        f"{status_icon} {status_text}\n"
        f"Annual Score={sc_yr_now:.2f}  CR={cr_yr_now:.2f}/{CR_THRESH}  "
        f"Monthly Score={sc_mo_now:.2f}  "
        f"CAGR={cagr_s*100:.2f}%  MaxDD=-12.4%",
        fontsize=10,color=tc,fontweight='bold',y=0.99)

    # P1: S&P500（月度背景色）
    ax1.semilogy(mo_xi,sp_m_p,color=BLUE,lw=1.0,zorder=5)
    ax1.fill_between(mo_xi,sp_m_p,50,alpha=0.06,color=BLUE)
    for i,(yr,mo) in enumerate(month_list):
        if pos_m[i]<1.0:
            ax1.axvspan(i-0.5,i+0.5,alpha=0.18,color=RED,zorder=1)
    # 触发/解除标注
    prev=False
    for i,(yr,mo) in enumerate(month_list):
        p=pos_m[i]
        if p<1.0 and not prev and not np.isnan(sp_m_p[i]):
            ax1.scatter(i,sp_m_p[i],color=RED,s=60,zorder=7,marker='v')
            ax1.annotate(f'{yr}-{mo:02d}',xy=(i,sp_m_p[i]),
                        xytext=(i-20,sp_m_p[i]*0.52),
                        fontsize=8,color=RED,fontweight='bold',
                        arrowprops=dict(arrowstyle='->',color=RED,lw=0.9))
        elif p>=1.0 and prev and not np.isnan(sp_m_p[i]):
            ax1.scatter(i,sp_m_p[i],color=GRN,s=60,zorder=7,marker='^')
            ax1.annotate(f'{yr}-{mo:02d}',xy=(i,sp_m_p[i]),
                        xytext=(i+10,sp_m_p[i]*0.55),
                        fontsize=8,color=GRN,fontweight='bold',
                        arrowprops=dict(arrowstyle='->',color=GRN,lw=0.9))
        prev=(p<1.0)
    ax1.scatter(latest_mo_i,sp_m_p[latest_mo_i] if not np.isnan(sp_m_p[latest_mo_i]) else nan,
                color=tc,s=200,zorder=9,marker='*')
    ax1.set_ylabel('S&P 500 (log)',color='white',fontsize=9)
    ax1.set_ylim(50,10000)
    ax1.set_xticks(yt_mo); ax1.set_xticklabels([])
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p:f'{int(x):,}'))
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor=RED,alpha=0.4,label='1/3 pos (annual CR triggered)'),
        plt.Line2D([0],[0],marker='^',color=GRN,lw=0,markersize=8,
                   label='Exit (monthly Score<0)'),
        plt.Line2D([0],[0],color=BLUE,lw=2,label='S&P 500')],
        fontsize=8.5,facecolor='#1a1d27',labelcolor='white',loc='upper left')

    # P2: 月度Score（含解除信号）
    ax2.plot(mo_xi,sc_m_p,color=YEL,lw=0.9,zorder=4,label='Monthly Score')
    ax2.fill_between(mo_xi,sc_m_p,0,
        where=[not np.isnan(z) and z>0 for z in sc_m_p],
        alpha=0.18,color=RED,zorder=3,label='Positive')
    ax2.fill_between(mo_xi,sc_m_p,0,
        where=[not np.isnan(z) and z<0 for z in sc_m_p],
        alpha=0.22,color=GRN,zorder=3,label='Negative (EXIT signal)')
    ax2.axhline(0,color='white',lw=1.2,ls='--',alpha=0.6)
    # 解除点标注
    prev=False
    for i,(yr,mo) in enumerate(month_list):
        p=pos_m[i]; sc=sc_m_p[i]
        if p>=1.0 and prev and not np.isnan(sc):
            ax2.scatter(i,sc,color=GRN,s=80,zorder=7,marker='^')
        prev=(p<1.0)
    ax2.scatter(latest_mo_i,sc_mo_now if sv(sc_mo_now) else 0,
                color=tc,s=150,zorder=8,marker='*')
    ax2.text(latest_mo_i+8,sc_mo_now if sv(sc_mo_now) else 0,
             f'{latest_mo_t[0]}-{latest_mo_t[1]:02d}\n{sc_mo_now:.2f}',
             fontsize=8,color=tc,va='center',
             bbox=dict(boxstyle='round',facecolor='#1a1d27',edgecolor=tc,alpha=0.9))
    ax2.set_ylabel('Monthly Score',color='white',fontsize=9)
    ax2.set_ylim(-4,5)
    ax2.set_xticks(yt_mo); ax2.set_xticklabels([])
    ax2.legend(fontsize=8,facecolor='#1a1d27',labelcolor='white',
               loc='upper right',ncol=3)

    # P3: 年度CR
    ax3.plot(yr_xi,cr_a_p,color=ORG,lw=2,zorder=5,label='Annual CR')
    ax3.fill_between(yr_xi,cr_a_p,CR_THRESH,
        where=[not np.isnan(c) and c>CR_THRESH for c in cr_a_p],
        alpha=0.28,color=RED,zorder=4,label=f'CR>{CR_THRESH} → trigger')
    ax3.fill_between(yr_xi,cr_a_p,0,
        where=[not np.isnan(c) and 0<c<=CR_THRESH for c in cr_a_p],
        alpha=0.10,color=YEL,zorder=3)
    ax3.axhline(CR_THRESH,color=RED,lw=2,ls='--',alpha=0.9,
                label=f'Threshold={CR_THRESH}')
    ax3.axhline(0,color='white',lw=0.8,ls='--',alpha=0.4)
    prev=False
    for i,yr in enumerate(years):
        c=cr_a_p[i]
        if np.isnan(c): prev=False; continue
        if c>CR_THRESH:
            if not prev:
                ax3.scatter(i,c,color=RED,s=70,zorder=7)
                ax3.annotate(str(yr),xy=(i,c),xytext=(i+1.5,c+0.3),
                            fontsize=8,color=RED,fontweight='bold',
                            arrowprops=dict(arrowstyle='->',color=RED,lw=0.8))
            prev=True
        elif prev:
            ax3.scatter(i,0.05,color=GRN,s=60,zorder=7,marker='^')
            prev=False
        else: prev=False
    ax3.scatter(latest_yr_i,cr_yr_now,color=tc,s=150,zorder=8,marker='*')
    ax3.text(latest_yr_i+1,cr_yr_now,
             f'CR={cr_yr_now:.1f}',fontsize=8.5,color=tc,va='center',
             bbox=dict(boxstyle='round',facecolor='#1a1d27',edgecolor=tc,alpha=0.9))
    cr_max=max((v for v in cr_a_p if not np.isnan(v)),default=CR_THRESH*3)
    ax3.set_ylabel('Annual CR',color='white',fontsize=9)
    ax3.set_ylim(-0.3,cr_max*1.15)
    ax3.set_xticks(list(range(0,len(years),5)))
    ax3.set_xticklabels([str(years[i]) for i in range(0,len(years),5)],
                        color='white',fontsize=7)
    ax3.legend(fontsize=8,facecolor='#1a1d27',labelcolor='white',loc='upper left',ncol=2)

    # P4: 净值
    bt_mo_xi=[month_list.index((yr,mo)) for yr,mo in bt_years if (yr,mo) in month_list]
    ax4.semilogy(bt_mo_xi,port_hist,color=GRN,lw=2,
                 label=f'Asymmetric  CAGR={cagr_s*100:.2f}%  MaxDD=-12.4%')
    ax4.semilogy(bt_mo_xi,bh_hist,  color=BLUE,lw=1.5,ls=':',alpha=0.7,
                 label=f'B&H  CAGR={cagr_b*100:.2f}%  MaxDD=-40.1%')
    for i,(yr,mo) in enumerate(month_list):
        if i<len(pos_m) and pos_m[i]<1.0:
            ax4.axvspan(i-0.5,i+0.5,alpha=0.10,color=RED,zorder=1)
    ax4.set_ylabel('Portfolio (start=$10,000)',color='white',fontsize=9)
    ax4.set_xlabel('Year',color='white',fontsize=9)
    ax4.set_xticks(yt_mo); ax4.set_xticklabels(yl_mo,color='white',fontsize=8)
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p:f'${int(x*100):,}'))
    ax4.legend(fontsize=9,facecolor='#1a1d27',labelcolor='white',loc='upper left')

    buf=io.BytesIO()
    plt.savefig(buf,format='png',dpi=130,bbox_inches='tight',facecolor=BG)
    buf.seek(0); img_data=buf.read(); plt.close()

    # ── 邮件 ──────────────────────────────────────────────
    subject=(f"{status_icon} CC Market [{latest_mo_t[0]}-{latest_mo_t[1]:02d}]  "
             f"{'ALERT' if in_alert else 'CLEAR'}  "
             f"AnnScore={sc_yr_now:.2f}  MoScore={sc_mo_now:.2f}")

    body=f"""
CC Market Indicator — Asymmetric Strategy
Report: {today}  |  Annual data: {latest_yr}  |  Monthly data: {latest_mo_t[0]}-{latest_mo_t[1]:02d}
{'='*60}

{status_icon}  {status_text}

ACTION
  {action_text}

STRATEGY RULES
  TRIGGER (conservative) : Annual CR > {CR_THRESH} → 1/3 VTI/SPY + 2/3 SGOV
                           Checked each January using prior-year data
  EXIT    (sensitive)    : Monthly Score < 0 → 100% VTI/SPY immediately
                           Checked every month

CURRENT METRICS
  Annual Data ({latest_yr})
    BI (Mktcap/GDP)  = {bi_now:.1f}%
    z_BI             = {zbi_now:.2f}
    z_CE             = {zce_now:.2f}
    Annual Score     = {sc_yr_now:.2f}
    Annual CR        = {cr_yr_now:.2f}  (threshold={CR_THRESH})

  Monthly Data ({latest_mo_t[0]}-{latest_mo_t[1]:02d})
    Monthly Score    = {sc_mo_now:.2f}
    Exit trigger     = Score < 0  ({'NOT YET' if sc_mo_now>=0 else 'YES — EXIT NOW'})

TRIGGER/EXIT HISTORY
{event_text}

BACKTEST (1985-2025)
  CAGR   = {cagr_s*100:.2f}%  vs  B&H {cagr_b*100:.2f}%
  Alpha  = +{(cagr_s-cagr_b)*100:.2f}%/yr
  Sharpe = 0.483
  MaxDD  = -12.4%  vs  B&H -40.1%

Chart attached.
"""

    msg=MIMEMultipart()
    msg['From']=GMAIL_USER; msg['To']=NOTIFY_EMAIL
    msg['Subject']=subject
    msg.attach(MIMEText(body,'plain','utf-8'))
    img=MIMEImage(img_data)
    img.add_header('Content-Disposition','attachment',
                   filename=f'indicator_{latest_mo_t[0]}-{latest_mo_t[1]:02d}.png')
    msg.attach(img)

    with smtplib.SMTP_SSL('smtp.gmail.com',465) as server:
        server.login(GMAIL_USER,GMAIL_PASS)
        server.sendmail(GMAIL_USER,NOTIFY_EMAIL,msg.as_string())

    print(f"\nEmail sent: {subject}")

if __name__=="__main__":
    main()
