#!/usr/bin/env python3
"""
CC's Market Indicator - Monthly CR_reset Strategy
notify_monthly_v2.py

月度CR_reset策略：
  Score = 0.5 * log_z(BI, 180月) + 0.5 * z(CE_12月, 180月)
  CR = Σ max(Score, 0)，Score≤0时清零
  触发：CR > 阈值 → 1/3仓
  解除：Score≤0（CR清零）→ 满仓

比Buffer策略更敏感：不积累保护期，Score变负立刻清零重来

GitHub Actions每月1日自动运行
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

FRED_KEY     = os.environ.get("FRED_API_KEY","")
GMAIL_USER   = os.environ.get("GMAIL_USER","")
GMAIL_PASS   = os.environ.get("GMAIL_APP_PASS","")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL","")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
START        = "1970-01-01"

# 策略参数
CE_W      = 12     # CE平滑窗口（月）
Z_W       = 180    # z-score窗口（月=15年）
CR_THRESH = 9.0    # CR触发阈值（月度版，年度版3.0×3月=9.0）
POS_A     = 1/3    # 报警时仓位
CASH_R    = 1.0    # 现金年化收益%
START_BT  = "1985-01"

def fetch_q(sid):
    url=(f"{FRED_BASE}?series_id={sid}&api_key={FRED_KEY}"
         f"&file_type=json&frequency=q&observation_start={START}")
    with urllib.request.urlopen(url,timeout=30) as r:
        data=json.loads(r.read())
    out={}
    for obs in data["observations"]:
        if obs["value"]==".": continue
        d=datetime.date.fromisoformat(obs["date"])
        out[f"{d.year}Q{(d.month-1)//3+1}"]=float(obs["value"])
    return out

def fetch_sp500_monthly():
    import yfinance as yf
    hist=yf.Ticker("^GSPC").history(start=START,interval="1mo")
    out={}
    for dt,row in hist.iterrows():
        out[f"{dt.year}-{dt.month:02d}"]=float(row['Close'])
    return out

def gen_months(sy=1970,ey=None):
    if ey is None:
        t=datetime.date.today(); ey=t.year+1
    return [f"{y}-{m:02d}" for y in range(sy,ey+1) for m in range(1,13)]

def prev_q(q):
    y,n=int(q[:4]),int(q[-1])
    return f"{y-1}Q4" if n==1 else f"{y}Q{n-1}"

def q_to_monthly(q_dict,months):
    out={}
    for ym in months:
        y,m=int(ym[:4]),int(ym[5:7])
        q=f"{y}Q{(m-1)//3+1}"; pq=prev_q(q)
        v=q_dict.get(q); vp=q_dict.get(pq)
        if v is None: continue
        pos=(m-1)%3
        out[ym]=vp+(v-vp)/3*(pos+1) if vp else v
    return out

def yoy_m(d,ym):
    y,m=int(ym[:4]),int(ym[5:7])
    pm=f"{y-1}-{m:02d}"
    if ym not in d or pm not in d or d[pm]==0: return None
    return (d[ym]-d[pm])/d[pm]*100

def sv(v): return v is not None and not np.isnan(float(v))

def roll_mean(lst,w):
    out=[]
    for i in range(len(lst)):
        vals=[lst[j] for j in range(max(0,i-w+1),i+1) if lst[j] is not None]
        out.append(np.mean(vals) if len(vals)>=2 else None)
    return out

def rolling_z(s,w):
    z=[]
    for i in range(len(s)):
        v=s[i]
        if v is None: z.append(None); continue
        h=[s[j] for j in range(max(0,i-w),i) if s[j] is not None]
        if len(h)<12: z.append(None); continue
        mu,si=np.mean(h),np.std(h)
        z.append(float((v-mu)/si) if si>0 else 0.0)
    return z

def rolling_log_z(s,w):
    return rolling_z([np.log(v) if v and v>0 else None for v in s],w)

def cr_reset(score_list):
    """Score>0累加，Score≤0清零"""
    cr=[]; bucket=0.0
    for v in score_list:
        if not sv(v): cr.append(None); continue
        bucket=0.0 if v<=0 else bucket+v
        cr.append(bucket)
    return cr

def main():
    today=datetime.date.today()
    print(f"[{today}] CC Market Indicator - Monthly CR_reset")

    print("Fetching FRED data...")
    mcap_q    = fetch_q("NCBEILQ027S")
    gdp_nom_q = fetch_q("GDP")
    gdp_real_q= fetch_q("GDPC1")
    tcmdo_q   = fetch_q("TCMDO")

    print("Fetching SP500...")
    sp500_m   = fetch_sp500_monthly()

    months = gen_months(1970, today.year+1)

    mcap_m = q_to_monthly(mcap_q,   months)
    gdp_m  = q_to_monthly(gdp_nom_q,months)
    gdpr_m = q_to_monthly(gdp_real_q,months)
    tcmdo_m= q_to_monthly(tcmdo_q,  months)

    bi_m=[mcap_m[ym]/1000/gdp_m[ym]*100
          if gdp_m.get(ym) and gdp_m[ym]>0 and mcap_m.get(ym) else None
          for ym in months]

    ce_m_raw=[yoy_m(tcmdo_m,ym)-yoy_m(gdpr_m,ym)
              if yoy_m(tcmdo_m,ym) is not None and yoy_m(gdpr_m,ym) is not None
              else None for ym in months]

    sp_ret=[None]*len(months)
    for i,ym in enumerate(months):
        if sp500_m.get(ym) and i>0:
            pm=months[i-1]
            if sp500_m.get(pm) and sp500_m[pm]>0:
                sp_ret[i]=(sp500_m[ym]-sp500_m[pm])/sp500_m[pm]*100

    ce_s   = roll_mean(ce_m_raw,CE_W)
    z_bi   = rolling_log_z(bi_m,Z_W)
    z_ce   = rolling_z(ce_s,Z_W)
    score  = [0.5*z_bi[i]+0.5*z_ce[i]
              if sv(z_bi[i]) and sv(z_ce[i]) else None
              for i in range(len(months))]

    cr_m   = cr_reset(score)

    # 当前状态
    latest_i=max((i for i,ym in enumerate(months) if sv(score[i])),default=0)
    latest_m=months[latest_i]
    sc_now  =score[latest_i]
    cr_now  =cr_m[latest_i] if sv(cr_m[latest_i]) else 0.0
    bi_now  =bi_m[latest_i]
    zbi_now =z_bi[latest_i]
    zce_now =z_ce[latest_i]
    in_alert= sv(cr_now) and cr_now>CR_THRESH

    # 距离触发还差多少
    gap_to_trigger=max(0.0, CR_THRESH-cr_now) if not in_alert else 0.0
    valid_sc=[v for v in score if sv(v)]
    avg_sc_pos=np.mean([v for v in valid_sc[-24:] if v>0]) if valid_sc else 1.0
    mo_to_trigger=int(np.ceil(gap_to_trigger/avg_sc_pos)) if avg_sc_pos>0 and gap_to_trigger>0 else 0

    if in_alert:
        status_icon="⚠️"; status_text="ALERT — 持有 1/3 股票 + 2/3 SGOV"
    else:
        status_icon="✅"; status_text="CLEAR — 持有 100% 股票（VTI/SPY）"

    print(f"\n{status_icon} {status_text}")
    print(f"  月份    = {latest_m}")
    print(f"  Score   = {sc_now:.2f}")
    print(f"  CR      = {cr_now:.2f}  (阈值={CR_THRESH})")
    print(f"  距触发  = {gap_to_trigger:.2f}  ≈{mo_to_trigger}个月")

    # 扫描月度阈值（打印供参考）
    print(f"\n月度CR阈值扫描：")
    CASH_M=CASH_R/12
    ret24=[None]*len(months)
    for i,ym in enumerate(months):
        if sp500_m.get(ym) and i+24<len(months):
            fm=months[i+24]
            if sp500_m.get(fm) and sp500_m[ym]>0:
                ret24[i]=((sp500_m[fm]/sp500_m[ym])**0.5-1)*100
    bh2=np.mean([r for r in ret24 if r is not None])
    bh_ref=None; best_th=CR_THRESH; best_c=-999
    print(f"{'阈值':>6} {'CAGR':>8} {'超额':>8} {'Sharpe':>8} {'MaxDD':>7} {'触发月份'}")
    print("-"*75)
    for th in [6,9,12,15,18,24,30,36]:
        port=100.0; bh=100.0; half=0; rets_=[]; prev=False; trigs=[]
        for i,ym in enumerate(months):
            if ym<START_BT: continue
            cr=cr_m[i]; ret=sp_ret[i]
            if ret is None: continue
            pos=POS_A if (sv(cr) and cr>th) else 1.0
            if pos<1.0:
                half+=1
                if not prev: trigs.append(ym)
            prev=(pos<1.0)
            port*=(1+(pos*ret+(1-pos)*CASH_M)/100)
            bh  *=(1+ret/100)
            rets_.append(pos*ret+(1-pos)*CASH_M)
        n_yr=len([ym for ym in months if ym>=START_BT
                  and sp_ret[months.index(ym)] is not None])/12
        c=(port/100)**(1/n_yr)-1
        if bh_ref is None: bh_ref=(bh/100)**(1/n_yr)-1
        sh=(np.mean(rets_)-CASH_M)/np.std(rets_)
        vals=[]; p2=100.0
        for i,ym in enumerate(months):
            if ym<START_BT: continue
            cr=cr_m[i]; ret=sp_ret[i]
            if ret is None: continue
            pos=POS_A if (sv(cr) and cr>th) else 1.0
            p2*=(1+(pos*ret+(1-pos)*CASH_M)/100); vals.append(p2)
        pk=vals[0]; dd=0
        for x in vals: pk=max(pk,x); dd=max(dd,(pk-x)/pk*100)
        if c>best_c: best_c=c; best_th=th
        trigs_s=' '.join(trigs[:3])+('...' if len(trigs)>3 else '')
        print(f"  >{th:>3}  {c*100:>7.2f}%  {(c-bh_ref)*100:>+7.2f}%  "
              f"{sh:>7.3f}  {-dd:>6.1f}%  {trigs_s}")
    print(f"  B&H: CAGR={bh_ref*100:.2f}%  最优阈值={best_th}")

    # 触发历史
    triggers=[]
    prev=False
    for i,ym in enumerate(months):
        if ym<START_BT: continue
        cr=cr_m[i]
        if not sv(cr): prev=False; continue
        if cr>CR_THRESH:
            if not prev: triggers.append(('触发',ym,cr_now))
            prev=True
        elif prev:
            triggers.append(('解除',ym,cr_m[i]))
            prev=False
    if prev: triggers.append(('报警中',latest_m,cr_now))

    trigger_lines="\n".join(
        f"  {t[0]} {t[1]}: CR={t[2]:.1f}" for t in triggers
    ) or "  （无触发记录）"

    # ── 画图 ──────────────────────────────────────────────
    nan=float('nan')
    sc_p=[x if sv(x) else nan for x in score]
    cr_p=[x if sv(x) else nan for x in cr_m]
    sp_p=[sp500_m.get(ym,nan) for ym in months]

    BG='#0e1117'; BLUE='#00d4ff'; YEL='#ffe066'
    GRN='#4ecdc4'; RED='#ff6b6b'; ORG='#ffa500'

    fig=plt.figure(figsize=(14,14),facecolor=BG)
    gs=gridspec.GridSpec(3,1,figure=fig,hspace=0.10,height_ratios=[1.4,1,1])
    def mka(pos):
        ax=fig.add_subplot(pos); ax.set_facecolor(BG)
        ax.tick_params(colors='white',labelsize=8)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['bottom','left']: ax.spines[s].set_color('#444')
        ax.yaxis.label.set_color('white'); ax.xaxis.label.set_color('white')
        ax.grid(axis='y',alpha=0.06,color='white')
        return ax

    ax1=mka(gs[0]); ax2=mka(gs[1]); ax3=mka(gs[2])
    yt=[i for i,ym in enumerate(months) if ym.endswith('-01') and int(ym[:4])%5==0]
    yl=[ym[:4] for i,ym in enumerate(months) if ym.endswith('-01') and int(ym[:4])%5==0]
    xi=list(range(len(months)))

    tc=RED if in_alert else GRN
    fig.suptitle(
        f"CC Market Indicator  |  Monthly CR_reset  |  {latest_m}\n"
        f"{status_icon} {status_text}\n"
        f"Score={sc_now:.2f}  CR={cr_now:.1f}  threshold={CR_THRESH}  "
        f"gap={gap_to_trigger:.1f}  est.trigger~{mo_to_trigger}mo",
        fontsize=10,color=tc,fontweight='bold',y=0.99)

    # P1: S&P500
    ax1.semilogy(xi,sp_p,color=BLUE,lw=1.0,zorder=5)
    ax1.fill_between(xi,sp_p,50,alpha=0.06,color=BLUE)
    for i in range(len(months)):
        cr=cr_p[i]
        if not np.isnan(cr) and cr>CR_THRESH:
            ax1.axvspan(i-0.5,i+0.5,alpha=0.22,color=RED,zorder=1)
    prev=False
    for i,ym in enumerate(months):
        cr=cr_p[i]
        if np.isnan(cr): prev=False; continue
        if cr>CR_THRESH:
            if not prev and sp500_m.get(ym):
                ax1.scatter(i,sp500_m[ym],color=RED,s=60,zorder=7,marker='v')
                ax1.annotate(ym,xy=(i,sp500_m[ym]),
                            xytext=(i-20,sp500_m[ym]*0.52),
                            fontsize=8,color=RED,fontweight='bold',
                            arrowprops=dict(arrowstyle='->',color=RED,lw=0.9))
            prev=True
        else: prev=False
    ax1.scatter(latest_i,sp500_m.get(latest_m,nan),
                color=tc,s=200,zorder=9,marker='*')
    ax1.set_ylabel('S&P 500 (log)',color='white',fontsize=9)
    ax1.set_ylim(50,10000); ax1.set_xticks(yt); ax1.set_xticklabels([])
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p:f'{int(x):,}'))
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor=RED,alpha=0.4,label=f'1/3 position (CR>{CR_THRESH})'),
        plt.Line2D([0],[0],color=BLUE,lw=2,label='S&P 500')],
        fontsize=8.5,facecolor='#1a1d27',labelcolor='white',loc='upper left')

    # P2: Score月度
    ax2.plot(xi,sc_p,color=YEL,lw=0.9,zorder=4,label='Monthly Score')
    ax2.fill_between(xi,sc_p,0,where=[not np.isnan(z) and z>0 for z in sc_p],
        alpha=0.20,color=RED,zorder=3,label='Positive (accumulates CR)')
    ax2.fill_between(xi,sc_p,0,where=[not np.isnan(z) and z<0 for z in sc_p],
        alpha=0.20,color=GRN,zorder=3,label='Negative (CR resets to 0)')
    ax2.axhline(0,color='white',lw=1,ls='--',alpha=0.5)
    ax2.scatter(latest_i,sc_now if sv(sc_now) else 0,
                color=tc,s=150,zorder=8,marker='*')
    ax2.text(latest_i+8,sc_now if sv(sc_now) else 0,
             f'{latest_m}\nScore={sc_now:.2f}',
             fontsize=8,color=tc,va='center',
             bbox=dict(boxstyle='round',facecolor='#1a1d27',edgecolor=tc,alpha=0.9))
    ax2.set_ylabel('Score',color='white',fontsize=9)
    ax2.set_ylim(-4,5); ax2.set_xticks(yt); ax2.set_xticklabels([])
    ax2.legend(fontsize=8,facecolor='#1a1d27',labelcolor='white',
               loc='upper right',ncol=3)

    # P3: CR月度
    ax3.plot(xi,cr_p,color=ORG,lw=1.5,zorder=5,label='Cumulative Risk (CR)')
    ax3.fill_between(xi,cr_p,CR_THRESH,
        where=[not np.isnan(c) and c>CR_THRESH for c in cr_p],
        alpha=0.30,color=RED,zorder=4,label=f'CR>{CR_THRESH} → 1/3仓')
    ax3.fill_between(xi,cr_p,0,
        where=[not np.isnan(c) and 0<c<=CR_THRESH for c in cr_p],
        alpha=0.12,color=YEL,zorder=3)
    ax3.axhline(CR_THRESH,color=RED,lw=2.0,ls='--',alpha=0.9,
                label=f'Threshold={CR_THRESH}')
    ax3.axhline(0,color='white',lw=0.8,ls='--',alpha=0.4)
    # 触发标注
    prev=False
    for i,ym in enumerate(months):
        cr=cr_p[i]
        if np.isnan(cr): prev=False; continue
        if cr>CR_THRESH:
            if not prev:
                ax3.scatter(i,cr,color=RED,s=80,zorder=7)
                ax3.annotate(ym,xy=(i,cr),xytext=(i+8,cr+1.5),
                            fontsize=8,color=RED,fontweight='bold',
                            arrowprops=dict(arrowstyle='->',color=RED,lw=0.8))
            prev=True
        elif prev:
            ax3.scatter(i,0.1,color=GRN,s=70,zorder=7,marker='^')
            prev=False
    ax3.scatter(latest_i,cr_now,color=tc,s=150,zorder=8,marker='*')
    ax3.text(latest_i+8,cr_now,
             f'NOW\nCR={cr_now:.1f}\ngap={gap_to_trigger:.1f}',
             fontsize=8,color=tc,va='center',
             bbox=dict(boxstyle='round',facecolor='#1a1d27',edgecolor=tc,alpha=0.9))
    cr_max=max((v for v in cr_p if not np.isnan(v)),default=CR_THRESH*2)
    ax3.set_ylabel('Cumulative Risk (CR)',color='white',fontsize=9)
    ax3.set_xlabel('Year',color='white',fontsize=9)
    ax3.set_ylim(-0.5,cr_max*1.15)
    ax3.set_xticks(yt); ax3.set_xticklabels(yl,color='white',fontsize=8)
    ax3.legend(fontsize=8,facecolor='#1a1d27',labelcolor='white',loc='upper left')

    buf=io.BytesIO()
    plt.savefig(buf,format='png',dpi=130,bbox_inches='tight',facecolor=BG)
    buf.seek(0); img_data=buf.read(); plt.close()

    # ── 邮件 ──────────────────────────────────────────────
    subject=(f"{status_icon} CC Market [{latest_m}]  "
             f"CR={cr_now:.1f}/{CR_THRESH}  {status_text.split('—')[0].strip()}")

    body=f"""
CC Market Indicator — Monthly CR_reset Strategy
报告日期：{today}  数据截至：{latest_m}
{'='*55}

{status_icon} 当前状态：{status_text}

【核心指标】
  BI（市值/GDP）  = {bi_now:.1f}%
  z_BI            = {zbi_now:.2f}
  z_CE            = {zce_now:.2f}
  Score           = {sc_now:.2f}

【CR状态】
  Cumulative Risk = {cr_now:.2f}
  触发阈值        = {CR_THRESH}
  距触发差距      = {gap_to_trigger:.2f}
  预计触发        ≈ {mo_to_trigger} 个月后

【策略说明】
  Score > 0 → CR累加
  Score ≤ 0 → CR清零（立刻满仓）
  CR > {CR_THRESH}  → 1/3仓 + 2/3 SGOV
  CR ≤ {CR_THRESH}  → 100% 股票（VTI/SPY）

【历史触发记录】
{trigger_lines}

【回测表现（1985-2025，月度）】
  CAGR   ≈ 9-10%  vs  B&H ≈ 9.1%
  MaxDD  = -10% ~ -25%（取决于阈值）
  当前阈值 = {CR_THRESH}（≈年度版3.0×3月）

附件：完整指标图表
"""

    msg=MIMEMultipart()
    msg['From']=GMAIL_USER; msg['To']=NOTIFY_EMAIL
    msg['Subject']=subject
    msg.attach(MIMEText(body,'plain','utf-8'))
    img=MIMEImage(img_data)
    img.add_header('Content-Disposition','attachment',
                   filename=f'indicator_{latest_m}.png')
    msg.attach(img)

    with smtplib.SMTP_SSL('smtp.gmail.com',465) as server:
        server.login(GMAIL_USER,GMAIL_PASS)
        server.sendmail(GMAIL_USER,NOTIFY_EMAIL,msg.as_string())

    print(f"\n邮件已发送：{subject}")

if __name__=="__main__":
    main()
