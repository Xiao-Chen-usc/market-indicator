#!/usr/bin/env python3
"""
CC's Market Indicator - Monthly Buffer Strategy
notify_monthly.py

月度Buffer策略：
  Score = 0.5 * log_z(BI, 180月) + 0.5 * z(CE_12月, 180月)
  触发：pos_area > buffer  → 1/3仓
  解除：连续4个月Score < 0 → 满仓

GitHub Actions每月1日自动运行，发送邮件报告

环境变量：
  FRED_API_KEY   - FRED API密钥
  GMAIL_USER     - Gmail发件人
  GMAIL_APP_PASS - Gmail App Password
  NOTIFY_EMAIL   - 收件人邮箱
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

# ── 配置 ─────────────────────────────────────────────────
FRED_KEY     = os.environ.get("FRED_API_KEY","")
GMAIL_USER   = os.environ.get("GMAIL_USER","")
GMAIL_PASS   = os.environ.get("GMAIL_APP_PASS","")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL","")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
START        = "1970-01-01"

# 策略参数
CE_W    = 12    # CE平滑窗口（月）
Z_W     = 180   # z-score窗口（月=15年）
CONFIRM = 4     # 解除确认期（月）
POS_A   = 1/3   # 报警时仓位
CASH_R  = 1.0   # 现金年化收益%
START_BT= "1985-01"

# ── 数据获取 ─────────────────────────────────────────────
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

# ── 时间序列工具 ─────────────────────────────────────────
def gen_months(sy=1970,ey=None):
    if ey is None:
        t=datetime.date.today()
        ey=t.year+1
    return [f"{y}-{m:02d}" for y in range(sy,ey+1) for m in range(1,13)]

def prev_q(q):
    y,n=int(q[:4]),int(q[-1])
    return f"{y-1}Q4" if n==1 else f"{y}Q{n-1}"

def q_to_monthly(q_dict,months):
    out={}
    for ym in months:
        y,m=int(ym[:4]),int(ym[5:7])
        q=f"{y}Q{(m-1)//3+1}"
        pq=prev_q(q)
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

# ── Buffer CR（月度，N月确认解除）────────────────────────
def buffer_cr(score_list,confirm_n=4):
    sig=[]; buf_l=[]; pos_l=[]; st_l=[]
    buffer=0.0; pos_area=0.0; neg_area=0.0
    in_pos=False; alert=False; neg_streak=0
    for v in score_list:
        if not sv(v):
            sig.append(None); buf_l.append(buffer)
            pos_l.append(pos_area); st_l.append('none')
            continue
        if v>=0:
            neg_streak=0
            if not in_pos: in_pos=True
            pos_area+=v
            eff=max(0.0,pos_area-buffer)
            if eff>0: alert=True
        else:
            neg_streak+=1
            neg_area+=abs(v); buffer=neg_area
            if alert:
                if neg_streak>=confirm_n:
                    alert=False; in_pos=False; pos_area=0.0
                eff=max(0.0,pos_area-buffer) if alert else 0.0
            else:
                if in_pos: pos_area=0.0; in_pos=False
                eff=0.0
        sig.append(eff if alert else 0.0)
        buf_l.append(buffer); pos_l.append(pos_area)
        st_l.append('alert' if alert else 'full')
    return sig,buf_l,pos_l,st_l

# ── 主流程 ───────────────────────────────────────────────
def main():
    today=datetime.date.today()
    print(f"[{today}] CC Market Indicator - Monthly Buffer")

    print("Fetching FRED data...")
    mcap_q    = fetch_q("NCBEILQ027S")
    gdp_nom_q = fetch_q("GDP")
    gdp_real_q= fetch_q("GDPC1")
    tcmdo_q   = fetch_q("TCMDO")

    print("Fetching SP500 (yfinance)...")
    sp500_m   = fetch_sp500_monthly()

    months = gen_months(1970, today.year+1)

    # BI月度
    mcap_m = q_to_monthly(mcap_q,  months)
    gdp_m  = q_to_monthly(gdp_nom_q,months)
    gdpr_m = q_to_monthly(gdp_real_q,months)
    tcmdo_m= q_to_monthly(tcmdo_q, months)

    bi_m=[mcap_m[ym]/1000/gdp_m[ym]*100
          if gdp_m.get(ym) and gdp_m[ym]>0 and mcap_m.get(ym) else None
          for ym in months]

    ce_m_raw=[yoy_m(tcmdo_m,ym)-yoy_m(gdpr_m,ym)
              if yoy_m(tcmdo_m,ym) is not None and yoy_m(gdpr_m,ym) is not None
              else None for ym in months]

    # SP500月度收益
    sp_ret=[None]*len(months)
    for i,ym in enumerate(months):
        if sp500_m.get(ym) and i>0:
            pm=months[i-1]
            if sp500_m.get(pm) and sp500_m[pm]>0:
                sp_ret[i]=(sp500_m[ym]-sp500_m[pm])/sp500_m[pm]*100

    # Score
    ce_s   = roll_mean(ce_m_raw,CE_W)
    z_bi   = rolling_log_z(bi_m,Z_W)
    z_ce   = rolling_z(ce_s,Z_W)
    score  = [0.5*z_bi[i]+0.5*z_ce[i]
              if sv(z_bi[i]) and sv(z_ce[i]) else None
              for i in range(len(months))]

    # Buffer CR
    sig,buf_l,pos_l,st_l = buffer_cr(score,CONFIRM)

    # 当前状态
    latest_i=max((i for i,ym in enumerate(months) if sv(score[i])),default=0)
    latest_m=months[latest_i]
    sc_now  =score[latest_i]
    buf_now =buf_l[latest_i]
    pos_now =pos_l[latest_i]
    eff_now =sig[latest_i]
    st_now  =st_l[latest_i]
    bi_now  =bi_m[latest_i]
    zbi_now =z_bi[latest_i]
    zce_now =z_ce[latest_i]

    gap     = buf_now - pos_now
    in_alert= sv(eff_now) and eff_now>0
    pos_pct = POS_A*100 if in_alert else 100

    # 需要多少个月触发（估算）：只用有效Score的最后24个月
    valid_scores=[v for v in score if sv(v)]
    avg_score=np.mean(valid_scores[-24:]) if valid_scores else 0
    if avg_score>0 and gap>0:
        months_to_trigger=int(np.ceil(gap/avg_score))
    else:
        months_to_trigger=0

    # 状态字符串
    if in_alert:
        status_icon="⚠️"
        status_text=f"ALERT — 持有 1/3 股票 + 2/3 SGOV"
    elif st_now=='alert':
        # 确认期
        status_icon="🟡"
        status_text=f"确认期 — 仍持 1/3 仓，等待连续{CONFIRM}月Score<0"
    else:
        status_icon="✅"
        status_text=f"CLEAR — 持有 100% 股票（VTI/SPY）"

    print(f"\n当前状态：{status_icon} {status_text}")
    print(f"  月份     = {latest_m}")
    print(f"  BI       = {bi_now:.1f}%" if bi_now else "  BI = n/a")
    print(f"  Score    = {sc_now:.2f}" if sv(sc_now) else "  Score = n/a")
    print(f"  buffer   = {buf_now:.2f}")
    print(f"  pos_area = {pos_now:.2f}")
    print(f"  gap      = {gap:.2f}（还差{months_to_trigger}个月触发）")

    # ── 触发历史 ──────────────────────────────────────────
    triggers=[]
    prev2=False
    for i,ym in enumerate(months):
        if ym<START_BT: continue
        s=sig[i]
        if not sv(s): prev2=False; continue
        if s>0:
            if not prev2:
                triggers.append(('触发',ym,buf_l[i],pos_l[i]))
            prev2=True
        elif prev2:
            triggers.append(('解除',ym,buf_l[i],pos_l[i]))
            prev2=False

    # ── 画图 ──────────────────────────────────────────────
    nan=float('nan')
    sc_p =[x if sv(x) else nan for x in score]
    eff_p=[x if sv(x) else nan for x in sig]
    buf_p=[x if sv(x) else nan for x in buf_l]
    pa_p =[x if sv(x) else nan for x in pos_l]
    sp_p =[sp500_m.get(ym,nan) for ym in months]

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

    yt=[i for i,ym in enumerate(months)
        if ym.endswith('-01') and int(ym[:4])%5==0]
    yl=[ym[:4] for i,ym in enumerate(months)
        if ym.endswith('-01') and int(ym[:4])%5==0]
    xi=list(range(len(months)))

    title_color=RED if in_alert else (ORG if st_now=='alert' else GRN)
    fig.suptitle(
        f"CC Market Indicator  |  Monthly Buffer  |  {latest_m}\n"
        f"{status_icon} {status_text}\n"
        f"Score={sc_now:.2f}  buffer={buf_now:.1f}  pos={pos_now:.1f}  "
        f"gap={gap:.1f}  est.trigger~{months_to_trigger}mo",
        fontsize=10,color=title_color,fontweight='bold',y=0.99)

    # P1: S&P500
    ax1.semilogy(xi,sp_p,color=BLUE,lw=1.0,zorder=5)
    ax1.fill_between(xi,sp_p,50,alpha=0.06,color=BLUE)
    for i in range(len(months)):
        e=eff_p[i]; v=score[i]; st=st_l[i]
        if not np.isnan(e) and e>0:
            ax1.axvspan(i-0.5,i+0.5,alpha=0.22,color=RED,zorder=1)
        elif sv(v) and v<0 and st=='alert':
            ax1.axvspan(i-0.5,i+0.5,alpha=0.18,color=ORG,zorder=1)
    # 触发/解除标注
    prev=False
    for i,ym in enumerate(months):
        e=eff_p[i]
        if np.isnan(e): prev=False; continue
        if e>0:
            if not prev and sp500_m.get(ym):
                ax1.scatter(i,sp500_m[ym],color=RED,s=60,zorder=7,marker='v')
                ax1.annotate(ym,xy=(i,sp500_m[ym]),
                            xytext=(i-20,sp500_m[ym]*0.52),
                            fontsize=8,color=RED,fontweight='bold',
                            arrowprops=dict(arrowstyle='->',color=RED,lw=0.9))
            prev=True
        else: prev=False
    ax1.scatter(latest_i,sp500_m.get(latest_m,nan),
                color=title_color,s=200,zorder=9,marker='*')
    ax1.set_ylabel('S&P 500 (log)',color='white',fontsize=9)
    ax1.set_ylim(50,10000); ax1.set_xticks(yt); ax1.set_xticklabels([])
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x,p:f'{int(x):,}'))
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor=RED,alpha=0.4,label='1/3 position (alert)'),
        Patch(facecolor=ORG,alpha=0.4,label=f'Confirm period ({CONFIRM}mo)'),
        plt.Line2D([0],[0],color=BLUE,lw=2,label='S&P 500')],
        fontsize=8.5,facecolor='#1a1d27',labelcolor='white',loc='upper left')

    # P2: Score月度
    ax2.plot(xi,sc_p,color=YEL,lw=0.9,zorder=4)
    ax2.fill_between(xi,sc_p,0,where=[not np.isnan(z) and z>0 for z in sc_p],
        alpha=0.20,color=RED,zorder=3,label='Positive (accumulates)')
    ax2.fill_between(xi,sc_p,0,where=[not np.isnan(z) and z<0 for z in sc_p],
        alpha=0.20,color=GRN,zorder=3,label='Negative (builds buffer)')
    for i in range(len(months)):
        v=score[i]; st=st_l[i]
        if sv(v) and v<0 and st=='alert':
            ax2.axvspan(i-0.5,i+0.5,alpha=0.18,color=ORG,zorder=2)
    ax2.axhline(0,color='white',lw=1,ls='--',alpha=0.5)
    ax2.scatter(latest_i,sc_now if sv(sc_now) else 0,
                color=title_color,s=150,zorder=8,marker='*')
    ax2.text(latest_i+8,sc_now if sv(sc_now) else 0,
             f'{latest_m}\nScore={sc_now:.2f}' if sv(sc_now) else '',
             fontsize=8,color=title_color,va='center',
             bbox=dict(boxstyle='round',facecolor='#1a1d27',
                       edgecolor=title_color,alpha=0.9))
    ax2.set_ylabel('Score',color='white',fontsize=9)
    ax2.set_ylim(-4,5); ax2.set_xticks(yt); ax2.set_xticklabels([])
    ax2.legend(fontsize=8,facecolor='#1a1d27',labelcolor='white',
               loc='upper right',ncol=2)

    # P3: Buffer / pos_area / 距离触发
    buf_max=max((v for v in buf_p if not np.isnan(v)),default=10)
    ax3.fill_between(xi,buf_p,0,alpha=0.18,color=GRN,zorder=2,
                     label=f'Buffer={buf_now:.1f}（负面积）')
    ax3.fill_between(xi,pa_p, 0,alpha=0.15,color=YEL,zorder=3,
                     label=f'Pos area={pos_now:.1f}（正面积）')
    ax3.plot(xi,eff_p,color=RED,lw=1.5,zorder=5,label='Effective（触发量）')
    ax3.fill_between(xi,eff_p,0,where=[not np.isnan(e) and e>0 for e in eff_p],
        alpha=0.35,color=RED,zorder=4)
    ax3.axhline(0,color='white',lw=1,ls='--',alpha=0.5)
    # 标注当前gap
    ax3.annotate(
        f'Gap={gap:.1f}\n≈{months_to_trigger}mo to trigger',
        xy=(latest_i,pos_now),xytext=(latest_i-60,pos_now+15),
        fontsize=8.5,color=YEL,fontweight='bold',
        arrowprops=dict(arrowstyle='->',color=YEL,lw=0.9))
    ax3.set_ylabel('Buffer / Pos Area',color='white',fontsize=9)
    ax3.set_xlabel('Year',color='white',fontsize=9)
    ax3.set_ylim(-2,buf_max*1.12)
    ax3.set_xticks(yt); ax3.set_xticklabels(yl,color='white',fontsize=8)
    ax3.legend(fontsize=8,facecolor='#1a1d27',labelcolor='white',
               loc='upper left',ncol=3)

    buf=io.BytesIO()
    plt.savefig(buf,format='png',dpi=130,bbox_inches='tight',facecolor=BG)
    buf.seek(0); img_data=buf.read(); plt.close()

    # ── 发送邮件 ──────────────────────────────────────────
    subject=(f"{status_icon} CC Market Indicator [{latest_m}]  "
             f"Score={sc_now:.2f}  {status_text.split('—')[0].strip()}")

    # 触发历史文本
    trigger_lines="\n".join(
        f"  {t[0]} {t[1]}: buf={t[2]:.1f}  pos={t[3]:.1f}"
        for t in triggers) or "  （无触发记录）"

    body=f"""
CC Market Indicator — Monthly Buffer Strategy
报告日期：{today}  数据截至：{latest_m}
{'='*55}

{status_icon} 当前状态：{status_text}

【核心指标】
  BI（市值/GDP）  = {bi_now:.1f}%
  z_BI            = {zbi_now:.2f}
  z_CE            = {zce_now:.2f}
  Score           = {sc_now:.2f}

【Buffer状态】
  buffer (负面积) = {buf_now:.2f}
  pos_area(正面积)= {pos_now:.2f}
  gap（差距）     = {gap:.2f}
  预计触发        ≈ {months_to_trigger} 个月后（按近24月Score均值估算）

【策略说明】
  触发条件：pos_area > buffer
  解除条件：连续 {CONFIRM} 个月 Score < 0
  报警仓位：股票 1/3 + SGOV 2/3
  正常仓位：股票 100%（VTI / SPY）

【历史触发记录】
{trigger_lines}

【回测表现（1985-2025）】
  CAGR   = 10.07%  vs  B&H 9.11%
  超额   = +0.96%/yr
  Sharpe = 0.201（月度，不可与年度Sharpe直接比较）
  MaxDD  = -32.2%  vs  B&H -40.1%

附件：完整指标图表
"""

    msg=MIMEMultipart()
    msg['From']=GMAIL_USER; msg['To']=NOTIFY_EMAIL
    msg['Subject']=subject
    msg.attach(MIMEText(body,'plain','utf-8'))
    img=MIMEImage(img_data); img.add_header('Content-ID','<chart>')
    img.add_header('Content-Disposition','attachment',
                   filename=f'indicator_{latest_m}.png')
    msg.attach(img)

    with smtplib.SMTP_SSL('smtp.gmail.com',465) as server:
        server.login(GMAIL_USER,GMAIL_PASS)
        server.sendmail(GMAIL_USER,NOTIFY_EMAIL,msg.as_string())

    print(f"\n邮件已发送至 {NOTIFY_EMAIL}")
    print(f"主题：{subject}")

if __name__=="__main__":
    main()
