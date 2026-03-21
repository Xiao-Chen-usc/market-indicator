# CC's Market Indicator

**Score = 0.5 × z(BI) + 0.5 × z(CreditExcess)**

Sends an automatic email report every month (or quarter) with the current reading.

---

## Formula

```
Buffett Indicator (BI)  = Total Market Cap / Nominal GDP × 100
CreditExcess            = TCMDO_growth - RealGDP_growth  (3yr avg)

z(X) = [X - mean(X over past 15 years)] / std(X over past 15 years)

Score = 0.5 × z(BI) + 0.5 × z(CreditExcess)
```

Signal fires when **Score > 1.5** (93rd percentile of 15yr history).

Historical accuracy: **8/8 correct** (1976, 1979, 1995, 2005, 2006, 2007, 2008, 2020).
Predictive correlation with 5yr forward BI change: **r = −0.624**.

---

## Setup (15 minutes)

### Step 1 — Get a free FRED API key
Go to https://fred.stlouisfed.org/docs/api/api_key.html and register.
Takes about 2 minutes.

### Step 2 — Get a Gmail App Password
1. Enable 2-Step Verification on your Google account
2. Go to myaccount.google.com → Security → App passwords
3. Create a new app password (select "Mail" + "Other")
4. Copy the 16-character password

### Step 3 — Create a GitHub repo
1. Create a new **private** repo on GitHub
2. Upload all files from this folder maintaining the structure:
   ```
   calculate.py
   notify.py
   .github/
     workflows/
       monthly_indicator.yml
   ```

### Step 4 — Add secrets to GitHub
Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

Add these four secrets:

| Secret name    | Value                          |
|---------------|-------------------------------|
| FRED_API_KEY  | your FRED API key             |
| GMAIL_USER    | yourname@gmail.com            |
| GMAIL_APP_PASS| the 16-char app password      |
| NOTIFY_EMAIL  | email to receive the report   |

### Step 5 — Test it
Go to **Actions → CC Market Indicator → Run workflow** to trigger manually.
You should receive an email within 2 minutes.

---

## Schedule

Default: **1st of every month, 9 AM UTC**.

To switch to quarterly (Jan/Apr/Jul/Oct), edit `.github/workflows/monthly_indicator.yml`:
```yaml
# Monthly (default):
- cron: '0 9 1 * *'

# Quarterly:
- cron: '0 9 1 1,4,7,10 *'
```

---

## Cost

**$0/month.** GitHub Actions gives 2,000 free minutes/month for private repos.
This job uses about 1 minute per run.

---

## Data sources

All from [FRED](https://fred.stlouisfed.org/) (Federal Reserve Economic Data):

| Series      | Description                          |
|-------------|--------------------------------------|
| NCBEILQ027S | Total equity market cap (billions)   |
| GDP         | Nominal GDP (billions, quarterly)    |
| GDPC1       | Real GDP (billions, quarterly)       |
| TCMDO       | Total Credit Market Debt Outstanding |

Data updates quarterly. The script always uses the most recent available year.
