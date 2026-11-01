"""
VOXA AI v7 - Market Intelligence bot (trend+pullback strategy, HTF/ADX/cost filters, S/R, candle patterns,
strength score, Top-5 scanner, data-freshness, alerts, paper-trading journal & report)

Commands:
  !analyze RELIANCE 15m      -> chart + signal + SL/TP + data-quality notes
  !backtest TCS 1d [long|both]-> costs-aware backtest (70% train / 30% unseen test)
  !scan [tf] [all|crypto|india|gold|forex] -> watchlist scan, strongest signals pehle
  !watch add/remove/list/reset  -> watchlist (max 50 symbols)
  !alerts on [tf] / off / status -> auto-scan, naye signal par chart ke saath alert (is channel me)
  !capital 100000                -> alerts me quantity (1% risk) dikhane ke liye
  !top5 / !voxa [tf] [market]    -> Top 5 setups dashboard
  !report / !journal / !clearjournal -> paper trading record
  !guide                         -> saare commands

Symbols:
  India stock : RELIANCE, TCS ...   (auto .NS)   1d = delivery, intraday TF = same-day squareoff
  India index : NIFTY, BANKNIFTY, SENSEX        (futures-cost proxy)
  India gold  : GOLD / SILVER -> GOLDBEES.NS / SILVERBEES.NS (NSE ETFs, MCX spot nahi)
  Intl        : GOLDUSD (COMEX USD), OIL, AAPL, EURUSD=X
  Crypto      : BTC/USDT, ETHUSDT

Install: pip install discord.py ccxt pandas numpy mplfinance matplotlib requests
Run:     export DISCORD_TOKEN="token" && python trading_bot_v7.py

IMPORTANT: Neeche ke saare cost numbers approximate hain (rates 2026 tak ke).
Apne broker/exchange ke charges calculator se match karke edit karo.
"""
import os, io, time, json, asyncio, datetime
from collections import Counter
import numpy as np
import pandas as pd
import requests
import ccxt
import mplfinance as mpf
import discord
from discord.ext import commands, tasks

EXCHANGE = ccxt.binance({"enableRateLimit": True})

# ---------------- Config ----------------
RISK_PER_TRADE = 0.01   # har trade me max 1% equity risk
SL_ATR, TP_ATR = 1.5, 3.0
SLIP = 0.0005           # 0.05% slippage per fill
WARMUP = 300            # shuruaati candles hatao (EMA200 settle hone ke liye)
MIN_TRADES = 30         # isse kam trades = result statistically kamzor
ADX_MIN = 20            # isse kam ADX = sideways market, trade nahi
RSI_LONG, RSI_SHORT = 45, 55   # pullback ke baad momentum wapas aane ka trigger
COST_MULT = 4           # target kam se kam 4x round-trip cost hona chahiye
HTF_MAP = {"1m": "15m", "5m": "1h", "15m": "1h", "30m": "1d", "1h": "1d", "4h": "1d"}

# Brokerage (fraction of turnover, per side). Zerodha-type: intraday/futures Rs20 ya 0.03% (jo kam)
BROKERAGE_DELIVERY = 0.0
BROKERAGE_INTRADAY = 0.0003   # upper bound; bade trade me asli kam hota hai
BROKERAGE_FUT = 0.00002
GST = 0.18
SEBI = 0.000001         # Rs10 per crore


def india_cost(stt_buy, stt_sell, stamp_buy, exch, brokerage):
    base = exch + SEBI + brokerage + GST * (brokerage + exch + SEBI)
    return dict(buy=stt_buy + stamp_buy + base, sell=stt_sell + base)


# hold = roz ka holding cost (long+short dono), short_extra = short ka extra roz ka cost
# short_extra=None => us instrument me short allowed nahi
COSTS = {
    "crypto": dict(buy=0.001, sell=0.001, hold=0.0, short_extra=0.0003),  # funding ASSUMPTION
    # STT: delivery 0.1% dono side, intraday 0.025% sell, futures 0.05% sell (1 Apr 2026 se)
    "in_delivery": dict(**india_cost(0.001, 0.001, 0.00015, 0.0000297, BROKERAGE_DELIVERY),
                        hold=0.0, short_extra=None),
    "in_intraday": dict(**india_cost(0.0, 0.00025, 0.00003, 0.0000297, BROKERAGE_INTRADAY),
                        hold=0.0, short_extra=0.0),
    # Gold/Silver ETF: STT nahi lagta (broker se confirm karo)
    "in_delivery_etf": dict(**india_cost(0.0, 0.0, 0.00015, 0.0000297, BROKERAGE_DELIVERY),
                            hold=0.0, short_extra=None),
    "in_intraday_etf": dict(**india_cost(0.0, 0.0, 0.00003, 0.0000297, BROKERAGE_INTRADAY),
                            hold=0.0, short_extra=0.0),
    # Index futures proxy; hold = margin ka opportunity cost (approx)
    "in_futures": dict(**india_cost(0.0, 0.0005, 0.00002, 0.0000173, BROKERAGE_FUT),
                       hold=0.00005, short_extra=0.0),
    "global": dict(buy=0.0005, sell=0.0005, hold=0.0, short_extra=0.0002),
}

ALIASES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
           "GOLD": "GOLDBEES.NS", "SILVER": "SILVERBEES.NS",
           "GOLDUSD": "GC=F", "SILVERUSD": "SI=F", "OIL": "CL=F"}
ETFS = {"GOLDBEES.NS", "SILVERBEES.NS"}
SYMBOL_NOTES = {
    "GC=F": "Ye COMEX gold futures (USD/oz) hai, Indian/MCX gold price nahi.",
    "SI=F": "Ye COMEX silver futures (USD/oz) hai, Indian/MCX price nahi.",
    "CL=F": "Ye WTI crude (USD) hai, MCX crude nahi.",
    "GOLDBEES.NS": "NSE gold ETF hai: Indian gold ko track karta hai, par MCX spot se thoda alag.",
    "SILVERBEES.NS": "NSE silver ETF hai: MCX spot se thoda alag ho sakta hai.",
}
KIND_NOTES = {
    "in_delivery": "Delivery me short nahi hota. SELL = exit/avoid.",
    "in_delivery_etf": "Delivery me short nahi hota. SELL = exit/avoid.",
    "in_intraday": "Intraday: position usi din square-off karni hai.",
    "in_intraday_etf": "Intraday: position usi din square-off karni hai.",
    "in_futures": "Index sirf futures/options se trade hota hai. Lot size aur margin broker se dekho.",
}
DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729, "1d": 1825}


# ---------------- Symbols / timeframes ----------------
def tfsec(tf):
    return int(tf[:-1]) * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[tf[-1]]


def is_crypto(sym):
    return "/" in sym


def is_india(sym):
    return sym.endswith((".NS", ".BO")) or sym.startswith(("^NSE", "^BSE"))


def normalize(sym):
    s = sym.upper().strip()
    if s in ALIASES:
        return ALIASES[s]
    if "/" in s:
        return s
    if s.endswith("USDT"):
        return s[:-4] + "/USDT"
    if any(ch in s for ch in ".^="):
        return s
    return s + ".NS"


def classify(sym, tf):
    if is_crypto(sym):
        return "crypto"
    if sym.startswith(("^NSE", "^BSE")):
        return "in_futures"
    if sym.endswith((".NS", ".BO")):
        base = "in_delivery" if tf == "1d" else "in_intraday"
        return base + ("_etf" if sym in ETFS else "")
    return "global"


# ---------------- Data ----------------
def fetch_yahoo(sym, tf, limit):
    if tf not in DAYS:
        raise ValueError(f"Timeframe {tf} supported nahi. Use: {', '.join(DAYS)}")
    now = int(time.time())
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"interval": tf, "period1": now - DAYS[tf] * 86400, "period2": now},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]},
                      index=pd.to_datetime(res["timestamp"], unit="s"))
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    df.index = df.index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    df.index.name = "time"
    df["volume"] = df["volume"].fillna(0)
    out = df.dropna(subset=["open", "high", "low", "close"]).tail(limit)
    out.attrs["tz"] = tz
    return out


def fetch_crypto_rows(symbol, tf, limit):
    ms = tfsec(tf) * 1000
    rows = EXCHANGE.fetch_ohlcv(symbol, tf, limit=min(limit, 1000))
    while len(rows) < limit:
        need = min(limit - len(rows), 1000)
        more = EXCHANGE.fetch_ohlcv(symbol, tf, since=rows[0][0] - need * ms, limit=need)
        more = [r for r in more if r[0] < rows[0][0]]
        if not more:
            break
        rows = more + rows
    return rows


def fetch(symbol, tf, limit=500):
    if is_crypto(symbol):
        raw = fetch_crypto_rows(symbol, tf, limit)
        df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.set_index("time")
        df.attrs["tz"] = "UTC"
        return df
    try:
        return fetch_yahoo(symbol, tf, limit)
    except Exception:
        if symbol.endswith(".NS") and symbol not in ETFS:  # shayad US stock ho
            return fetch_yahoo(symbol[:-3], tf, limit)
        raise


def load(symbol, tf, limit):
    """Data + quality checks. Returns (df, notes, gap_pct)."""
    df = fetch(symbol, tf, limit)
    tz = df.attrs.get("tz", "UTC")
    now = pd.Timestamp.now(tz=tz).tz_localize(None)
    sec, notes = tfsec(tf), []

    # 1) adhoori (chalu) candle hatao taaki signal repaint na ho
    last = df.index[-1]
    if tf == "1d" and is_india(symbol):
        end = last.normalize() + pd.Timedelta(hours=15, minutes=30)
    else:
        end = last + pd.Timedelta(seconds=sec)
    if end > now and len(df) > 1:
        df = df.iloc[:-1]
        notes.append("Chalu candle hata di (signal sirf band candle par)")

    # 2) missing candles
    gap_pct = 0.0
    if sec < 86400:
        s = df.index.to_series()
        d = s.diff().dt.total_seconds()
        mask = d > 1.5 * sec
        if not is_crypto(symbol):
            mask &= (s.dt.date == s.dt.date.shift())  # raat/weekend gap ginti me nahi
        gaps = int(mask.sum())
        gap_pct = gaps / len(df) * 100
        if gaps:
            notes.append(f"{gaps} jagah candles missing ({gap_pct:.1f}%)")
        if gap_pct > 2:
            notes.append("Missing data zyada hai: backtest/signal par bharosa kam rakho")

    # 3) data kitna purana hai
    age_min = (now - df.index[-1]).total_seconds() / 60
    notes.append(f"Last closed candle: {df.index[-1]:%d-%b %H:%M} ({age_min:.0f} min purani)")
    if is_india(symbol):
        open_now = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)
        if open_now and age_min > 2 * sec / 60 + 20:
            notes.append("Market khula hai par data purana lag raha hai (Yahoo delay)")
    return df, notes, gap_pct


def cost_ok(c, d, price, atr):
    ef, xf = (c["buy"], c["sell"]) if d == 1 else (c["sell"], c["buy"])
    return TP_ATR * atr >= COST_MULT * price * (ef + xf + 2 * SLIP)


HTF_CACHE = {}


def get_htf(symbol, htf_tf):
    key, now = (symbol, htf_tf), time.time()
    hit = HTF_CACHE.get(key)
    ttl = min(3600, max(300, tfsec(htf_tf) / 4))
    if hit and now - hit[0] < ttl:
        return hit[1]
    df, _, _ = load(symbol, htf_tf, 1000 if is_crypto(symbol) else 1500)
    HTF_CACHE[key] = (now, df)
    return df


def prepare(symbol, tf, limit):
    """Data + HTF + indicators. Returns (df, notes, gap_pct, age_min)."""
    df, notes, gap = load(symbol, tf, limit)
    tz = df.attrs.get("tz", "UTC")
    age_min = (pd.Timestamp.now(tz=tz).tz_localize(None) - df.index[-1]).total_seconds() / 60
    htf_tf, htf = HTF_MAP.get(tf), None
    if htf_tf:
        try:
            htf = get_htf(symbol, htf_tf)
            notes.append(f"HTF filter: {htf_tf}")
        except Exception:
            notes.append(f"{htf_tf} data nahi mila: bade timeframe ka filter band")
    else:
        notes.append("Is timeframe par bade timeframe ka filter nahi")
    return add_indicators(df, tf, htf, htf_tf), notes, gap, age_min


# ---------------- Indicators ----------------
def add_indicators(df, tf="1h", htf=None, htf_tf=None):
    df = df.copy()
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd"], df["macd_sig"], df["hist"] = macd, sig, macd - sig

    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = ma + 2 * sd, ma - 2 * sd

    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    s = pd.Series(0, index=df.index)
    s += np.where(c > df["ema200"], 1, -1)
    s += np.where(df["ema20"] > df["ema50"], 1, -1)
    s += np.where((df["hist"] > 0) & (df["hist"] > df["hist"].shift()), 1,
                  np.where((df["hist"] < 0) & (df["hist"] < df["hist"].shift()), -1, 0))
    s += np.where(df["rsi"] < 30, 1, np.where(df["rsi"] > 70, -1, 0))
    s += np.where(c < df["bb_lo"], 1, np.where(c > df["bb_up"], -1, 0))
    df["score"] = s

    # ADX (trend ki taakat)
    up_m, dn_m = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Bade timeframe ka trend: sirf tab use hota hai jab HTF candle BAND ho chuki ho (no look-ahead)
    if htf is not None and htf_tf:
        hc = htf["close"]
        e50 = hc.ewm(span=50, adjust=False).mean()
        e200 = hc.ewm(span=200, adjust=False).mean()
        ht = pd.Series(np.where((hc > e200) & (e50 > e200), 1, np.where((hc < e200) & (e50 < e200), -1, 0)),
                       index=htf.index).iloc[WARMUP:]
        ht.index = ht.index + pd.Timedelta(seconds=tfsec(htf_tf))
        lc = df.index + pd.Timedelta(seconds=tfsec(tf))
        df["htf"] = ht.reindex(ht.index.union(lc)).ffill().reindex(lc).values
    else:
        df["htf"] = np.nan

    # Naya setup: trend ke saath pullback ke baad momentum wapas
    trend_up = (c > df["ema200"]) & (df["ema50"] > df["ema200"])
    trend_dn = (c < df["ema200"]) & (df["ema50"] < df["ema200"])
    htf_up = (df["htf"] == 1) | df["htf"].isna()
    htf_dn = (df["htf"] == -1) | df["htf"].isna()
    reg = df["adx"] >= ADX_MIN
    long_t = ((df["rsi"].rolling(4).min() < RSI_LONG) & (df["rsi"] >= RSI_LONG) & (df["rsi"] < 65)
              & (df["hist"] > df["hist"].shift()))
    short_t = ((df["rsi"].rolling(4).max() > RSI_SHORT) & (df["rsi"] <= RSI_SHORT) & (df["rsi"] > 35)
               & (df["hist"] < df["hist"].shift()))
    df["setup"] = np.where(trend_up & htf_up & reg & long_t, 1,
                           np.where(trend_dn & htf_dn & reg & short_t, -1, 0))
    df["setup_old"] = np.where(df["score"] >= 3, 1, np.where(df["score"] <= -3, -1, 0))

    # pehle WARMUP candles ke indicators unstable hote hain -> hata do
    df = df.iloc[WARMUP:].dropna(subset=[x for x in df.columns if x != "htf"])
    if len(df) < 100:
        raise ValueError(f"Data kam hai (warm-up ke baad {len(df)} candles). Bada timeframe try karo.")
    return df


def signal_from_score(score):
    return "BUY" if score >= 3 else ("SELL" if score <= -3 else "NEUTRAL")


# ---------------- Chart ----------------
def make_chart(df, symbol, tf, levels=None):
    d = df.tail(120).rename(columns=str.title)
    vol = bool(d["Volume"].sum() > 0)
    p = 2 if vol else 1
    ap = [
        mpf.make_addplot(d["Ema20"], color="orange", width=0.8),
        mpf.make_addplot(d["Ema50"], color="blue", width=0.8),
        mpf.make_addplot(d["Ema200"], color="purple", width=0.8),
        mpf.make_addplot(d["Bb_Up"], color="gray", linestyle="--", width=0.6),
        mpf.make_addplot(d["Bb_Lo"], color="gray", linestyle="--", width=0.6),
        mpf.make_addplot(d["Rsi"], panel=p, color="teal", ylabel="RSI"),
        mpf.make_addplot(d["Macd"], panel=p + 1, color="blue", ylabel="MACD"),
        mpf.make_addplot(d["Macd_Sig"], panel=p + 1, color="red"),
        mpf.make_addplot(d["Hist"], panel=p + 1, type="bar", color="gray"),
    ]
    buf = io.BytesIO()
    kw = {}
    if levels and levels[0]:
        kw["hlines"] = dict(hlines=levels[0], colors=levels[1], linestyle="-.", linewidths=1.0)
    mpf.plot(d, type="candle", style="charles", addplot=ap, volume=vol,
             title=f"{symbol} {tf}", panel_ratios=(5, 1, 2, 2) if vol else (5, 2, 2),
             savefig=dict(fname=buf, dpi=100, format="png"), **kw)
    buf.seek(0)
    return buf


# ---------------- Backtest ----------------
def run_bt(df, kind, tf, allow_short=False, col="setup"):
    c, sec = COSTS[kind], tfsec(tf)
    can_short = allow_short and c["short_extra"] is not None
    intraday = kind.startswith("in_intraday")
    dates, n = df.index.date, len(df)

    def fees(d):  # (entry_fee, exit_fee): long = buy phir sell, short = sell phir buy
        return (c["buy"], c["sell"]) if d == 1 else (c["sell"], c["buy"])

    def settle(p, raw_exit):
        ex = raw_exit * (1 - p["dir"] * SLIP)
        gross = p["dir"] * (ex - p["entry"]) * p["size"]
        return gross - ex * p["size"] * fees(p["dir"])[1] - p["efee"] - p["carry"]

    equity, pos, pnls, curve = 1000.0, None, [], []
    for i in range(n):
        r = df.iloc[i]
        eod = intraday and (i == n - 1 or dates[i + 1] != dates[i])
        done = False
        if pos:
            d = pos["dir"]
            rate = c["hold"] + (c["short_extra"] if d == -1 else 0.0)
            pos["carry"] += pos["entry"] * pos["size"] * rate * sec / 86400
            sl_hit = r["low"] <= pos["sl"] if d == 1 else r["high"] >= pos["sl"]
            tp_hit = r["high"] >= pos["tp"] if d == 1 else r["low"] <= pos["tp"]
            if sl_hit or tp_hit:
                if sl_hit:  # SL pehle maana; gap ho to open par exit
                    ex = min(r["open"], pos["sl"]) if d == 1 else max(r["open"], pos["sl"])
                else:
                    ex = pos["tp"]
            elif eod:
                ex = r["close"]  # intraday square-off
            else:
                ex = None
            if ex is not None:
                pnl = settle(pos, ex)
                equity += pnl
                pnls.append(pnl)
                pos, done = None, True
        if pos is None and not done and not eod:
            d = int(r[col]) if (r[col] == 1 or (can_short and r[col] == -1)) else 0
            if d and not cost_ok(c, d, r["close"], r["atr"]):
                d = 0
            if d:
                ef, xf = fees(d)
                entry = r["close"] * (1 + d * SLIP)
                risk = SL_ATR * r["atr"] + entry * (ef + xf + 2 * SLIP)  # costs bhi risk me
                size = min(equity * RISK_PER_TRADE / risk, equity / entry)  # no leverage
                pos = dict(dir=d, entry=entry, size=size, efee=entry * size * ef, carry=0.0,
                           sl=entry - d * SL_ATR * r["atr"], tp=entry + d * TP_ATR * r["atr"])
        unreal = 0.0
        if pos:
            unreal = pos["dir"] * (r["close"] - pos["entry"]) * pos["size"] - pos["efee"] - pos["carry"]
        curve.append(equity + unreal)
    if pos:  # aakhri open trade band karo
        pnl = settle(pos, df["close"].iloc[-1])
        equity += pnl
        pnls.append(pnl)
        curve[-1] = equity
    cv = pd.Series(curve)
    dd = ((cv.cummax() - cv) / cv.cummax()).max() * 100
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p <= 0]
    pf = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float("inf")
    return dict(trades=len(pnls), wr=len(wins) / len(pnls) * 100 if pnls else 0.0,
                ret=(equity / 1000 - 1) * 100, dd=dd, pf=pf,
                bh=(df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100)


def fmt(name, s):
    return (f"{name}: trades {s['trades']} | win {s['wr']:.0f}% | return {s['ret']:.1f}% | "
            f"maxDD {s['dd']:.1f}% | PF {s['pf']:.2f} | buy&hold {s['bh']:.1f}%")


# ---------------- Discord ----------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# ---------------- Voxa engine: S/R, candle patterns, market status, evaluation ----------------
def sr_levels(df, n=5, lookback=300):
    d = df.tail(lookback)
    hi, lo = d["high"].values, d["low"].values
    atr, price = float(d["atr"].iloc[-1]), float(d["close"].iloc[-1])
    piv = []
    for i in range(n, len(d) - n):
        if hi[i] == hi[i - n:i + n + 1].max():
            piv.append(float(hi[i]))
        if lo[i] == lo[i - n:i + n + 1].min():
            piv.append(float(lo[i]))
    piv.sort()
    cl = []  # [sum, count]
    for p in piv:
        if cl and p - cl[-1][0] / cl[-1][1] <= 0.5 * atr:
            cl[-1][0] += p
            cl[-1][1] += 1
        else:
            cl.append([p, 1])
    lv = [s / k for s, k in cl if k >= 2] or [s / k for s, k in cl]
    return dict(support=sorted([x for x in lv if x < price], reverse=True)[:2],
                resistance=sorted([x for x in lv if x > price])[:2])


def candle_patterns(df):
    """Last candle ke patterns: [(naam, bias)] bias +1 bullish, -1 bearish, 0 neutral. Sirf reference."""
    if len(df) < 8:
        return []
    r = df.tail(3)
    o, h, l, c = (r[k].values for k in ("open", "high", "low", "close"))
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    up, lw = h - np.maximum(o, c), np.minimum(o, c) - l
    cl = df["close"]
    prior_down, prior_up = cl.iloc[-2] < cl.iloc[-6], cl.iloc[-2] > cl.iloc[-6]
    out = []
    if body[2] <= 0.1 * rng[2]:
        out.append(("Doji", 0))
    if lw[2] >= 2 * body[2] and up[2] <= 0.15 * rng[2] and body[2] > 0.05 * rng[2] and prior_down:
        out.append(("Hammer", 1))
    if up[2] >= 2 * body[2] and lw[2] <= 0.15 * rng[2] and body[2] > 0.05 * rng[2] and prior_up:
        out.append(("Shooting star", -1))
    if c[1] < o[1] and c[2] > o[2] and o[2] <= c[1] and c[2] >= o[1]:
        out.append(("Bullish engulfing", 1))
    if c[1] > o[1] and c[2] < o[2] and o[2] >= c[1] and c[2] <= o[1]:
        out.append(("Bearish engulfing", -1))
    if c[0] < o[0] and body[0] >= 0.5 * rng[0] and body[1] <= 0.3 * rng[1] and c[2] > o[2] and c[2] > (o[0] + c[0]) / 2:
        out.append(("Morning star", 1))
    if c[0] > o[0] and body[0] >= 0.5 * rng[0] and body[1] <= 0.3 * rng[1] and c[2] < o[2] and c[2] < (o[0] + c[0]) / 2:
        out.append(("Evening star", -1))
    return out


def market_status(sym):
    """OPEN/CLOSED (NSE holidays check nahi hote)."""
    now = datetime.datetime.utcnow()
    if is_crypto(sym):
        return "OPEN"
    if is_india(sym):
        ist = now + datetime.timedelta(hours=5, minutes=30)
        return "OPEN" if ist.weekday() < 5 and (9, 15) <= (ist.hour, ist.minute) <= (15, 30) else "CLOSED"
    wd, hr = now.weekday(), now.hour
    return "CLOSED" if (wd == 5 or (wd == 4 and hr >= 22) or (wd == 6 and hr < 22)) else "OPEN"


def freshness(mkt, tf, age_min):
    if mkt == "CLOSED":
        return "CLOSED"
    m = tfsec(tf) / 60
    return "LIVE" if age_min <= 2 * m + 10 else ("DELAYED" if age_min <= 4 * m + 30 else "STALE")


def evaluate(symbol, tf, df, notes, gap, age_min):
    last = df.iloc[-1]
    c, atr = float(last["close"]), float(last["atr"])
    kind = classify(symbol, tf)
    mkt = market_status(symbol)
    fresh = freshness(mkt, tf, age_min)
    setup = int(last["setup"])
    trend_up = bool(c > last["ema200"] and last["ema50"] > last["ema200"])
    trend_dn = bool(c < last["ema200"] and last["ema50"] < last["ema200"])
    adx_ok = bool(last["adx"] >= ADX_MIN)
    d = setup if setup else (1 if trend_up else (-1 if trend_dn else 0))
    trend = "Bullish" if (trend_up and adx_ok) else ("Bearish" if (trend_dn and adx_ok) else "Sideways")
    sr, pats = sr_levels(df), candle_patterns(df)
    ok_cost = True if d == 0 else bool(cost_ok(COSTS[kind], d, c, atr))
    checks = []
    if d != 0:
        h = last["htf"]
        checks.append(("EMA200 trend", (d == 1 and trend_up) or (d == -1 and trend_dn), 15))
        checks.append(("EMA20/50", bool((last["ema20"] > last["ema50"]) == (d == 1)), 10))
        if not pd.isna(h):
            checks.append(("HTF trend", int(h) == d, 15))
        checks.append(("ADX", adx_ok, 10))
        rs = float(last["rsi"])
        checks.append(("RSI zone", (45 <= rs <= 65) if d == 1 else (35 <= rs <= 55), 10))
        hp = float(df["hist"].iloc[-2])
        checks.append(("MACD momentum", bool(last["hist"] > hp) if d == 1 else bool(last["hist"] < hp), 10))
        checks.append(("Pullback trigger", setup == d, 15))
        if df["volume"].tail(20).sum() > 0:
            checks.append(("Volume", bool(last["volume"] > df["volume"].tail(20).mean()), 5))
        al, ct = [n for n, b in pats if b == d], [n for n, b in pats if b == -d]
        if al or ct:
            checks.append(("Candle pattern", bool(al) and not ct, 5))
        opp = sr["resistance"] if d == 1 else sr["support"]
        checks.append(("S/R room", (not opp) or abs(opp[0] - c) >= 1.5 * atr, 10))
        checks.append(("Costs", ok_cost, 5))
    tot = sum(w for _, _, w in checks)
    strength = int(round(100 * sum(w for _, p, w in checks if p) / tot)) if tot else 0
    why = None
    if setup == 0:
        why = "Pullback setup nahi bana" if d else "Trend saaf nahi"
    elif not ok_cost:
        why = "Charges ke baad target chhota"
    elif fresh == "STALE":
        why = "Data STALE: naya signal nahi"
    sig = "WAIT" if why else ("BUY" if setup == 1 else "SELL")
    passed = [n for n, p, w in sorted(checks, key=lambda x: -x[2]) if p][:3]
    reason = why or (" + ".join(passed) or "-")
    if why and passed:
        reason = f"{why} ({' + '.join(passed[:2])} theek)"
    atrp = (df["atr"] / df["close"]).tail(300)
    pct = float((atrp <= atrp.iloc[-1]).mean())
    risk = "Low" if pct < 0.35 else ("Medium" if pct < 0.75 else "High")
    return dict(symbol=symbol, tf=tf, price=c, atr=atr, signal=sig, dir=d, setup=setup, strength=strength,
                n_checks=len(checks), trend=trend, risk=risk, reason=reason, mkt=mkt, fresh=fresh,
                sr=sr, patterns=pats, cost_ok=ok_cost, gap=gap, age_min=age_min, notes=notes,
                time=str(df.index[-1]), score=int(last["score"]), kind=kind)


def bar(n):
    k = int(round(n / 10))
    return "█" * k + "░" * (10 - k)


def track_record(symbol):
    tr = [t for t in JOURNAL if t["symbol"] == symbol and t["status"] == "closed"]
    if len(tr) < 5:
        return None
    gw = sum(t["r"] for t in tr if t["r"] > 0)
    gl = -sum(t["r"] for t in tr if t["r"] <= 0)
    pf = gw / gl if gl > 0 else float("inf")
    return f"Paper record: {sum(1 for t in tr if t['r'] > 0)}/{len(tr)} jeete | PF {pf:.2f}"


def quick_backtest(df, symbol, tf):
    kind = classify(symbol, tf)
    cut = int(len(df) * 0.7)
    te = run_bt(df.iloc[cut:], kind, tf, False, "setup")
    txt = f"{te['trades']} trades, PF {te['pf']:.2f}, return {te['ret']:.1f}%, maxDD {te['dd']:.1f}%"
    if te["trades"] < MIN_TRADES:
        txt += f" (sample chhota <{MIN_TRADES}: bharosa mat karo)"
    return txt


def analysis_embed(ev, bt=None):
    sig = ev["signal"]
    color = {"BUY": discord.Color.green(), "SELL": discord.Color.red()}.get(sig, discord.Color.light_grey())
    e = discord.Embed(title=f"VOXA AI · {ev['symbol']} · {ev['tf']}", color=color,
                      description=f"**{sig}** · Strength **{ev['strength']}/100** {bar(ev['strength'])}\n"
                                  f"Price: {ev['price']:.4f}")
    e.add_field(name="Trend / Risk", value=f"{ev['trend']} · Risk: {ev['risk']}", inline=True)
    e.add_field(name="Market / Data", value=f"{ev['mkt']} · {ev['fresh']}", inline=True)
    e.add_field(name="Reason", value=ev["reason"][:300], inline=False)
    if sig != "WAIT":
        d, p, atr = ev["dir"], ev["price"], ev["atr"]
        txt = (f"{'LONG' if d == 1 else 'SHORT'} entry {p:.4f}\nStop-loss {p - d * SL_ATR * atr:.4f}\n"
               f"Target {p + d * TP_ATR * atr:.4f}\nRisk: max 1% capital")
        cap = STATE.get("capital")
        if cap:
            txt += f"\nQty (1% risk, capital {cap:g}): {min(cap * RISK_PER_TRADE / (SL_ATR * atr), cap / p):.4g}"
        if sig == "SELL" and COSTS[ev["kind"]]["short_extra"] is None:
            txt += "\nDelivery me short nahi hota: SELL = exit/avoid"
        e.add_field(name="Trade plan", value=txt, inline=False)
    sr = ev["sr"]
    e.add_field(name="Support / Resistance",
                value=("S: " + (", ".join(f"{x:.4g}" for x in sr["support"]) or "-") + "\nR: "
                       + (", ".join(f"{x:.4g}" for x in sr["resistance"]) or "-")), inline=True)
    pt = ", ".join(f"{n} ({'bullish' if b > 0 else 'bearish' if b < 0 else 'neutral'})" for n, b in ev["patterns"])
    e.add_field(name="Candle pattern (reference)", value=pt or "Koi nahi", inline=True)
    if bt:
        e.add_field(name="Backtest (unseen 30%)", value=bt[:300], inline=False)
    tr = track_record(ev["symbol"])
    if tr:
        e.add_field(name="Track record", value=tr, inline=False)
    extra = [SYMBOL_NOTES[ev["symbol"]]] if ev["symbol"] in SYMBOL_NOTES else []
    if ev["kind"] in KIND_NOTES:
        extra.append(KIND_NOTES[ev["kind"]])
    e.add_field(name="Data quality", value=" | ".join(ev["notes"] + extra)[:600] or "-", inline=False)
    e.set_footer(text="Strength = rules ka score, jeetne ki probability nahi. Not financial advice. Verify karo.")
    return e


def chart_levels(ev, df):
    t = df.tail(120)
    lo, hi, atr = float(t["low"].min()), float(t["high"].max()), ev["atr"]
    prices, colors = [], []
    for x in ev["sr"]["support"]:
        if lo - 2 * atr <= x <= hi + 2 * atr:
            prices.append(x)
            colors.append("green")
    for x in ev["sr"]["resistance"]:
        if lo - 2 * atr <= x <= hi + 2 * atr:
            prices.append(x)
            colors.append("red")
    return prices, colors


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    if not auto_scan.is_running():
        auto_scan.start()


@bot.command()
async def analyze(ctx, symbol: str = "BTC/USDT", tf: str = "1h"):
    symbol = normalize(symbol)
    try:
        df, notes, gap, age = await asyncio.to_thread(prepare, symbol, tf, 1500)
    except Exception as e:
        return await ctx.send(f"Data error: {e}")
    ev = evaluate(symbol, tf, df, notes, gap, age)
    bt = await asyncio.to_thread(quick_backtest, df, symbol, tf)
    emb = analysis_embed(ev, bt)
    chart = await asyncio.to_thread(make_chart, df, symbol, tf, chart_levels(ev, df))
    emb.set_image(url="attachment://chart.png")
    await ctx.send(embed=emb, file=discord.File(chart, "chart.png"))


@bot.command(name="guide")
async def guide_cmd(ctx):
    await ctx.send(
        "**VOXA AI commands**\n"
        "`!top5 [tf] [all|crypto|india|gold|forex]` - top 5 setups (alias `!voxa`)\n"
        "`!analyze RELIANCE 15m` - poora analysis + chart (S/R ke saath)\n"
        "`!scan 1h india` - poori watchlist ki list\n"
        "`!backtest TCS 1d both` - costs ke saath past test\n"
        "`!watch add SBIN BTC/USDT` | `!watch remove SBIN` | `!watch list` | `!watch reset`\n"
        "`!alerts on 1h` | `!alerts off` | `!alerts status`\n"
        "`!report` - paper trading performance | `!journal 10` - last trades | `!clearjournal yes`\n"
        "`!capital 100000` - alerts me quantity (1% risk)\n"
        "Symbols: RELIANCE, NIFTY, BANKNIFTY, GOLD (NSE ETF), GOLDUSD (COMEX), EURUSD=X, BTC/USDT")


@bot.command(name="backtest")
async def backtest_cmd(ctx, symbol: str = "BTC/USDT", tf: str = "1h", mode: str = "long"):
    symbol = normalize(symbol)
    try:
        df, notes, gap, _ = await asyncio.to_thread(prepare, symbol, tf, 4000 if is_crypto(symbol) else 1500)
    except Exception as e:
        return await ctx.send(f"Data error: {e}")
    kind = classify(symbol, tf)
    c = COSTS[kind]
    short = mode.lower() == "both"
    lines = [f"**Backtest {symbol} {tf}** [{kind}] {'Long+Short' if short else 'Long only'}",
             f"Cost/side: buy {c['buy']*100:.3f}% | sell {c['sell']*100:.3f}% | slippage {SLIP*100:.2f}%"]
    if short and c["short_extra"] is None:
        lines.append("Is instrument me short allowed nahi (delivery), isliye sirf long chala.")
    elif short:
        lines.append(f"Short carry (assumption): {c['short_extra']*100:.3f}%/din. Asli funding/borrow alag hoga.")
    n = len(df)
    cut = int(n * 0.7)
    tr = run_bt(df.iloc[:cut], kind, tf, short, "setup")
    te = run_bt(df.iloc[cut:], kind, tf, short, "setup")
    old = run_bt(df.iloc[cut:], kind, tf, short, "setup_old")
    folds = [run_bt(df.iloc[k * n // 3:(k + 1) * n // 3], kind, tf, short, "setup") for k in range(3)]
    lines += [fmt("NAYA pehla 70%", tr), fmt("NAYA aakhri 30% (unseen)", te),
              fmt("PURANA score, aakhri 30%", old),
              "NAYA 3 hisso PF(trades): " + " / ".join(f"{f['pf']:.2f}({f['trades']})" for f in folds)]
    real = [f for f in folds if f["trades"] >= 5]
    good = (te["trades"] >= MIN_TRADES and te["pf"] >= 1.2 and len(real) == 3
            and all(f["pf"] > 1 for f in real))
    lines.append("Verdict: test aur teeno hisso me PF>1 - consistent dikh raha hai (phir bhi paper trade karo)."
                 if good else "Verdict: edge SABIT NAHI hua (trades kam ya hisson me PF<1). Is par paisa mat lagao.")
    if gap > 2:
        lines.append("Warning: data me missing candles zyada hain.")
    lines.append("Data: " + " | ".join(notes)[:200])
    lines.append("Leverage/margin call model nahi hai. Past result future ki guarantee nahi.")
    await ctx.send("\n".join(lines)[:1990])


@bot.command(name="capital")
async def capital_cmd(ctx, amount: float = None):
    if amount is None or amount <= 0:
        STATE["capital"] = None
        save_state()
        return await ctx.send("Capital hata diya. Use: `!capital 100000`")
    STATE["capital"] = amount
    save_state()
    await ctx.send(f"Capital {amount:g} set. Alerts me 1% risk wali quantity dikhegi.")


# ---------------- Scanner + Alerts ----------------
STATE_FILE = "bot_state.json"
SCAN_MINUTES = 15            # auto-scan har itne minute baad
MAX_ALERTS_PER_SCAN = 5
MAX_WATCH = 50
MAX_GAP_PCT = 5              # isse zyada missing candles wale symbols skip
SCAN_DELAY = 0.4             # Yahoo/exchange rate-limit se bachne ke liye
MIN_STRENGTH_ALERT = 60     # isse kam strength par alert nahi

DEFAULT_WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
    "GOLDBEES.NS", "GC=F",
    "EURUSD=X", "GBPUSD=X", "USDINR=X",
]
GOLD_SYMS = {"GOLDBEES.NS", "SILVERBEES.NS", "GC=F", "SI=F"}


def market_of(sym):
    if sym in GOLD_SYMS:
        return "gold"
    if is_crypto(sym):
        return "crypto"
    if sym.endswith("=X"):
        return "forex"
    if is_india(sym):
        return "india"
    return "other"


def load_state():
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
        st.setdefault("symbols", DEFAULT_WATCHLIST[:])
        st.setdefault("tf", "1h")
        st.setdefault("channel", None)
        st.setdefault("last", {})
        st.setdefault("capital", None)
        return st
    except Exception:
        return dict(symbols=DEFAULT_WATCHLIST[:], tf="1h", channel=None, last={}, capital=None)


STATE = load_state()


def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(STATE, f)
    except Exception as e:
        print("state save error:", e)


def scan_one(symbol, tf):
    try:
        df, notes, gap, age = prepare(symbol, tf, 600)
        ev = evaluate(symbol, tf, df, notes, gap, age)
    except Exception as e:
        return dict(symbol=symbol, error=str(e)[:80])
    ev["df"] = df
    return ev


def scan_all(symbols, tf):
    out = []
    for s in symbols:
        out.append(scan_one(s, tf))
        time.sleep(SCAN_DELAY)
    return out


ICON = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪"}


def pick_symbols(market):
    return [s for s in STATE["symbols"] if market == "all" or market_of(s) == market]


@bot.command(name="scan")
async def scan_cmd(ctx, tf: str = None, market: str = "all"):
    tf, market = tf or STATE["tf"], market.lower()
    syms = pick_symbols(market)
    if not syms:
        return await ctx.send("Watchlist khali hai ya market naam galat hai (all/crypto/india/gold/forex).")
    await ctx.send(f"{len(syms)} symbols scan ho rahe hain ({tf})... thoda time lagega.")
    res = await asyncio.to_thread(scan_all, syms, tf)
    ok = sorted([r for r in res if "error" not in r], key=lambda r: (r["signal"] != "WAIT", r["strength"]), reverse=True)
    bad = [r["symbol"] for r in res if "error" in r]
    lines = [f"**Scan {tf} ({market})** - {len(ok)} ok, {len(bad)} fail"]
    for r in ok[:25]:
        lines.append(f"{ICON[r['signal']]} {r['symbol']}: {r['signal']} {r['strength']}/100 [{r['fresh']}]")
    if bad:
        lines.append("Fail: " + ", ".join(bad[:8]) + " (timeframe/symbol check karo)")
    await ctx.send("\n".join(lines)[:1990])


def market_line(results):
    seen = {}
    for r in results:
        seen.setdefault(market_of(r["symbol"]), market_status(r["symbol"]))
    return " | ".join(f"{m}: {s}" for m, s in seen.items()) or "-"


@bot.command(name="top5", aliases=["voxa"])
async def top5_cmd(ctx, tf: str = None, market: str = "all"):
    tf, market = tf or STATE["tf"], market.lower()
    syms = pick_symbols(market)
    if not syms:
        return await ctx.send("Watchlist khali hai ya market naam galat hai (all/crypto/india/gold/forex).")
    await ctx.send(f"{len(syms)} symbols scan ho rahe hain ({tf})... thoda time lagega.")
    res = await asyncio.to_thread(scan_all, syms, tf)
    ok = sorted([r for r in res if "error" not in r], key=lambda r: (r["signal"] != "WAIT", r["strength"]), reverse=True)
    bad = [r["symbol"] for r in res if "error" in r]
    fr = Counter(r["fresh"] for r in ok)
    dq = "OK" if not fr.get("STALE") and not bad else "CHECK REQUIRED"
    e = discord.Embed(title="VOXA AI · MARKET INTELLIGENCE", color=discord.Color.blue(),
                      description=f"SYSTEM ONLINE · timeframe {tf} · {market}")
    e.add_field(name="Market status", value=market_line(ok), inline=False)
    e.add_field(name="Checks run", value=str(sum(r["n_checks"] for r in ok)), inline=True)
    e.add_field(name="Symbols", value=f"{len(ok)} ok / {len(bad)} fail", inline=True)
    e.add_field(name=f"Data quality: {dq}",
                value=f"LIVE {fr['LIVE']} · DELAYED {fr['DELAYED']} · STALE {fr['STALE']} · CLOSED {fr['CLOSED']}"
                      + ("\nPurana data hone par naya signal nahi dikhaya jata." if dq != "OK" else ""),
                inline=False)
    e.add_field(name="TOP 5 SETUPS", value="\u200b", inline=False)
    for i, r in enumerate(ok[:5], 1):
        e.add_field(name=f"{i:02d}. {r['symbol'].replace('.NS', '')} · {r['signal']}",
                    value=f"Strength **{r['strength']}/100** {bar(r['strength'])}\n"
                          f"{r['trend']} · Risk: {r['risk']} · {r['fresh']}\nReason: {r['reason'][:150]}",
                    inline=False)
    if bad:
        e.add_field(name="Fail", value=", ".join(bad[:8]), inline=False)
    e.set_footer(text="Rules-based analysis. Strength != probability. No guaranteed profit. Details: !analyze SYMBOL")
    await ctx.send(embed=e)


# ---------------- Paper trading journal ----------------
JOURNAL_FILE = "journal.json"


def load_journal():
    try:
        with open(JOURNAL_FILE) as f:
            return json.load(f)
    except Exception:
        return []


JOURNAL = load_journal()


def save_journal():
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(JOURNAL, f)
    except Exception as e:
        print("journal save error:", e)


def open_paper_trade(ev):
    d = 1 if ev["signal"] == "BUY" else -1
    c = COSTS[ev["kind"]]
    if d == -1 and c["short_extra"] is None:
        return None  # delivery me short nahi
    if any(t["status"] == "open" and t["symbol"] == ev["symbol"] and t["tf"] == ev["tf"] for t in JOURNAL):
        return None
    ef, xf = (c["buy"], c["sell"]) if d == 1 else (c["sell"], c["buy"])
    entry = ev["price"] * (1 + d * SLIP)
    atr = ev["atr"]
    t = dict(id=(JOURNAL[-1]["id"] + 1) if JOURNAL else 1, symbol=ev["symbol"], tf=ev["tf"], kind=ev["kind"],
             dir=d, entry=entry, sl=entry - d * SL_ATR * atr, tp=entry + d * TP_ATR * atr,
             risk=SL_ATR * atr + entry * (ef + xf + 2 * SLIP), open_time=ev["time"], status="open",
             strength=ev["strength"], market=market_of(ev["symbol"]))
    JOURNAL.append(t)
    save_journal()
    return t


def update_journal(cache):
    closed = []
    for t in JOURNAL:
        if t["status"] != "open":
            continue
        df = cache.get((t["symbol"], t["tf"]))
        if df is None:
            try:
                df = load(t["symbol"], t["tf"], 400)[0]
            except Exception:
                continue
        ot = pd.Timestamp(t["open_time"])
        later = df[df.index > ot]
        if later.empty:
            continue
        c, d = COSTS[t["kind"]], t["dir"]
        ef, xf = (c["buy"], c["sell"]) if d == 1 else (c["sell"], c["buy"])
        intraday = t["kind"].startswith("in_intraday")
        prev_close, ex, why, ets = t["entry"], None, None, None
        for ts, r in later.iterrows():
            if intraday and ts.date() > ot.date():
                ex, why, ets = prev_close, "EOD", ts
                break
            sl_hit = r["low"] <= t["sl"] if d == 1 else r["high"] >= t["sl"]
            tp_hit = r["high"] >= t["tp"] if d == 1 else r["low"] <= t["tp"]
            if sl_hit or tp_hit:
                if sl_hit:
                    ex = min(r["open"], t["sl"]) if d == 1 else max(r["open"], t["sl"])
                    why = "SL"
                else:
                    ex, why = t["tp"], "TP"
                ets = ts
                break
            prev_close = r["close"]
        if ex is None:
            continue
        exa = float(ex) * (1 - d * SLIP)
        held = (ets - ot).total_seconds()
        rate = c["hold"] + ((c["short_extra"] or 0.0) if d == -1 else 0.0)
        pnl = d * (exa - t["entry"]) - exa * xf - t["entry"] * ef - t["entry"] * rate * held / 86400
        t.update(status="closed", exit=exa, exit_time=str(ets), reason=why,
                 r=float(pnl / t["risk"]), pnl_pct=float(pnl / t["entry"] * 100))
        closed.append(t)
    if closed:
        save_journal()
    return closed


def report_embed():
    cl = [t for t in JOURNAL if t["status"] == "closed"]
    op = [t for t in JOURNAL if t["status"] == "open"]
    e = discord.Embed(title="VOXA AI · Paper Trading Report", color=discord.Color.blue())
    if not cl:
        e.description = f"Abhi koi closed trade nahi. Open trades: {len(op)}. Alerts ON karo, trades apne aap record honge."
        return e
    rs = [t["r"] for t in cl]
    gw, gl = sum(x for x in rs if x > 0), -sum(x for x in rs if x <= 0)
    pf = gw / gl if gl > 0 else float("inf")
    cum = np.cumsum(rs)
    dd = float((np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum).max())
    e.add_field(name="Trades", value=f"{len(cl)} closed · {len(op)} open", inline=True)
    e.add_field(name="Win rate", value=f"{sum(1 for x in rs if x > 0) / len(rs) * 100:.0f}%", inline=True)
    e.add_field(name="Avg R / PF", value=f"{np.mean(rs):+.2f}R / {pf:.2f}", inline=True)
    e.add_field(name="Return (1% risk/trade)", value=f"{sum(rs) * RISK_PER_TRADE * 100:+.1f}%", inline=True)
    e.add_field(name="Max drawdown", value=f"{dd * RISK_PER_TRADE * 100:.1f}%", inline=True)
    by = {}
    for t in cl:
        by.setdefault(t["market"], []).append(t["r"])
    e.add_field(name="Market-wise", value="\n".join(f"{m}: {len(v)} trades, avg {np.mean(v):+.2f}R"
                                                    for m, v in by.items()), inline=False)
    if len(cl) < MIN_TRADES:
        e.add_field(name="Warning", value=f"Sirf {len(cl)} trades (<{MIN_TRADES}). Result abhi bharosemand nahi.",
                    inline=False)
    e.set_footer(text="Costs, slippage, SL/TP ke saath virtual trades. Asli paisa nahi.")
    return e


@bot.command(name="report")
async def report_cmd(ctx):
    await ctx.send(embed=report_embed())


@bot.command(name="journal")
async def journal_cmd(ctx, n: int = 10):
    if not JOURNAL:
        return await ctx.send("Journal khali hai. `!alerts on 1h` chalu karo.")
    lines = []
    for t in JOURNAL[-max(1, min(n, 20)):][::-1]:
        side = "LONG" if t["dir"] == 1 else "SHORT"
        res = f"{t['reason']} {t['r']:+.2f}R" if t["status"] == "closed" else "OPEN"
        lines.append(f"#{t['id']} {t['symbol']} {t['tf']} {side} @ {t['entry']:.4g} -> {res}")
    await ctx.send("\n".join(lines)[:1990])


@bot.command(name="clearjournal")
async def clearjournal_cmd(ctx, confirm: str = ""):
    if confirm.lower() != "yes":
        return await ctx.send("Poora journal hata dena hai? `!clearjournal yes` likho.")
    JOURNAL.clear()
    save_journal()
    await ctx.send("Journal saaf.")


@bot.group(name="watch", invoke_without_command=True)
async def watch(ctx):
    await ctx.send("Use: `!watch add SYM...` | `!watch remove SYM...` | `!watch list` | `!watch reset`")


@watch.command(name="add")
async def watch_add(ctx, *symbols):
    added = []
    for s in symbols:
        n = normalize(s)
        if n in STATE["symbols"]:
            continue
        if len(STATE["symbols"]) >= MAX_WATCH:
            await ctx.send(f"Limit {MAX_WATCH} symbols ki hai.")
            break
        STATE["symbols"].append(n)
        added.append(n)
    save_state()
    await ctx.send(f"Added: {', '.join(added) or 'kuch nahi'} | Total: {len(STATE['symbols'])}")


@watch.command(name="remove")
async def watch_remove(ctx, *symbols):
    gone = []
    for s in symbols:
        n = normalize(s)
        if n in STATE["symbols"]:
            STATE["symbols"].remove(n)
            gone.append(n)
    save_state()
    await ctx.send(f"Removed: {', '.join(gone) or 'kuch nahi'} | Total: {len(STATE['symbols'])}")


@watch.command(name="list")
async def watch_list(ctx):
    groups = {}
    for s in STATE["symbols"]:
        groups.setdefault(market_of(s), []).append(s)
    lines = [f"**Watchlist ({len(STATE['symbols'])})**"]
    lines += [f"{m}: {', '.join(v)}" for m, v in groups.items()]
    await ctx.send("\n".join(lines)[:1990])


@watch.command(name="reset")
async def watch_reset(ctx):
    STATE["symbols"] = DEFAULT_WATCHLIST[:]
    save_state()
    await ctx.send(f"Default watchlist ({len(STATE['symbols'])} symbols) wapas set.")


@bot.group(name="alerts", invoke_without_command=True)
async def alerts(ctx):
    await ctx.send("Use: `!alerts on [tf]` | `!alerts off` | `!alerts status`")


@alerts.command(name="on")
async def alerts_on(ctx, tf: str = "1h"):
    STATE["channel"], STATE["tf"] = ctx.channel.id, tf
    save_state()
    await ctx.send(f"Alerts ON: is channel me, timeframe {tf}, har {SCAN_MINUTES} min scan. "
                   f"Sirf naye band candle par, trend+pullback setup, max {MAX_ALERTS_PER_SCAN} alerts/scan. "
                   "Phone me is channel ke notifications ON rakhna. Bot tabhi chalega jab ye script chal rahi ho.")


@alerts.command(name="off")
async def alerts_off(ctx):
    STATE["channel"] = None
    save_state()
    await ctx.send("Alerts OFF.")


@alerts.command(name="status")
async def alerts_status(ctx):
    on = STATE["channel"] is not None
    await ctx.send(f"Alerts: {'ON' if on else 'OFF'} | tf {STATE['tf']} | symbols {len(STATE['symbols'])} | "
                   f"strategy trend+pullback | min strength {MIN_STRENGTH_ALERT}")


@tasks.loop(minutes=SCAN_MINUTES)
async def auto_scan():
    ch_id = STATE.get("channel")
    ch = bot.get_channel(ch_id) if ch_id else None
    if ch is None:
        return
    tf = STATE["tf"]
    res = await asyncio.to_thread(scan_all, list(STATE["symbols"]), tf)
    cache = {(r["symbol"], tf): r["df"] for r in res if "error" not in r}
    for t in await asyncio.to_thread(update_journal, cache):
        await ch.send(f"Paper trade band: #{t['id']} {t['symbol']} {t['tf']} {'LONG' if t['dir'] == 1 else 'SHORT'} "
                      f"-> {t['reason']} ({t['r']:+.2f}R)")
    cands = []
    for r in res:
        if "error" in r or r["gap"] > MAX_GAP_PCT or r["signal"] == "WAIT":
            continue
        if r["fresh"] != "LIVE" or r["strength"] < MIN_STRENGTH_ALERT:
            continue
        prev = STATE["last"].get(f"{r['symbol']}|{tf}")
        if prev:
            pt, ps = prev.rsplit("|", 1)
            if ps == r["signal"] and (pd.Timestamp(r["time"]) - pd.Timestamp(pt)).total_seconds() < 3 * tfsec(tf):
                continue  # same direction ka alert abhi hi ja chuka
        cands.append(r)
    cands.sort(key=lambda r: r["strength"], reverse=True)
    for r in cands[:MAX_ALERTS_PER_SCAN]:
        emb = analysis_embed(r)
        emb.title = "🔔 " + emb.title
        chart = await asyncio.to_thread(make_chart, r["df"], r["symbol"], tf, chart_levels(r, r["df"]))
        emb.set_image(url="attachment://chart.png")
        opened = await asyncio.to_thread(open_paper_trade, r)
        await ch.send(embed=emb, file=discord.File(chart, "chart.png"))
        if opened:
            await ch.send(f"Paper trade #{opened['id']} khula (virtual). `!report` se record dekho.")
        STATE["last"][f"{r['symbol']}|{tf}"] = f"{r['time']}|{r['signal']}"
    save_state()


@auto_scan.before_loop
async def _before_auto_scan():
    await bot.wait_until_ready()


if __name__ == "__main__":
    bot.run(os.environ["DISCORD_TOKEN"])
