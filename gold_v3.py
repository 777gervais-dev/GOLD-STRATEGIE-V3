#!/usr/bin/env python3
"""
GOLD ML v3 — XAU/USD Dashboard
python gold_v3.py → http://localhost:8083
"""
import time, math, random, threading, traceback, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, Response

app  = Flask(__name__)
PORT = 8083
CACHE = {}
READY = False
PRIME = 0.0039  # prime futures GC=F → spot XAU/USD

# ══════════════════════════════════════════════
#  DONNÉES
# ══════════════════════════════════════════════
def _fetch(sym, interval="1m", period="2d"):
    import requests as r
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36",
         "Accept": "application/json", "Cache-Control": "no-cache"}
    for host in ["query1", "query2"]:
        try:
            url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
                   f"?interval={interval}&range={period}&_={int(time.time())}")
            res = r.get(url, headers=h, timeout=12).json()["chart"]["result"][0]
            return res
        except: pass
    return None

def get_prix():
    res = _fetch("GC=F", "1m", "2d")
    if not res: return None, 0.0
    try:
        cls = [float(x) for x in res["indicators"]["quote"][0]["close"] if x]
        if not cls: return None, 0.0
        med = sorted(cls)[len(cls)//2]
        pf  = cls[-1]
        if abs(pf - med)/med > 0.06: pf = med
        p   = round(pf * (1-PRIME), 2)
        pv  = round(float(res["meta"].get("previousClose") or pf) * (1-PRIME), 2)
        v   = round((p-pv)/pv*100, 2) if pv else 0.0
        if 1500 < p < 15000:
            print(f"[PRIX] ✅ ${p:.2f}")
            return p, v
    except Exception as e:
        print(f"[PRIX] ❌ {e}")
    return None, 0.0

def get_ohlcv(interval="15m", period="5d"):
    res = _fetch("GC=F", interval, period)
    if not res: return None, False
    try:
        tss = res["timestamp"]
        q   = res["indicators"]["quote"][0]
        rows = []
        for i, tv in enumerate(tss):
            c = q["close"][i]
            if c is None: continue
            o  = float(q["open"][i]  or c)
            hi = float(q["high"][i]  or c)
            lo = float(q["low"][i]   or c)
            c  = float(c)
            hi = max(hi,o,c); lo = min(lo,o,c)
            dt = datetime.fromtimestamp(tv, tz=timezone.utc)
            rows.append({"ts": dt,
                         "date": dt.strftime("%d/%m %H:%M"),
                         "open":  round(o*(1-PRIME),2),
                         "high":  round(hi*(1-PRIME),2),
                         "low":   round(lo*(1-PRIME),2),
                         "close": round(c*(1-PRIME),2),
                         "vol":   int(q["volume"][i] or 0)})
        if len(rows) < 30: return None, False
        df = pd.DataFrame(rows)
        print(f"[OHLCV {interval}] ✅ {len(df)} bougies")
        return df, True
    except Exception as e:
        print(f"[OHLCV] ❌ {e}"); return None, False

def synth(p=3200, mn=15, n=200):
    random.seed(int(time.time()//(mn*60)))
    now = datetime.now(timezone.utc)
    t0  = now.replace(minute=(now.minute//mn)*mn,second=0,microsecond=0)
    t0 -= timedelta(minutes=mn*n)
    rows, cur = [], p
    for i in range(n):
        dt = t0 + timedelta(minutes=mn*i)
        vf = 2.0 if 7<=dt.hour<10 or 12<=dt.hour<15 else 0.7
        c  = round(cur*(1+random.gauss(0,0.0006*vf)),2)
        o  = round(c*(1+random.gauss(0,0.0002)),2)
        sp = abs(c-o)*random.uniform(1.2,2.5)
        hi = round(max(o,c)+sp*random.uniform(.2,.8),2)
        lo = round(min(o,c)-sp*random.uniform(.2,.8),2)
        rows.append({"ts":dt,"date":dt.strftime("%d/%m %H:%M"),
                     "open":o,"high":hi,"low":lo,"close":c,
                     "vol":int(abs(random.gauss(800,300)))})
        cur = c
    rows[-1]["close"] = p
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════
#  INDICATEURS
# ══════════════════════════════════════════════
def calc_ind(df):
    c=df["close"]; h=df["high"]; l=df["low"]; v=df["vol"]; o=df["open"]
    n = len(df)
    df["ema9"]   = c.ewm(span=9,  adjust=False).mean()
    df["ema21"]  = c.ewm(span=21, adjust=False).mean()
    df["mm20"]   = c.rolling(20).mean()
    df["mm50"]   = c.rolling(50).mean()
    df["mm200"]  = c.ewm(span=min(200,n), adjust=False).mean()
    d2 = c.diff()
    up = d2.clip(lower=0).rolling(14).mean()
    dn = (-d2.clip(upper=0)).rolling(14).mean()
    df["rsi"]    = 100 - 100/(1 + up/(dn+1e-9))
    e12=c.ewm(span=12,adjust=False).mean()
    e26=c.ewm(span=26,adjust=False).mean()
    df["macd"]   = e12-e26
    df["msig"]   = df["macd"].ewm(span=9,adjust=False).mean()
    df["mhist"]  = df["macd"]-df["msig"]
    lo14=l.rolling(14).min(); hi14=h.rolling(14).max()
    df["stk"]    = 100*(c-lo14)/(hi14-lo14+1e-9)
    df["std"]    = df["stk"].rolling(3).mean()
    df["bb_m"]   = c.rolling(20).mean()
    bs           = c.rolling(20).std()
    df["bb_u"]   = df["bb_m"]+2*bs
    df["bb_d"]   = df["bb_m"]-2*bs
    df["bb_p"]   = (c-df["bb_d"])/(df["bb_u"]-df["bb_d"]+1e-9)
    tr = pd.Series(
        [max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1]))
         for i in range(1,n)], index=df.index[1:])
    df["atr"]    = tr.reindex(df.index).rolling(14).mean()
    df["atr50"]  = tr.reindex(df.index).rolling(50).mean()
    df["vol_ma"] = v.rolling(20).mean()
    df["vol_r"]  = v/(df["vol_ma"]+1e-9)
    df["mom5"]   = c.pct_change(5)
    df["mom10"]  = c.pct_change(10)
    df["body"]   = (c-o).abs()/(df["atr"]+1e-9)
    df["bull_c"] = (c>o).astype(float)
    df["mm20_sl"]= df["mm20"].diff(3)
    df["mm50_sl"]= df["mm50"].diff(5)
    return df

def last_ind(df):
    r = df.iloc[-1]
    def s(k, d=2): return round(float(r[k]),d) if pd.notna(r.get(k,None)) else 0.0
    p   = float(r["close"])
    atr = s("atr")
    return {
        "prix":round(p,2), "atr":atr, "rsi":s("rsi",1),
        "stk":s("stk",1), "std":s("std",1),
        "macd":s("macd"), "mhist":s("mhist"), "macd_bull":bool(r["mhist"]>0),
        "mm20":s("mm20"), "mm50":s("mm50"), "mm200":s("mm200"),
        "ema9":s("ema9"), "ema21":s("ema21"),
        "bb_p":s("bb_p",3), "bb_u":s("bb_u"), "bb_d":s("bb_d"),
        "vol_r":s("vol_r",2), "vol_ok":bool(r["vol_r"]>1.2),
        "atr50":s("atr50"),
        "vol_lbl":("FORTE🔥" if atr>s("atr50")*1.3 else "FAIBLE😴" if atr<s("atr50")*0.7 else "NORMALE✓"),
        "rsi_l":  [round(float(v),1) for v in df["rsi"].tail(60)  if pd.notna(v)],
        "mhist_l":[round(float(v),3) for v in df["mhist"].tail(60) if pd.notna(v)],
        "vol_l":  [int(v) for v in df["vol"].tail(60)],
        "close_l":[round(float(v),2) for v in df["close"].tail(60) if pd.notna(v)],
        "mm20_l": [round(float(v),2) for v in df["mm20"].tail(60)  if pd.notna(v)],
        "bb_u_l": [round(float(v),2) for v in df["bb_u"].tail(60)  if pd.notna(v)],
        "bb_d_l": [round(float(v),2) for v in df["bb_d"].tail(60)  if pd.notna(v)],
    }

# ══════════════════════════════════════════════
#  ML — 3 MODÈLES ENSEMBLE
# ══════════════════════════════════════════════
class Stump:
    def __init__(self): self.j=self.t=0; self.lv=self.rv=0.5
    def fit(self, X, r):
        best=1e18
        for j in range(X.shape[1]):
            vals=np.unique(X[:,j])
            thrs=(vals[:-1]+vals[1:])/2 if len(vals)>1 else vals
            for t in thrs[:30]:
                lm=r[X[:,j]<=t].mean() if (X[:,j]<=t).any() else 0
                rm=r[X[:,j]>t].mean()  if (X[:,j]>t).any()  else 0
                e=((r-np.where(X[:,j]<=t,lm,rm))**2).sum()
                if e<best: best=e;self.j=j;self.t=t;self.lv=lm;self.rv=rm
    def pred(self, X):
        return np.where(np.asarray(X)[:,self.j]<=self.t, self.lv, self.rv)

class GBM:
    def __init__(self, n=100, lr=0.08):
        self.n=n; self.lr=lr; self.stumps=[]; self.base=0.5; self.mu=self.sd=None
    def _sig(self, x): return 1/(1+np.exp(-np.clip(x,-12,12)))
    def fit(self, X, y):
        X=np.array(X,float); y=np.array(y,float)
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9
        Xn=(X-self.mu)/self.sd; self.base=y.mean(); F=np.full(len(y),self.base)
        for _ in range(self.n):
            g=y-self._sig(F); s=Stump(); s.fit(Xn,g)
            F+=self.lr*(s.pred(Xn)-.5)*2; self.stumps.append(s)
    def proba(self, x):
        xn=(np.array(x,float)-self.mu)/self.sd; F=self.base
        for s in self.stumps: F+=self.lr*(s.pred([xn])-.5)*2
        return float(np.clip(self._sig(np.atleast_1d(F))[0],0.01,0.99))

class LR:
    def __init__(self, lr=0.05, n=400):
        self.lr=lr; self.n=n; self.w=None; self.b=0.; self.mu=self.sd=None
    def _sig(self, x): return 1/(1+np.exp(-np.clip(x,-12,12)))
    def fit(self, X, y):
        X=np.array(X,float); y=np.array(y,float)
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9
        Xn=(X-self.mu)/self.sd; self.w=np.zeros(Xn.shape[1])
        for _ in range(self.n):
            e=self._sig(Xn@self.w+self.b)-y
            self.w-=self.lr*(Xn.T@e/len(y)+0.01*self.w); self.b-=self.lr*e.mean()
    def proba(self, x):
        xn=(np.array(x,float)-self.mu)/self.sd
        return float(np.clip(self._sig(float(np.dot(xn,self.w)+self.b)),0.01,0.99))

class RF:
    def __init__(self, n=40):
        self.n=n; self.stumps=[]; self.mu=self.sd=None
    def fit(self, X, y):
        X=np.array(X,float); y=np.array(y,float)
        self.mu=X.mean(0); self.sd=X.std(0)+1e-9
        Xn=(X-self.mu)/self.sd
        for _ in range(self.n):
            idx=np.random.choice(len(X),len(X),replace=True)
            Xb=Xn[idx]; yb=y[idx]
            fi=np.random.choice(Xn.shape[1],max(4,Xn.shape[1]//2),replace=False)
            Xs=np.zeros_like(Xb); Xs[:,fi]=Xb[:,fi]
            s=Stump(); s.fit(Xs,yb); self.stumps.append(s)
    def proba(self, x):
        xn=(np.array(x,float)-self.mu)/self.sd
        return float(np.clip(np.mean([s.pred([xn])[0] for s in self.stumps]),0.01,0.99))

_MDL = {}

def build_feats(df):
    rows=[]
    for i in range(60, len(df)):
        r=df.iloc[i]; p=float(r["close"]); a=float(r["atr"])+1e-9
        def g(k): return float(r[k]) if pd.notna(r.get(k)) else 0.
        rows.append([
            g("rsi")/100, g("stk")/100, g("std")/100, g("bb_p"),
            g("mhist")/a, g("macd")/a,
            (p-g("mm20"))/a, (p-g("mm50"))/a, (p-g("mm200"))/a,
            (g("ema9")-g("ema21"))/a,
            min(g("mom5")*100,5), min(g("mom10")*100,5),
            min(g("vol_r"),5), g("atr")/(p+1e-9)*100,
            g("body"), g("bull_c"), g("mm20_sl")/a, g("mm50_sl")/a,
            math.sin(2*math.pi*df.iloc[i]["ts"].hour/24),
            math.cos(2*math.pi*df.iloc[i]["ts"].hour/24),
        ])
    return rows

def train_ml(df):
    global _MDL
    X = build_feats(df)
    cl = df["close"].tolist()
    y  = [1 if cl[min(i+4,len(cl)-1)]>cl[i] else 0 for i in range(60,len(df))]
    m  = min(len(X),len(y)); X,y=X[:m],y[:m]
    if m < 80: return False
    sp = int(m*0.8)
    Xtr,Xte,ytr,yte = X[:sp],X[sp:],y[:sp],y[sp:]
    Xn=np.array(Xtr); Xt=np.array(Xte)
    print("[ML] GBM..."); gb=GBM(n=100,lr=0.08); gb.fit(Xn,np.array(ytr))
    gb_acc=round(np.mean([(gb.proba(x)>0.5)==yi for x,yi in zip(Xte,yte)])*100,1)
    print("[ML] LR..."); lr=LR(lr=0.05,n=400); lr.fit(Xn,np.array(ytr))
    lr_acc=round(np.mean([(lr.proba(x)>0.5)==yi for x,yi in zip(Xte,yte)])*100,1)
    print("[ML] RF..."); rf=RF(n=40); rf.fit(Xn,np.array(ytr))
    rf_acc=round(np.mean([(rf.proba(x)>0.5)==yi for x,yi in zip(Xte,yte)])*100,1)
    _MDL={"gb":gb,"gb_acc":gb_acc,"lr":lr,"lr_acc":lr_acc,
          "rf":rf,"rf_acc":rf_acc,"n":len(Xtr)}
    print(f"[ML] ✅ GB:{gb_acc}% LR:{lr_acc}% RF:{rf_acc}%")
    return True

def predict_ml(df, bu=0, be=0):
    if not _MDL: return None
    X = build_feats(df)
    if not X: return None
    x = X[-1]
    gp=_MDL["gb"].proba(x); lp=_MDL["lr"].proba(x); rp=_MDL["rf"].proba(x)
    raw = gp*0.45 + lp*0.30 + rp*0.25
    bias= (bu-be)*0.06
    bull= max(0.01,min(0.99, raw+bias)); bear=1-bull
    sig = "BULL" if bull>=0.58 else "BEAR" if bear>=0.58 else "NEUTRE"
    conf= "FORTE" if max(bull,bear)>=0.72 else "MODÉRÉE" if max(bull,bear)>=0.58 else "FAIBLE"
    return {"signal":sig,"conf":conf,
            "bull":round(bull*100,1),"bear":round(bear*100,1),
            "bull_raw":round(raw*100,1),"bias":round(bias*100,1),
            "gb":round(gp*100,1),"gb_acc":_MDL["gb_acc"],
            "lr":round(lp*100,1),"lr_acc":_MDL["lr_acc"],
            "rf":round(rp*100,1),"rf_acc":_MDL["rf_acc"],
            "n":_MDL["n"]}

# ══════════════════════════════════════════════
#  PATTERNS
# ══════════════════════════════════════════════
def get_patterns(df):
    if len(df)<3: return []
    r=[]
    o1,h1,l1,c1=df["open"].iloc[-1],df["high"].iloc[-1],df["low"].iloc[-1],df["close"].iloc[-1]
    o2,_2,_3,c2=df["open"].iloc[-2],df["high"].iloc[-2],df["low"].iloc[-2],df["close"].iloc[-2]
    bd=abs(c1-o1); rng=h1-l1
    wu=h1-max(o1,c1); wd=min(o1,c1)-l1
    atr=float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else rng+1e-9
    if bd<rng*0.1 and rng>atr*0.3 and wu>bd*2 and wd>bd*2:
        r.append({"n":"DOJI","ic":"🔄","s":"NEUT","d":"Indécision marché"})
    if wd>bd*2 and wu<bd*0.5 and c2<o2:
        r.append({"n":"HAMMER","ic":"🔨","s":"BULL","d":"Renversement haussier"})
    if wu>bd*2 and wd<bd*0.5 and c2>o2:
        r.append({"n":"SHOOTING STAR","ic":"💫","s":"BEAR","d":"Renversement baissier"})
    if c2<o2 and c1>o1 and o1<=c2 and c1>=o2 and bd>abs(c2-o2)*1.1:
        r.append({"n":"ENGULFING BULL","ic":"🟢","s":"BULL","d":"Avalement haussier"})
    if c2>o2 and c1<o1 and o1>=c2 and c1<=o2 and bd>abs(c2-o2)*1.1:
        r.append({"n":"ENGULFING BEAR","ic":"🔴","s":"BEAR","d":"Avalement baissier"})
    if wd>rng*0.6 and bd<rng*0.3:
        r.append({"n":"PIN BAR BULL","ic":"📍","s":"BULL","d":"Rejet du bas"})
    if wu>rng*0.6 and bd<rng*0.3:
        r.append({"n":"PIN BAR BEAR","ic":"📌","s":"BEAR","d":"Rejet du haut"})
    return r

# ══════════════════════════════════════════════
#  KILL ZONES
# ══════════════════════════════════════════════
_KZ=[(0,3,"ASIE","🌏"),(7,10,"LONDON","🇬🇧"),(12,15,"NEW YORK","🇺🇸"),(19,21,"NY CLOSE","🔔")]
def get_kz():
    now=datetime.now(timezone.utc); h=now.hour; act=None
    for s,e,nm,ic in _KZ:
        if s<=h<e:
            end=now.replace(hour=e,minute=0,second=0,microsecond=0)
            act={"name":nm,"ic":ic,"start":f"{s:02d}:00","end":f"{e:02d}:00",
                 "rem":int((end-now).total_seconds()/60)}
    nxt=[]
    for s,e,nm,ic in _KZ:
        nx=now.replace(hour=s,minute=0,second=0,microsecond=0)
        if nx<=now: nx+=timedelta(days=1)
        dm=int((nx-now).total_seconds()/60)
        nxt.append({"name":nm,"ic":ic,"start":f"{s:02d}:00","end":f"{e:02d}:00",
                    "dans":f"{dm//60}h{dm%60:02d}m"})
    return {"active":act,"next":nxt}

# ══════════════════════════════════════════════
#  SCORE ICT
# ══════════════════════════════════════════════
def score_ict(p, ind, ml, pats, kz, trend_cls):
    sc=0; det=[]
    if trend_cls=="bull":   sc+=20; det.append(("H4 HAUSSIÈRE",+20))
    elif trend_cls=="bear": sc-=20; det.append(("H4 BAISSIÈRE",-20))
    if ml:
        if ml["signal"]=="BULL":
            v=25 if ml["conf"]=="FORTE" else 15; sc+=v; det.append((f"ML BULL {ml['bull']}%",+v))
        elif ml["signal"]=="BEAR":
            v=25 if ml["conf"]=="FORTE" else 15; sc-=v; det.append((f"ML BEAR {ml['bear']}%",-v))
    if ind["macd_bull"]: sc+=10; det.append(("MACD ▲",+10))
    else:                sc-=10; det.append(("MACD ▼",-10))
    rsi=ind["rsi"]
    if 50<rsi<70:   sc+=10; det.append((f"RSI {rsi:.0f} zone bull",+10))
    elif 30<rsi<=50:sc-=10; det.append((f"RSI {rsi:.0f} zone bear",-10))
    elif rsi>=70:   sc-=5;  det.append((f"RSI {rsi:.0f} suracheté",-5))
    elif rsi<=30:   sc+=5;  det.append((f"RSI {rsi:.0f} survendu",+5))
    if p>ind["mm200"]: sc+=10; det.append(("Prix>MM200",+10))
    else:              sc-=10; det.append(("Prix<MM200",-10))
    if ind["vol_ok"]:  sc+=5;  det.append((f"Vol ×{ind['vol_r']}",+5))
    bp =[x for x in pats if x["s"]=="BULL"]
    brp=[x for x in pats if x["s"]=="BEAR"]
    if bp:  sc+=5; det.append((f"Pattern {bp[0]['n']}",+5))
    elif brp:sc-=5;det.append((f"Pattern {brp[0]['n']}",-5))
    if kz["active"]: sc+=5; det.append((f"KZ {kz['active']['name']}",+5))
    sc=max(-100,min(100,sc)); ab=abs(sc)
    sig=("BULL FORT" if sc>=70 else "BULL" if sc>=50 else "LÉGÈREMENT BULL" if sc>=25
         else "BEAR FORT" if sc<=-70 else "BEAR" if sc<=-50 else "LÉGÈREMENT BEAR" if sc<=-25
         else "NEUTRE")
    return {"score":round(sc,1),"abs":round(ab,1),"signal":sig,
            "dir":"bull" if sc>10 else "bear" if sc<-10 else "neut","det":det}

# ══════════════════════════════════════════════
#  MTF
# ══════════════════════════════════════════════
def get_mtf(p):
    res={}
    for tf,iv,pd2 in [("M15","15m","5d"),("H1","1h","10d"),("H4","4h","60d")]:
        df,real=get_ohlcv(iv,pd2)
        if df is None or len(df)<20:
            res[tf]={"trend":"N/A","cls":"neut","rsi":50,"macd_bull":False,"real":False}
            continue
        df2=calc_ind(df); last=df2.iloc[-1]
        mm20 =float(last["mm20"])  if pd.notna(last["mm20"])  else p
        mm200=float(last["mm200"]) if pd.notna(last["mm200"]) else p
        rsi  =float(last["rsi"])   if pd.notna(last["rsi"])   else 50
        mhist=float(last["mhist"]) if pd.notna(last["mhist"]) else 0
        sl=(float(df2["mm20"].iloc[-1])-float(df2["mm20"].iloc[max(-6,-len(df2))])) if len(df2)>5 else 0
        if   p>mm200 and mm20>mm200 and sl>0: trend="HAUSSIÈRE ▲"; cls="bull"
        elif p<mm200 and mm20<mm200 and sl<0: trend="BAISSIÈRE ▼"; cls="bear"
        elif p>mm200: trend="HAUSSIÈRE ▲"; cls="bull"
        elif p<mm200: trend="BAISSIÈRE ▼"; cls="bear"
        else:         trend="NEUTRE ➡";    cls="neut"
        res[tf]={"trend":trend,"cls":cls,"rsi":round(rsi,1),"macd_bull":mhist>0,"real":real}
    trs=[res[t]["cls"] for t in ["M15","H1","H4"]]
    bu=trs.count("bull"); be=trs.count("bear")
    if   bu==3: conf="BULL TOTAL ✅✅✅"
    elif bu==2: conf="BULL MAJORITAIRE ✅✅"
    elif be==3: conf="BEAR TOTAL ✅✅✅"
    elif be==2: conf="BEAR MAJORITAIRE ✅✅"
    else:       conf="MIXTE ⚠️"
    res["conf"]=conf; res["bulls"]=bu; res["bears"]=be
    return res

# ══════════════════════════════════════════════
#  POSITION — SL = ATR × 0.4 (serré)
# ══════════════════════════════════════════════
def get_position(p, atr, direction, capital=10000, risk=1.0):
    rd   = round(capital*risk/100, 2)
    sl_d = max(round(atr*0.4, 2), 1.5)
    if direction=="bull":
        sl=round(p-sl_d,2); tp1=round(p+sl_d*1.5,2); tp2=round(p+sl_d*2.5,2); tp3=round(p+sl_d*4.0,2)
    else:
        sl=round(p+sl_d,2); tp1=round(p-sl_d*1.5,2); tp2=round(p-sl_d*2.5,2); tp3=round(p-sl_d*4.0,2)
    lots=round(rd/sl_d,4) if sl_d>0 else 0
    return {"dir":direction,"capital":capital,"risk":risk,"risk_usd":rd,
            "entry":p,"sl":sl,"sl_d":sl_d,"tp1":tp1,"tp2":tp2,"tp3":tp3,"lots":lots}

# ══════════════════════════════════════════════
#  CALCUL PRINCIPAL
# ══════════════════════════════════════════════
def compute():
    global CACHE, READY
    try:
        print("[BG] Démarrage...")
        df_raw,real=get_ohlcv("15m","5d")
        if df_raw is None: df_raw=synth(); real=False
        p,var=get_prix()
        if not p:
            p=round(float(df_raw["close"].iloc[-1]),2); var=0.0
        df_raw.iloc[-1,df_raw.columns.get_loc("close")]=p
        df=calc_ind(df_raw)
        print(f"[BG] Prix=${p} · Indicateurs OK")
        train_ml(df)
        ml=predict_ml(df)
        pats=get_patterns(df); kzone=get_kz()
        print("[BG] MTF...")
        mtf=get_mtf(p); trend_cls=mtf.get("H4",{}).get("cls","neut")
        ml=predict_ml(df,mtf.get("bulls",0),mtf.get("bears",0))
        ind=last_ind(df)
        ict=score_ict(p,ind,ml,pats,kzone,trend_cls)
        pos=get_position(p,ind["atr"],ict["dir"])
        ohlc=df_raw[["date","open","high","low","close","vol"]].tail(80).to_dict("records")
        pct24=round((p-float(df_raw["close"].iloc[0]))/float(df_raw["close"].iloc[0])*100,2)
        CACHE={"prix":p,"var":round(var,2),"pct24":pct24,
               "hi":round(max(r["high"] for r in ohlc),2),
               "lo":round(min(r["low"]  for r in ohlc),2),
               "heure":datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
               "utc":datetime.now(timezone.utc).strftime("%H:%M UTC"),
               "trend":mtf.get("H4",{}).get("trend","?"),
               "trend_cls":trend_cls,
               "ind":ind,"ml":ml,"ict":ict,"mtf":mtf,
               "pats":pats,"pos":pos,"kz":kzone,"ohlc":ohlc,
               "src":"RÉEL✅" if real else "SYNTH⚠️"}
        READY=True
        print("[BG] ✅ READY")
    except Exception as e:
        print(f"[BG] ❌ {e}"); traceback.print_exc()

def prix_loop():
    while True:
        time.sleep(10)
        try:
            if not CACHE: continue
            p,v=get_prix()
            if p:
                CACHE["prix"]=p; CACHE["var"]=v
                CACHE["utc"]=datetime.now(timezone.utc).strftime("%H:%M UTC")
        except: pass

def mtf_loop():
    time.sleep(120)
    while True:
        try:
            if READY and CACHE:
                p=CACHE.get("prix"); print("[MTF] ♻️ Recalcul...")
                mtf=get_mtf(p); CACHE["mtf"]=mtf
                df_raw,_=get_ohlcv("15m","5d")
                if df_raw is not None:
                    df_raw.iloc[-1,df_raw.columns.get_loc("close")]=p
                    df=calc_ind(df_raw); ind=last_ind(df)
                    ml=predict_ml(df,mtf.get("bulls",0),mtf.get("bears",0))
                    ict=score_ict(p,ind,ml,CACHE.get("pats",[]),
                                  CACHE.get("kz",{"active":None,"next":[]}),
                                  mtf.get("H4",{}).get("cls","neut"))
                    CACHE.update({"mtf":mtf,"ml":ml,"ict":ict,"ind":ind,
                                  "trend":mtf.get("H4",{}).get("trend","?"),
                                  "trend_cls":mtf.get("H4",{}).get("cls","neut"),
                                  "pos":get_position(p,ind["atr"],ict["dir"]),
                                  "heure":datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
                print(f"[MTF] ✅ {mtf.get('conf')}")
        except Exception as e:
            print(f"[MTF] ❌ {e}")
        time.sleep(300)

# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════
@app.route("/")
def index(): return Response(PAGE, mimetype="text/html; charset=utf-8")

@app.route("/api/ready")
def api_ready(): return jsonify({"ready": READY})

@app.route("/api/data")
def api_data():
    if not READY: return jsonify({"ready": False})
    return jsonify(CACHE)

@app.route("/api/prix")
def api_prix():
    if not CACHE: return jsonify({"ok":False})
    return jsonify({"ok":True,"prix":CACHE.get("prix"),
                    "var":CACHE.get("var",0),"utc":CACHE.get("utc","")})

@app.route("/api/candle")
def api_candle():
    if not CACHE or not CACHE.get("ohlc"): return jsonify({"ok":False})
    p=CACHE.get("prix"); last=dict(CACHE["ohlc"][-1])
    if p: last["close"]=p; last["high"]=max(last["high"],p); last["low"]=min(last["low"],p)
    now=datetime.now(timezone.utc)
    return jsonify({"ok":True,"prix":p,"var":CACHE.get("var",0),
                    "utc":now.strftime("%H:%M UTC"),"candle":last})

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    if not READY: return jsonify({"ok":False})
    def _do():
        try:
            df_raw,_=get_ohlcv("15m","5d"); p=CACHE.get("prix",3200)
            if df_raw is None: return
            df_raw.iloc[-1,df_raw.columns.get_loc("close")]=p
            df=calc_ind(df_raw); train_ml(df)
            mtf=CACHE.get("mtf",{}); ind=last_ind(df)
            ml=predict_ml(df,mtf.get("bulls",0),mtf.get("bears",0))
            ict=score_ict(p,ind,ml,CACHE.get("pats",[]),
                          CACHE.get("kz",{"active":None,"next":[]}),
                          CACHE.get("trend_cls","neut"))
            CACHE.update({"ml":ml,"ict":ict,"ind":ind,
                          "pos":get_position(p,ind["atr"],ict["dir"]),
                          "heure":datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
            print("[RETRAIN] ✅")
        except Exception as e: print(f"[RETRAIN] ❌ {e}")
    threading.Thread(target=_do,daemon=True).start()
    return jsonify({"ok":True})

@app.route("/api/ia", methods=["POST"])
def api_ia():
    if not READY: return jsonify({"ok":False})
    import requests as req
    d=CACHE; ml=d.get("ml"); ict=d.get("ict",{}); I=d.get("ind",{}); mtf=d.get("mtf",{}); pos=d.get("pos",{})
    ml_s=(f"ML:{ml['signal']} Bull{ml['bull']}% Bear{ml['bear']}% "
          f"GB:{ml['gb_acc']}% LR:{ml['lr_acc']}% RF:{ml['rf_acc']}%") if ml else "N/A"
    prompt=(f"Expert XAU/USD ICT+ML. Réponds en français 400 mots max.\n"
            f"Prix:${d['prix']} Var:{d['pct24']:+.2f}%\n"
            f"Score ICT:{ict.get('score',0)}/100 Signal:{ict.get('signal','?')}\n{ml_s}\n"
            f"MTF:{mtf.get('conf','?')} M15:{mtf.get('M15',{}).get('trend','?')} "
            f"H1:{mtf.get('H1',{}).get('trend','?')} H4:{mtf.get('H4',{}).get('trend','?')}\n"
            f"RSI:{I.get('rsi',0):.1f} MACD:{'▲' if I.get('macd_bull') else '▼'} ATR:${I.get('atr',0)}\n"
            f"Setup:{pos.get('dir','?').upper()} Entrée:{pos.get('entry')} SL:{pos.get('sl')} TP1:{pos.get('tp1')}\n"
            f"1.MARCHÉ 2.ML 3.MTF 4.RECOMMANDATION 5.RISQUE")
    try:
        r=req.post("https://text.pollinations.ai/",
                   headers={"Content-Type":"application/json"},
                   json={"messages":[{"role":"user","content":prompt}],
                         "model":"openai","seed":42,"temperature":0.4,"private":True},timeout=30)
        if r.status_code==200 and len(r.text)>50:
            return jsonify({"ok":True,"text":r.text.strip(),"model":"GPT-4o-mini"})
    except: pass
    return jsonify({"ok":True,"text":f"Score ICT:{ict.get('score',0)} — {ict.get('signal','?')}\n{ml_s}","model":"Local"})

# ══════════════════════════════════════════════
#  HTML — ZÉRO CDN
# ══════════════════════════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>GOLD ML v3</title>
<style>
:root{--bg:#03050a;--bg1:#080d18;--gold:#e8b84b;--gold2:#f5d07a;
  --gr:#00d17a;--rd:#f03050;--bl:#2880f0;--cy:#00b8d4;--wa:#f0a020;--pu:#9b59b6;
  --tx:#b8c4d8;--tx2:#48586e;--tx3:#788898;--brd:rgba(200,152,42,.12);--r:10px}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:monospace;font-size:12px;min-height:100vh}
.hdr{position:sticky;top:0;z-index:99;background:rgba(3,5,10,.97);border-bottom:1px solid var(--brd);padding:6px 12px}
.hr1{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:4px}
.hr2{display:flex;gap:5px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px}
.hr2::-webkit-scrollbar{display:none}
.logo{font-size:13px;font-weight:900;letter-spacing:3px;color:var(--gold)}
.sub{font-size:7px;letter-spacing:2px;color:var(--tx2)}
.hprice{font-size:20px;font-weight:900;color:var(--gold2);letter-spacing:1px}
.vtag{font-size:9px;font-weight:700;padding:2px 7px;border-radius:4px}
.up{background:rgba(0,209,122,.1);color:var(--gr);border:1px solid rgba(0,209,122,.25)}
.dn{background:rgba(240,48,80,.1);color:var(--rd);border:1px solid rgba(240,48,80,.25)}
.ttag{font-size:8px;font-weight:700;padding:3px 8px;border-radius:4px}
.tb{background:rgba(0,209,122,.07);color:var(--gr);border:1px solid rgba(0,209,122,.18)}
.tr{background:rgba(240,48,80,.07);color:var(--rd);border:1px solid rgba(240,48,80,.18)}
.tn{background:rgba(40,128,240,.07);color:var(--bl);border:1px solid rgba(40,128,240,.18)}
.live{display:flex;align-items:center;gap:4px;background:rgba(0,209,122,.05);border:1px solid rgba(0,209,122,.18);border-radius:20px;padding:2px 8px;font-size:8px;color:var(--gr)}
.dot{width:5px;height:5px;border-radius:50%;background:var(--gr);box-shadow:0 0 6px var(--gr);animation:bk 1.4s infinite}
@keyframes bk{0%,100%{opacity:1}50%{opacity:.25}}
.btn{font-family:monospace;font-size:8px;font-weight:700;letter-spacing:1px;padding:6px 11px;border-radius:5px;cursor:pointer;border:1px solid var(--brd);white-space:nowrap;flex-shrink:0;transition:opacity .15s}
.btn:active{opacity:.7}
.bg{background:rgba(200,152,42,.1);color:var(--gold);border-color:rgba(200,152,42,.25)}
.btr{background:rgba(240,160,32,.1);color:var(--wa);border-color:rgba(240,160,32,.25)}
.bai{background:rgba(0,184,212,.07);color:var(--cy);border-color:rgba(0,184,212,.2)}
.bex{background:rgba(0,209,122,.06);color:var(--gr);border-color:rgba(0,209,122,.18)}
.wrap{max-width:1200px;margin:0 auto;padding:10px 10px 30px}
.ubar{text-align:center;font-size:8px;color:var(--tx2);letter-spacing:1px;padding:4px;margin-bottom:9px;min-height:18px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:9px}
.mb{margin-bottom:9px}
.card{background:var(--bg1);border:1px solid var(--brd);border-radius:var(--r);padding:13px;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--gold),transparent)}
.ct{font-size:8px;font-weight:700;letter-spacing:2px;color:var(--tx2);margin-bottom:11px;display:flex;align-items:center;gap:7px}
.cb{font-size:7px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:3px;padding:2px 5px;color:var(--tx3)}
.sb{background:linear-gradient(135deg,rgba(155,89,182,.07),rgba(40,128,240,.03));border-color:rgba(155,89,182,.16)}
.sb::before{background:linear-gradient(90deg,var(--pu),var(--bl),transparent)}
.gauge{position:relative;width:100px;height:100px;flex-shrink:0}
.gval{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px;font-weight:900;text-align:center;line-height:1}
.ml-sig{font-size:28px;font-weight:900;text-align:center;letter-spacing:3px;margin:6px 0}
.mrow{display:flex;align-items:center;gap:7px;margin-bottom:6px}
.mn{font-size:8px;color:var(--tx3);width:100px;flex-shrink:0}
.mbar{flex:1;height:6px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden}
.mf{height:100%;border-radius:4px}
.mfb{background:linear-gradient(90deg,var(--gr),rgba(0,209,122,.3))}
.mfr{background:linear-gradient(90deg,var(--rd),rgba(240,48,80,.3))}
.mpct{font-size:9px;font-weight:700;width:38px;text-align:right;flex-shrink:0}
.macc{font-size:7px;color:var(--tx2);width:38px;text-align:right;flex-shrink:0}
.mtr{display:flex;align-items:center;justify-content:space-between;padding:6px 9px;border-radius:5px;margin-bottom:5px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.04)}
.mtf{font-size:9px;font-weight:700;color:var(--tx3);width:36px;flex-shrink:0}
.nlv{display:flex;align-items:center;justify-content:space-between;padding:8px 11px;border-radius:6px;margin-bottom:5px;font-size:9px}
.nlv-r{background:rgba(240,48,80,.05);border:1px solid rgba(240,48,80,.18)}
.nlv-g{background:rgba(200,152,42,.05);border:1px solid rgba(200,152,42,.22)}
.nlv-v{background:rgba(0,209,122,.05);border:1px solid rgba(0,209,122,.18)}
.nlvp{font-size:13px;font-weight:900}
.ir{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.03);font-size:9px}
.il{color:var(--tx2)}.iv{font-weight:700}
.vg{color:var(--gr)}.vr{color:var(--rd)}.vw{color:var(--wa)}.vn{color:var(--tx3)}
.pat{display:inline-flex;align-items:center;gap:4px;border:1px solid rgba(255,255,255,.07);border-radius:5px;padding:4px 8px;margin:2px;font-size:8px}
.patb{border-color:rgba(0,209,122,.22);background:rgba(0,209,122,.04)}
.patr{border-color:rgba(240,48,80,.22);background:rgba(240,48,80,.04)}
.kza{background:rgba(0,209,122,.05);border:1px solid rgba(0,209,122,.18);border-radius:var(--r);padding:10px;margin-bottom:7px}
.kzn{font-size:12px;font-weight:700;color:var(--gr);margin:3px 0}
.kznx{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:8px}
.inp{width:100%;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:4px;padding:5px 7px;color:var(--tx);font-family:inherit;font-size:10px;margin-top:3px}
.pr{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.03);font-size:9px}
.aip{background:var(--bg1);border:1px solid rgba(0,184,212,.12);border-radius:var(--r);position:relative;overflow:hidden}
.aip::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--cy),var(--bl),transparent)}
.aih{display:flex;justify-content:space-between;align-items:center;padding:10px 13px;border-bottom:1px solid rgba(0,184,212,.1)}
.ait{font-size:9px;font-weight:700;letter-spacing:2px;color:var(--cy)}
.aib{padding:12px;min-height:60px;font-size:9px;line-height:1.8}
.splash{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;gap:14px}
.sr{width:42px;height:42px;border:3px solid rgba(200,152,42,.15);border-top-color:var(--gold);border-radius:50%;animation:sp .9s linear infinite}
.st{font-size:9px;letter-spacing:3px;color:var(--tx2);font-weight:700;text-align:center}
@keyframes sp{to{transform:rotate(360deg)}}
canvas{display:block;width:100%}
@media(max-width:600px){.g2{grid-template-columns:1fr}.hprice{font-size:17px}}
</style>
</head>
<body>
<header class="hdr">
  <div class="hr1">
    <div><div class="logo">XAU/USD</div><div class="sub">GOLD ML v3.0</div></div>
    <div style="display:flex;align-items:center;gap:7px;flex:1;justify-content:center;flex-wrap:wrap">
      <span class="hprice" id="HP">---</span>
      <div style="display:flex;flex-direction:column;gap:2px">
        <span class="vtag up" id="HV">---</span>
        <div style="font-size:7px;color:var(--tx2);display:flex;gap:5px">
          H<span id="HH" style="color:var(--gr)">---</span>
          L<span id="HL" style="color:var(--rd)">---</span>
          <span id="HUTC" style="color:var(--tx3)">--:--</span>
        </div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:3px;align-items:flex-end;flex-shrink:0">
      <span class="ttag tn" id="HTREND">H4 ---</span>
      <div class="live"><div class="dot" id="DOT"></div>LIVE</div>
    </div>
  </div>
  <div class="hr2">
    <button class="btn bg"  onclick="doRefresh()">⟳ REFRESH</button>
    <button class="btn btr" id="BTR" onclick="doRetrain()">🔄 ENTRAÎNER</button>
    <button class="btn bai" id="BAI" onclick="doIA()">✦ IA</button>
    <button class="btn bex" onclick="doCSV()">📥 CSV</button>
  </div>
</header>

<div class="wrap">
  <div class="ubar" id="UB">Initialisation...</div>
  <div id="ROOT">
    <div class="splash">
      <div class="sr"></div>
      <div class="st">ANALYSE ML EN COURS</div>
      <div style="font-size:8px;color:var(--tx2);margin-top:4px" id="SD">Connexion...</div>
    </div>
  </div>
</div>

<script>
// ════════════════════════════════════════════════════════
//  GOLD ML v3 — SCRIPT COMPLET AUTO-CONTENU
//  Tout est défini ICI avant le démarrage du polling.
//  Zéro dépendance CDN pour l'affichage.
// ════════════════════════════════════════════════════════
var _D = null;

function f2(n,d){ d=d!=null?d:2; return n==null?'---':Number(n).toLocaleString('fr-FR',{minimumFractionDigits:d,maximumFractionDigits:d}); }
function pc(n){ return n==null?'---':(n>=0?'+':'')+Number(n).toFixed(2)+'%'; }
function esc(s){ return String(s||'').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Header ──────────────────────────────────────────
function setHeader(p,v,utc){
  var e=document.getElementById('HP'); if(e) e.textContent='$'+f2(p);
  var ve=document.getElementById('HV');
  if(ve){ve.textContent=pc(v);ve.className='vtag '+(v>=0?'up':'dn');}
  var ue=document.getElementById('HUTC'); if(ue) ue.textContent=utc||'';
}

// ── RENDER ──────────────────────────────────────────
function renderAll(D){
  _D=D;
  setHeader(D.prix,D.var,D.utc);
  if(D.hi) document.getElementById('HH').textContent=' $'+f2(D.hi);
  if(D.lo) document.getElementById('HL').textContent=' $'+f2(D.lo);
  var tc={bull:'tb',bear:'tr',neut:'tn'}[D.trend_cls]||'tn';
  var te=document.getElementById('HTREND');
  te.textContent='H4 '+(D.trend_cls==='bull'?'▲':D.trend_cls==='bear'?'▼':'➡');
  te.className='ttag '+tc;
  document.getElementById('DOT').style.background='var(--gr)';
  document.getElementById('UB').textContent=
    (D.heure||'')+' · '+(D.utc||'')+' · '+D.src+(D.ml?' · ML✅':'')+' · MTF refresh 5min';
  document.getElementById('ROOT').innerHTML=buildUI(D);
  setTimeout(function(){ drawAll(D); },30);
  // refresh prix toutes les 10s
  if(!window._pxInt) window._pxInt=setInterval(function(){
    fetch('/api/prix').then(function(r){return r.json();}).then(function(d){
      if(d.ok&&_D){ setHeader(d.prix,d.var,d.utc); _D.prix=d.prix; }
    });
  },10000);
  // refresh complet toutes les 5min
  if(!window._fullInt) window._fullInt=setInterval(function(){
    xget('/api/data',function(d){
      if(d&&d.prix) renderAll(d);
    });
  },300000);
}

// ── BUILD UI ─────────────────────────────────────────
function buildUI(D){
  var ML=D.ml,ICT=D.ict,MTF=D.mtf,I=D.ind,K=D.kz,POS=D.pos;
  var dir=ICT?ICT.dir:'neut';
  var clr=dir==='bull'?'var(--gr)':dir==='bear'?'var(--rd)':'var(--tx3)';
  var isN=dir==='neut';
  var ci=2*Math.PI*46, arc=ci*((ICT?ICT.abs:0)/100);

  // Bannière
  var bg=dir==='bull'?'rgba(0,209,122,.04)':dir==='bear'?'rgba(240,48,80,.04)':'rgba(40,128,240,.03)';
  var bc=dir==='bull'?'rgba(0,209,122,.18)':dir==='bear'?'rgba(240,48,80,.18)':'rgba(40,128,240,.12)';
  var ic=dir==='bull'?'🟢':dir==='bear'?'🔴':'🔵';
  var bnr='<div style="display:flex;align-items:center;gap:11px;padding:11px 14px;border-radius:var(--r);margin-bottom:9px;border:1px solid '+bc+';background:'+bg+'">'
    +'<div style="font-size:20px">'+ic+'</div>'
    +'<div><div style="font-size:13px;font-weight:900;color:'+clr+'">'+(ICT?ICT.signal:'---')+'</div>'
    +'<div style="font-size:8px;color:var(--tx3)">'+(ML?'ML:'+ML.signal+' · Bull '+ML.bull+'% Bear '+ML.bear+'% · '+ML.conf:'ML calcul...')+'</div></div></div>';

  // ICT Score
  var det=(ICT?ICT.det:[]).map(function(d){
    return '<div style="display:flex;justify-content:space-between;font-size:8px;padding:2px 0;border-bottom:1px solid rgba(255,255,255,.03)">'
      +'<span style="color:var(--tx3)">'+esc(d[0])+'</span>'
      +'<span style="color:'+(d[1]>0?'var(--gr)':d[1]<0?'var(--rd)':'var(--tx2)')+'\">'+(d[1]>0?'+':'')+d[1]+'</span></div>';
  }).join('');
  var sc='<div class="card mb"><div class="ct">🎯 SCORE ICT<span class="cb">0-100</span></div>'
    +'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
    +'<div class="gauge"><svg width="100" height="100" viewBox="0 0 100 100" style="transform:rotate(-90deg)">'
    +'<circle cx="50" cy="50" r="46" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="8"/>'
    +'<circle cx="50" cy="50" r="46" fill="none" stroke="'+clr+'" stroke-width="8" stroke-dasharray="'+arc+' '+ci+'" stroke-linecap="round"/></svg>'
    +'<div class="gval" style="color:'+clr+'">'+(ICT?ICT.abs:0)+'</div></div>'
    +'<div style="flex:1;min-width:110px"><div style="font-size:12px;font-weight:900;color:'+clr+';margin-bottom:6px">'+(ICT?ICT.signal:'---')+'</div>'+det+'</div></div></div>';

  // ML Card
  function mb(nm,bull,acc,ic){
    var b=bull>50,pct=b?bull:100-bull;
    return '<div class="mrow"><div class="mn">'+ic+' '+nm+'</div>'
      +'<div class="mbar"><div class="mf '+(b?'mfb':'mfr')+'" style="width:'+pct+'%"></div></div>'
      +'<div class="mpct" style="color:'+(b?'var(--gr)':'var(--rd)')+'\">'+(b?'▲'+bull:'▼'+(100-bull))+'%</div>'
      +'<div class="macc">'+acc+'%</div></div>';
  }
  var ml='<div class="card sb mb"><div class="ct">🤖 MACHINE LEARNING<span class="cb">'+(ML?'3 MODÈLES':'⟳')+'</span></div>'
    +(ML?'<div class="ml-sig" style="color:'+clr+'">'+ML.signal+'</div>'
      +'<div style="text-align:center;font-size:8px;color:var(--tx3);margin-bottom:10px">Confiance <span style="color:'+(ML.conf==='FORTE'?clr:'var(--wa)')+'">'+ML.conf+'</span> · '+ML.n+' ex.</div>'
      +'<div style="display:flex;justify-content:center;gap:18px;margin-bottom:11px">'
      +'<div style="text-align:center"><div style="font-size:18px;font-weight:900;color:var(--gr)">'+ML.bull+'%</div><div style="font-size:7px;color:var(--tx3)">BULL</div></div>'
      +'<div style="text-align:center"><div style="font-size:18px;font-weight:900;color:var(--rd)">'+ML.bear+'%</div><div style="font-size:7px;color:var(--tx3)">BEAR</div></div></div>'
      +mb('Gradient Boost',ML.gb,ML.gb_acc,'⚡')
      +mb('Logistic Regr.',ML.lr,ML.lr_acc,'📈')
      +mb('Random Forest',ML.rf,ML.rf_acc,'🌲')
      +'<div style="margin-top:6px;padding:4px 8px;background:rgba(255,255,255,.03);border-radius:4px;display:flex;justify-content:space-between;font-size:7px">'
      +'<span style="color:var(--tx3)">Biais MTF</span>'
      +'<span style="color:'+(ML.bias>=0?'var(--gr)':'var(--rd)')+'\">'+(ML.bias>=0?'+':'')+ML.bias+'%</span></div>'
    :'<div style="text-align:center;padding:16px;color:var(--tx3);font-size:9px">⟳ Calcul...</div>')+'</div>';

  // Patterns
  var pts=(D.pats||[]).map(function(p){
    return '<div class="pat '+(p.s==='BULL'?'patb':'patr')+'">'+p.ic+' <div><div style="font-weight:700">'+p.n+'</div><div style="font-size:7px;color:var(--tx3)">'+p.d+'</div></div></div>';
  }).join('');
  var pCard='<div class="card mb"><div class="ct">🕯️ PATTERNS BOUGIES</div>'
    +(pts?'<div style="display:flex;flex-wrap:wrap;gap:3px">'+pts+'</div>'
     :'<div style="color:var(--tx3);font-size:9px;text-align:center;padding:8px">Aucun pattern confirmé</div>')+'</div>';

  // MTF
  function mr(tf,d){
    if(!d) return '';
    var c=d.cls==='bull'?'vg':d.cls==='bear'?'vr':'vn';
    return '<div class="mtr"><div class="mtf">'+tf+'</div>'
      +'<div class="iv '+c+'" style="font-size:8px">'+d.trend+'</div>'
      +'<div style="font-size:7px;color:var(--tx2)">RSI '+d.rsi+'</div>'
      +'<div style="font-size:7px;color:var(--tx2)">MACD '+(d.macd_bull?'▲':'▼')+'</div></div>';
  }
  var cc=(MTF&&MTF.conf||'').includes('TOTAL')?clr:(MTF&&MTF.conf||'').includes('MAJORI')?'var(--wa)':'var(--tx3)';
  var mtf='<div class="card mb"><div class="ct">📊 MULTI-TIMEFRAME<span class="cb">M15·H1·H4</span></div>'
    +mr('M15',MTF&&MTF.M15)+mr('H1',MTF&&MTF.H1)+mr('H4',MTF&&MTF.H4)
    +'<div style="margin-top:7px;padding:6px 9px;background:rgba(255,255,255,.02);border-radius:5px;font-size:9px;font-weight:700;color:'+cc+'">📡 '+(MTF?MTF.conf:'---')+'</div>'
    +'<div style="margin-top:4px;font-size:7px;color:var(--tx2);text-align:right">♻️ refresh 5min · '+(D.utc||'')+'</div></div>';

  // KZ
  var kzh=K&&K.active?'<div class="kza"><div style="font-size:18px">'+K.active.ic+'</div>'
    +'<div class="kzn">'+K.active.name+'</div>'
    +'<div style="font-size:8px;color:var(--tx3)">'+K.active.start+'–'+K.active.end+' UTC</div>'
    +'<div style="margin-top:5px;background:rgba(0,209,122,.09);border:1px solid rgba(0,209,122,.2);border-radius:20px;display:inline-flex;padding:2px 8px;font-size:8px;color:var(--gr)">⏱ '+K.active.rem+' min</div></div>'
    :'<div style="text-align:center;padding:12px;color:var(--tx3);font-size:9px">😴 Hors Kill Zone</div>';
  var kzn=(K&&K.next||[]).map(function(n){
    return '<div class="kznx"><span>'+n.ic+' '+n.name+' ('+n.start+'–'+n.end+')</span><span style="color:var(--wa)">'+n.dans+'</span></div>';}).join('');
  var kz='<div class="card mb"><div class="ct">⏰ KILL ZONES</div>'+kzh
    +'<div style="font-size:7px;color:var(--tx2);letter-spacing:1px;margin:9px 0 5px">PROCHAINES</div>'+kzn+'</div>';

  // Setup
  var sup='<div class="card mb"><div class="ct">⚡ SETUP<span class="cb" style="color:'+clr+'">'+dir.toUpperCase()+'</span></div>'
    +(isN
      ?'<div style="text-align:center;padding:20px 10px">'
       +'<div style="font-size:26px;margin-bottom:9px">⏳</div>'
       +'<div style="font-size:10px;color:var(--wa);font-weight:700;letter-spacing:2px;margin-bottom:7px">EN ATTENTE</div>'
       +'<div style="font-size:8px;color:var(--tx3);line-height:1.8">Aucun trade recommandé<br>'
       +'Attendez alignement ICT+ML+MTF<br>'
       +'Score: '+(ICT?ICT.abs:0)+'/100 · ATR $'+(I?I.atr:0)+'</div></div>'
      :'<div class="nlv nlv-v"><span>🎯 TP3 (1:4.0)</span><span class="nlvp" style="color:var(--gr)">$'+f2(POS&&POS.tp3)+'</span></div>'
       +'<div class="nlv nlv-v"><span>🎯 TP2 (1:2.5)</span><span class="nlvp" style="color:var(--gr)">$'+f2(POS&&POS.tp2)+'</span></div>'
       +'<div class="nlv nlv-v"><span>🎯 TP1 (1:1.5)</span><span class="nlvp" style="color:var(--gr)">$'+f2(POS&&POS.tp1)+'</span></div>'
       +'<div class="nlv nlv-g"><span>⚡ ENTRÉE</span><span class="nlvp" style="color:var(--gold)">$'+f2(POS&&POS.entry)+'</span></div>'
       +'<div class="nlv nlv-r"><span>🛑 SL</span><span class="nlvp" style="color:var(--rd)">$'+f2(POS&&POS.sl)+' <span style="font-size:8px">('+f2(POS&&POS.sl_d)+'$)</span></span></div>'
       +'<div style="display:flex;justify-content:space-between;margin-top:7px;font-size:8px;color:var(--tx3)"><span>RR 1:1.5/2.5/4</span><span>ATR $'+(I?I.atr:0)+'</span></div>')
    +'</div>';

  // Calculateur
  function rpos(P){
    if(!P) return '';
    return '<div class="pr"><span style="color:var(--tx2)">Direction</span><span class="iv '+(P.dir==='bull'?'vg':'vr')+'">'+(P.dir==='bull'?'▲ LONG':'▼ SHORT')+'</span></div>'
      +'<div class="pr"><span style="color:var(--tx2)">Risque $</span><span class="iv" style="color:var(--gold)">$'+f2(P.risk_usd)+'</span></div>'
      +'<div class="pr"><span style="color:var(--tx2)">Taille (oz)</span><span class="iv">'+P.lots+'</span></div>'
      +'<div class="pr"><span style="color:var(--tx2)">SL distance</span><span class="iv vr">$'+f2(P.sl_d)+'</span></div>';
  }
  var calc=isN
    ?'<div class="card mb"><div class="ct">💰 POSITION</div><div style="text-align:center;padding:16px;color:var(--tx3);font-size:9px">Disponible sur signal BULL/BEAR</div></div>'
    :'<div class="card mb"><div class="ct">💰 POSITION</div>'
     +'<div class="g2" style="gap:7px;margin-bottom:9px">'
     +'<div><label style="font-size:7px;color:var(--tx2)">Capital ($)</label><input id="IC" class="inp" type="number" value="10000"></div>'
     +'<div><label style="font-size:7px;color:var(--tx2)">Risque (%)</label><input id="IR" class="inp" type="number" value="1" step="0.5" min="0.5" max="5"></div></div>'
     +'<button onclick="calcPos()" class="btn bg" style="width:100%;margin-bottom:9px">⟳ CALCULER</button>'
     +'<div id="PR">'+rpos(POS)+'</div></div>';

  // Indicateurs
  function ir(l,v,c){ return '<div class="ir"><span class="il">'+l+'</span><span class="iv '+c+'">'+esc(v)+'</span></div>'; }
  var rc=I.rsi>70?'vw':I.rsi<30?'vg':I.rsi>50?'vg':'vr';
  var ind='<div class="card mb"><div class="ct">📐 INDICATEURS</div>'
    +ir('Prix','$'+f2(D.prix),'vn')
    +ir('MM20','$'+f2(I.mm20),D.prix>I.mm20?'vg':'vr')
    +ir('MM200','$'+f2(I.mm200),D.prix>I.mm200?'vg':'vr')
    +ir('RSI',I.rsi.toFixed(1),rc)
    +ir('Stoch K/D',I.stk.toFixed(0)+'/'+I.std.toFixed(0),I.stk>50?'vg':'vr')
    +ir('BB %',(I.bb_p*100).toFixed(1)+'%',I.bb_p>0.5?'vg':'vr')
    +ir('MACD',(I.mhist>=0?'+':'')+I.mhist,I.macd_bull?'vg':'vr')
    +ir('ATR','$'+I.atr,'vn')
    +ir('Volatilité',I.vol_lbl,I.vol_lbl.includes('FORTE')?'vw':'vn')
    +ir('Volume','×'+I.vol_r,I.vol_ok?'vg':'vr')
    +ir('Var 24H',pc(D.pct24),D.pct24>=0?'vg':'vr')
    +'</div>';

  // Graphiques canvas
  var charts='<div class="card mb"><div class="ct">🕯️ PRIX M15<span class="cb">LIVE 10s</span></div>'
    +'<canvas id="CPX" height="220" style="height:220px"></canvas></div>'
    +'<div class="g2 mb">'
    +'<div class="card"><div class="ct">RSI<span class="cb">14</span></div><canvas id="CRS" height="100" style="height:100px"></canvas></div>'
    +'<div class="card"><div class="ct">MACD<span class="cb">histo</span></div><canvas id="CMC" height="100" style="height:100px"></canvas></div>'
    +'</div>'
    +'<div class="card mb"><div class="ct">VOLUME</div><canvas id="CVL" height="80" style="height:80px"></canvas></div>';

  // IA
  var ai='<div class="aip mb" id="AIP"><div class="aih"><div class="ait">✦ ANALYSE IA</div>'
    +'<div style="font-size:7px;color:var(--tx2)" id="AIM">Pollinations · Gratuit · Sans clé</div></div>'
    +'<div class="aib" id="AIB" style="text-align:center;color:var(--tx3)">'
    +'<div style="font-size:18px;opacity:.3;margin-bottom:6px">✦</div>Cliquer ✦ IA</div></div>';

  var foot='<div style="text-align:center;padding:12px;font-size:7px;color:var(--tx2);border-top:1px solid var(--brd)">GOLD ML v3.0 · '+D.heure+' · ⚠️ Usage éducatif uniquement</div>';

  return bnr
    +'<div class="g2 mb">'+sc+ml+'</div>'
    +pCard
    +'<div class="g2 mb">'+mtf+kz+'</div>'
    +'<div class="g2 mb">'+sup+calc+'</div>'
    +ind+charts+ai+foot;
}

function calcPos(){
  if(!_D) return;
  var cap=parseFloat(document.getElementById('IC').value)||10000;
  var risk=parseFloat(document.getElementById('IR').value)||1;
  var p=_D.prix,atr=_D.ind.atr,dir=_D.ict?_D.ict.dir:'bull';
  var sld=Math.max(Math.round(atr*0.4*100)/100,1.5);
  var rd=Math.round(cap*risk/100*100)/100;
  var lots=Math.round(rd/sld*10000)/10000;
  var P={dir:dir,risk_usd:rd,lots:lots,sl_d:sld,entry:p,
    sl:dir==='bull'?Math.round((p-sld)*100)/100:Math.round((p+sld)*100)/100,
    tp1:dir==='bull'?Math.round((p+sld*1.5)*100)/100:Math.round((p-sld*1.5)*100)/100,
    tp2:dir==='bull'?Math.round((p+sld*2.5)*100)/100:Math.round((p-sld*2.5)*100)/100,
    tp3:dir==='bull'?Math.round((p+sld*4)*100)/100:Math.round((p-sld*4)*100)/100};
  var pr=document.getElementById('PR'); if(!pr) return;
  pr.innerHTML='<div class="pr"><span style="color:var(--tx2)">Direction</span><span class="iv '+(P.dir==='bull'?'vg':'vr')+'">'+(P.dir==='bull'?'▲ LONG':'▼ SHORT')+'</span></div>'
    +'<div class="pr"><span style="color:var(--tx2)">Risque $</span><span class="iv" style="color:var(--gold)">$'+f2(P.risk_usd)+'</span></div>'
    +'<div class="pr"><span style="color:var(--tx2)">Taille</span><span class="iv">'+P.lots+' oz</span></div>'
    +'<div class="pr"><span style="color:var(--tx2)">SL</span><span class="iv vr">$'+f2(P.sl)+' (−$'+f2(P.sl_d)+')</span></div>'
    +'<div class="pr"><span style="color:var(--tx2)">TP1</span><span class="iv vg">$'+f2(P.tp1)+'</span></div>'
    +'<div class="pr"><span style="color:var(--tx2)">TP2</span><span class="iv vg">$'+f2(P.tp2)+'</span></div>'
    +'<div class="pr"><span style="color:var(--tx2)">TP3</span><span class="iv vg">$'+f2(P.tp3)+'</span></div>';
}

// ── GRAPHIQUES CANVAS PUR ───────────────────────────
function drawAll(D){
  try{ drawCandles(D); }catch(e){}
  try{ drawLine('CRS',D.ind.rsi_l,'#9b59b6',0,100,[{y:70,c:'rgba(240,48,80,.3)'},{y:30,c:'rgba(0,209,122,.3)'}]); }catch(e){}
  try{ drawBars('CMC',D.ind.mhist_l,false); }catch(e){}
  try{ drawBars('CVL',D.ind.vol_l,true); }catch(e){}
}

function drawCandles(D){
  var cv=document.getElementById('CPX'); if(!cv) return;
  var ohlc=D.ohlc||[]; if(!ohlc.length) return;
  var W=cv.offsetWidth||320; var H=220;
  cv.width=W; cv.height=H;
  var ctx=cv.getContext('2d'); if(!ctx) return;
  var n=Math.min(ohlc.length,60); var data=ohlc.slice(-n);
  var mx=Math.max.apply(null,data.map(function(b){return b.high;}));
  var mn=Math.min.apply(null,data.map(function(b){return b.low;}));
  var pad=10; var rng=mx-mn||1;
  var bw=Math.max(2,Math.floor((W-pad*2)/n)-1);
  function py(v){return Math.round(H-pad-((v-mn)/rng)*(H-pad*2));}
  // Grid
  for(var g=0;g<5;g++){
    var gy=pad+g*(H-pad*2)/4;
    ctx.strokeStyle='rgba(255,255,255,.04)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(W,gy); ctx.stroke();
    ctx.fillStyle='rgba(120,136,152,.55)'; ctx.font='8px monospace';
    ctx.fillText('$'+Math.round(mx-g*rng/4),2,gy-2);
  }
  // BB
  var I=D.ind;
  if(I.bb_u_l&&I.bb_u_l.length>=n){
    var U=I.bb_u_l.slice(-n), L2=I.bb_d_l.slice(-n);
    ctx.strokeStyle='rgba(200,152,42,.22)'; ctx.lineWidth=1;
    ctx.beginPath();
    U.forEach(function(v,i){var x=pad+i*(bw+1)+bw/2;i===0?ctx.moveTo(x,py(v)):ctx.lineTo(x,py(v));});
    ctx.stroke();
    ctx.beginPath();
    L2.forEach(function(v,i){var x=pad+i*(bw+1)+bw/2;i===0?ctx.moveTo(x,py(v)):ctx.lineTo(x,py(v));});
    ctx.stroke();
  }
  // MM20
  if(I.mm20_l&&I.mm20_l.length>=n){
    ctx.strokeStyle='rgba(40,128,240,.65)'; ctx.lineWidth=1.5;
    ctx.beginPath();
    I.mm20_l.slice(-n).forEach(function(v,i){if(!v)return;var x=pad+i*(bw+1)+bw/2;i===0?ctx.moveTo(x,py(v)):ctx.lineTo(x,py(v));});
    ctx.stroke();
  }
  // Bougies
  data.forEach(function(b,i){
    var x=pad+i*(bw+1);
    var bull=b.close>=b.open;
    var col=bull?'#00d17a':'#f03050';
    ctx.strokeStyle=col; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x+bw/2,py(b.high)); ctx.lineTo(x+bw/2,py(b.low)); ctx.stroke();
    var top=py(Math.max(b.open,b.close)), bot=py(Math.min(b.open,b.close));
    ctx.fillStyle=col; ctx.fillRect(x,top,bw,Math.max(1,bot-top));
  });
  // SL/TP
  if(D.pos&&D.ict&&D.ict.dir!=='neut'){
    [{p:D.pos.entry,c:'rgba(232,184,75,.85)',l:'⚡'},
     {p:D.pos.sl,  c:'rgba(240,48,80,.75)',l:'SL'},
     {p:D.pos.tp1, c:'rgba(0,209,122,.5)',l:'TP1'},
     {p:D.pos.tp2, c:'rgba(0,209,122,.7)',l:'TP2'}].forEach(function(ln){
      if(ln.p>=mn&&ln.p<=mx){
        ctx.strokeStyle=ln.c; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
        ctx.beginPath(); ctx.moveTo(pad,py(ln.p)); ctx.lineTo(W-26,py(ln.p)); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle=ln.c; ctx.font='8px monospace';
        ctx.fillText(ln.l,W-25,py(ln.p)+3);
      }
    });
    ctx.setLineDash([]);
  }
  // Prix live
  var p=D.prix;
  if(p>=mn&&p<=mx){
    ctx.strokeStyle='rgba(245,208,122,.5)'; ctx.lineWidth=1; ctx.setLineDash([2,2]);
    ctx.beginPath(); ctx.moveTo(0,py(p)); ctx.lineTo(W,py(p)); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle='rgba(245,208,122,.9)'; ctx.font='bold 8px monospace';
    ctx.fillText('$'+f2(p,0),2,py(p)-2);
  }
}

function drawLine(id,data,col,ymin,ymax,refs){
  var cv=document.getElementById(id); if(!cv||!data||!data.length) return;
  var W=cv.offsetWidth||150; var H=parseInt(cv.getAttribute('height'))||100;
  cv.width=W; cv.height=H;
  var ctx=cv.getContext('2d'); if(!ctx) return;
  var pad=3;
  var mn=ymin!=null?ymin:Math.min.apply(null,data);
  var mx=ymax!=null?ymax:Math.max.apply(null,data);
  var rng=mx-mn||1;
  function py(v){return H-pad-((v-mn)/rng)*(H-pad*2);}
  (refs||[]).forEach(function(r){
    ctx.strokeStyle=r.c; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(0,py(r.y)); ctx.lineTo(W,py(r.y)); ctx.stroke();
    ctx.setLineDash([]);
  });
  ctx.strokeStyle=col; ctx.lineWidth=2; ctx.lineJoin='round';
  ctx.beginPath();
  data.forEach(function(v,i){
    var x=pad+(i/(data.length-1||1))*(W-pad*2);
    i===0?ctx.moveTo(x,py(v)):ctx.lineTo(x,py(v));
  });
  ctx.stroke();
  ctx.fillStyle=col; ctx.font='9px monospace';
  ctx.fillText(data[data.length-1].toFixed(1),2,11);
}

function drawBars(id,data,vol){
  var cv=document.getElementById(id); if(!cv||!data||!data.length) return;
  var W=cv.offsetWidth||150; var H=parseInt(cv.getAttribute('height'))||100;
  cv.width=W; cv.height=H;
  var ctx=cv.getContext('2d'); if(!ctx) return;
  var pad=3; var n=data.length;
  var mx=Math.max.apply(null,data.map(Math.abs))||1;
  var mn2=vol?0:-mx;
  var bw=Math.max(2,Math.floor((W-pad*2)/n)-1);
  var mean=vol?data.reduce(function(a,b){return a+b;},0)/n:0;
  function py(v){return H-pad-((v-mn2)/(mx-mn2))*(H-pad*2);}
  data.forEach(function(v,i){
    var x=pad+i*(bw+1);
    var col=vol?(v>mean*1.5?'rgba(0,209,122,.8)':v>mean?'rgba(0,209,122,.4)':'rgba(72,88,110,.35)')
               :(v>=0?'rgba(0,209,122,.7)':'rgba(240,48,80,.7)');
    ctx.fillStyle=col;
    var top=py(Math.max(v,0)), bot=py(Math.min(v,0));
    ctx.fillRect(x,top,bw,Math.max(1,bot-top));
  });
  if(vol){
    ctx.strokeStyle='rgba(240,160,32,.5)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(0,py(mean)); ctx.lineTo(W,py(mean)); ctx.stroke();
    ctx.setLineDash([]);
  } else {
    ctx.strokeStyle='rgba(255,255,255,.08)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(0,py(0)); ctx.lineTo(W,py(0)); ctx.stroke();
  }
}

// ── Candle live ──────────────────────────────────────
setInterval(function(){
  xget('/api/candle',function(d){
    if(!d.ok||!_D) return;
    setHeader(d.prix,d.var,d.utc); _D.prix=d.prix;
    if(_D.ohlc&&_D.ohlc.length){
      var last=_D.ohlc[_D.ohlc.length-1];
      last.close=d.prix; last.high=Math.max(last.high,d.prix); last.low=Math.min(last.low,d.prix);
      try{ drawCandles(_D); }catch(e){}
    }
    var dot=document.getElementById('DOT');
    if(dot){dot.style.background='#fff';setTimeout(function(){dot.style.background='var(--gr)';},120);}
  }).catch(function(){});
},10000);

// ── Actions ──────────────────────────────────────────
function doRefresh(){
  if(window._pT){clearInterval(window._pT);window._pT=null;}
  if(window._pxInt){clearInterval(window._pxInt);window._pxInt=null;}
  if(window._fullInt){clearInterval(window._fullInt);window._fullInt=null;}
  window._pDone=false; window._pN=0;
  document.getElementById('ROOT').innerHTML='<div class="splash"><div class="sr"></div><div class="st">RECHARGEMENT...</div></div>';
  startPoll();
}

function doRetrain(){
  var b=document.getElementById('BTR'); b.textContent='⟳...'; b.disabled=true;
  fetch('/api/retrain',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
    .then(function(r){return r.json();}).then(function(res){
      if(res.ok){
        document.getElementById('UB').textContent='✅ Ré-entraînement lancé... (30s)';
        setTimeout(function(){
          xget('/api/data',function(d){
            if(d&&d.prix) renderAll(d);
          }).catch(function(){});
          b.textContent='🔄 ENTRAÎNER'; b.disabled=false;
        },30000);
      } else { b.textContent='🔄 ENTRAÎNER'; b.disabled=false; }
    }).catch(function(){b.textContent='🔄 ENTRAÎNER';b.disabled=false;});
}

function doIA(){
  var b=document.getElementById('BAI'); b.disabled=true; b.textContent='⟳...';
  var body=document.getElementById('AIB');
  if(body) body.innerHTML='<div style="text-align:center;color:var(--tx3)">⟳ Analyse IA... (15-30s)</div>';
  fetch('/api/ia',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
    .then(function(r){return r.json();}).then(function(res){
      b.textContent='✦ IA'; b.disabled=false;
      if(body&&res.ok){
        var txt=res.text.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');
        body.innerHTML=txt;
        var m=document.getElementById('AIM'); if(m) m.textContent=res.model||'';
      }
    }).catch(function(){b.textContent='✦ IA';b.disabled=false;});
}

function doCSV(){
  if(!_D){return;}
  var D=_D,ML=D.ml,ICT=D.ict,I=D.ind,MTF=D.mtf,POS=D.pos;
  var lines=['GOLD ML v3.0 - XAU/USD','Date,'+D.heure,'Source,'+D.src,'',
    'PRIX','Actuel,$'+D.prix,'Var 24H,'+D.pct24+'%','High,$'+D.hi,'Low,$'+D.lo,'',
    'ML','Signal,'+(ML?ML.signal:'N/A'),'Bull%,'+(ML?ML.bull:'N/A'),'Bear%,'+(ML?ML.bear:'N/A'),
    'GB acc,'+(ML?ML.gb_acc+'%':'N/A'),'LR acc,'+(ML?ML.lr_acc+'%':'N/A'),'RF acc,'+(ML?ML.rf_acc+'%':'N/A'),'',
    'ICT','Score,'+(ICT?ICT.score:0),'Signal,'+(ICT?ICT.signal:'N/A'),'',
    'MTF','Conf,'+(MTF?MTF.conf:'N/A'),
    'M15,'+(MTF&&MTF.M15?MTF.M15.trend:'N/A'),
    'H1,'+(MTF&&MTF.H1?MTF.H1.trend:'N/A'),
    'H4,'+(MTF&&MTF.H4?MTF.H4.trend:'N/A'),'',
    'SETUP','Direction,'+(POS?POS.dir:'N/A'),'Entrée,$'+(POS?POS.entry:'N/A'),
    'SL,$'+(POS?POS.sl:'N/A'),'TP1,$'+(POS?POS.tp1:'N/A'),
    'TP2,$'+(POS?POS.tp2:'N/A'),'TP3,$'+(POS?POS.tp3:'N/A')];
  var blob=new Blob([lines.join('\n')],{type:'text/csv'});
  var a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='gold_ml_'+new Date().toISOString().slice(0,16).replace(':','-')+'.csv';
  a.click();
}

// ════════════════════════════════════════════════════════
//  POLLING — démarré EN DERNIER après que tout soit défini
// ════════════════════════════════════════════════════════
window._pN=0; window._pDone=false; window._pT=null;

// ── GET XHR (XMLHttpRequest) — plus fiable que fetch sur Android ──
function xget(url, cb){
  var x=new XMLHttpRequest();
  x.open('GET', url, true);
  x.timeout=8000;
  x.onreadystatechange=function(){
    if(x.readyState===4){
      if(x.status===200){
        try{ cb(JSON.parse(x.responseText)); }catch(e){}
      }
    }
  };
  x.ontimeout=function(){};
  x.onerror=function(){};
  try{ x.send(); }catch(e){}
}

function _tick(){
  if(window._pDone) return;
  window._pN++;
  var sp='⣾⣽⣻⢿⡿⣟⣯⣷';
  var ub=document.getElementById('UB');
  var sd=document.getElementById('SD');
  if(ub) ub.textContent=sp[window._pN%8]+' Calcul ML+MTF... ('+window._pN*2+'s)';
  if(sd){
    if(window._pN<3) sd.textContent='Connexion http://127.0.0.1:8083...';
    else if(window._pN<10) sd.textContent='GBM + LR + Random Forest en cours...';
    else sd.textContent='Analyse MTF M15 H1 H4...';
  }

  xget('/api/ready', function(s){
    if(!s || !s.ready) return;
    // Backend prêt → charger les données
    window._pDone=true;
    clearInterval(window._pT); window._pT=null;
    if(ub) ub.textContent='✅ Chargement données...';
    xget('/api/data', function(d){
      if(d && d.prix){
        renderAll(d);
      } else {
        window._pDone=false; // retry
      }
    });
  });

  if(window._pN>200){
    clearInterval(window._pT); window._pT=null;
    if(ub) ub.textContent='⚠️ Timeout — cliquer ⟳ REFRESH';
  }
}

function startPoll(){
  if(window._pT) return;
  window._pT=setInterval(_tick,2000);
  setTimeout(_tick, 500); // premier appel après 500ms
}

// ✅ Tout est défini — démarrer le polling
startPoll();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"[BOOT] GOLD ML v3.0 — port {PORT}")
    threading.Thread(target=compute,   daemon=True).start()
    threading.Thread(target=prix_loop, daemon=True).start()
    threading.Thread(target=mtf_loop,  daemon=True).start()
    print(f"[BOOT] http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
