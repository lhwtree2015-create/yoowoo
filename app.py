import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="웅덩이매매법 V12 스캐너", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp{background:#fff;color:#172033}.block-container{padding-top:1.2rem;max-width:1200px}
h1{font-size:1.8rem!important;font-weight:800!important}
.stButton>button{width:100%;min-height:3.2rem;font-size:1.1rem;font-weight:800;border-radius:12px}
[data-testid="stMetric"]{background:#f7f9fc;border:1px solid #e3e8ef;padding:10px;border-radius:12px}
@media(max-width:600px){.block-container{padding-left:.7rem;padding-right:.7rem}h1{font-size:1.5rem!important}}
</style>""", unsafe_allow_html=True)
st.title("📊 웅덩이매매법 V12 스캐너")
st.caption("나스닥 100 · 일봉 기준 · 수동 검사")

with st.expander("검사 설정", expanded=True):
    years = st.selectbox("가격 데이터 기간", [2, 3, 5], index=0)
    workers = st.slider("동시 조회 수", 2, 12, 6)
    extra = st.checkbox("뉴스·애널리스트 목표가도 조회 (느릴 수 있음)", False)

@st.cache_data(ttl=3600, show_spinner=False)
def get_tickers():
    url = "https://thaywebsearch.github.io/nasdaq100-list/nasdaq100-table.csv"
    r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    cols = [c for c in df.columns if str(c).strip().lower() in ("symbol","ticker","ticker symbol","stock symbol")]
    if not cols:
        cols = [c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()]
    if not cols: raise ValueError("종목 목록에서 티커 열을 찾지 못했습니다.")
    tickers = df[cols[0]].astype(str).str.strip().str.replace(".", "-", regex=False)
    tickers = tickers[~tickers.isin(["", "nan", "None"])].drop_duplicates().tolist()
    if len(tickers) < 80: raise ValueError(f"종목 목록이 너무 적습니다: {len(tickers)}개")
    return sorted(tickers)

def rma(s, n):
    x = pd.Series(s, dtype=float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    a = x.to_numpy()
    if len(a) < n: return out
    if not np.isfinite(a[:n]).all(): return out
    prev = float(np.mean(a[:n])); out.iloc[n-1] = prev
    for i in range(n, len(a)):
        if np.isfinite(a[i]):
            prev = (prev*(n-1)+a[i])/n
            out.iloc[i] = prev
    return out

def rsi_calc(c, n=14):
    delta = c.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag, al = rma(gain, n), rma(loss, n)
    rs = ag / al.replace(0, np.nan)
    ans = 100 - 100/(1+rs)
    ans = ans.mask((al == 0) & (ag > 0), 100)
    ans = ans.mask((ag == 0) & (al > 0), 0)
    ans = ans.mask((ag == 0) & (al == 0), 50)
    return ans

def compute(df):
    d = df.copy().sort_index()
    for k in ["High","Low","Close"]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d = d.dropna(subset=["High","Low","Close"])
    if len(d) < 220: raise ValueError("유효한 일봉 데이터가 220개 미만입니다.")
    c,h,l = d.Close,d.High,d.Low
    for n in [10,20,50,100,200]: d[f"MA{n}"] = c.rolling(n).mean()
    d["RSI"] = rsi_calc(c)
    d["BBBasis"] = c.rolling(20).mean()
    dev = c.rolling(20).std(ddof=0)*2
    d["BBUpper"],d["BBLower"] = d.BBBasis+dev,d.BBBasis-dev
    width = d.BBUpper-d.BBLower
    d["BB%B"] = ((c-d.BBLower)/width.replace(0,np.nan)).fillna(.5)
    lowpb = ((l-d.BBLower)/width.replace(0,np.nan)).fillna(.5)
    near = lambda ma,p: ((c-ma).abs()/ma*100 <= p).fillna(False)
    score = pd.Series(np.select([near(d.MA20,2),near(d.MA50,3),near(d.MA100,4),near(d.MA200,5)],[1,2,3,4],default=0),index=d.index)
    below20 = c < d.MA20
    peak = c.rolling(60).max()
    dd = ((peak-c)/peak.replace(0,np.nan)*100).fillna(0)
    lowdd = ((peak-l)/peak.replace(0,np.nan)*100).fillna(0)
    rebound = ((c-l)/l.replace(0,np.nan)*100).fillna(0)
    panic = (lowdd>=14)&(lowpb<=.15)&(d.RSI<=42)&below20&(rebound>=.5)
    normal = below20&(score>=1)&(d.RSI<=42)&(d["BB%B"]<=.25)
    strong = below20&(score>=1)&(d.RSI<=35)&(d["BB%B"]<=.20)
    crash = ((dd>=17)&(d.RSI<=33)&(d["BB%B"]<=.15)&below20)|panic
    extreme = (dd>=23)&(d.RSI<=28)&(d["BB%B"]<=-.05)&below20

    n=len(d); nb=np.zeros(n,bool); sb=np.zeros(n,bool); cb=np.zeros(n,bool); eb=np.zeros(n,bool)
    lastnormal=None; laststrong=np.nan; lastextreme=np.nan; armed=False; wasbelow=False
    for i in range(n):
        nr,sr,cr,er = [bool(x.iloc[i]) if pd.notna(x.iloc[i]) else False for x in (normal,strong,crash,extreme)]
        prevnr=bool(normal.iloc[i-1]) if i and pd.notna(normal.iloc[i-1]) else False
        nb[i]=nr and not prevnr and (lastnormal is None or i-lastnormal>=12)
        if nb[i]: lastnormal=i
        sb[i]=sr and (pd.isna(laststrong) or c.iloc[i]<=laststrong*.96)
        if sb[i]: laststrong=float(c.iloc[i])
        if c.iloc[i]>d.MA20.iloc[i]: laststrong=np.nan
        if armed and c.iloc[i]<d.MA10.iloc[i]: wasbelow=True
        cross=bool(i and pd.notna(d.MA10.iloc[i]) and pd.notna(d.MA10.iloc[i-1]) and c.iloc[i]>d.MA10.iloc[i] and c.iloc[i-1]<=d.MA10.iloc[i-1])
        first=cr and not armed
        add=armed and wasbelow and cross
        cb[i]=first or add
        if first: armed=True; wasbelow=False
        if add: wasbelow=False
        if c.iloc[i]>d.MA20.iloc[i] and d.RSI.iloc[i]>50: armed=False; wasbelow=False
        eb[i]=er and (pd.isna(lastextreme) or c.iloc[i]<=lastextreme*.96)
        if eb[i]: lastextreme=float(c.iloc[i])
        if c.iloc[i]>d.MA20.iloc[i] and d.RSI.iloc[i]>50: lastextreme=np.nan
    for name,arr in [("웅줍",nb),("강웅줍",sb),("급락웅줍",cb),("급락웅줍+",eb)]: d[name]=arr

    d["상승정렬"]=(d.MA10>d.MA20)&(d.MA20>d.MA50)&(d.MA50>d.MA100)&(d.MA100>d.MA200)
    premium=((c-d.MA20)/d.MA20.replace(0,np.nan)*100).fillna(0)
    heat=((d.RSI>=70).astype(int)+(premium>=8).astype(int)+(d["BB%B"]>=1).astype(int))>=2
    cooling=(d.RSI>=60)&(d.RSI<d.RSI.shift(1))&(d.RSI.shift(1)>=70)
    reentry=(d["BB%B"]<1)&(d["BB%B"].shift(1)>=1)
    highfail=(h<h.rolling(10).max().shift(1))&(c<c.shift(1))
    rsifall=(d.RSI<d.RSI.shift(1))&(d.RSI.shift(1)<d.RSI.shift(2))
    momentum=(cooling.astype(int)+reentry.astype(int)+highfail.astype(int)+rsifall.astype(int))>=2
    m10break=d.MA10<d.MA20; close20=c<d.MA20; m20break=d.MA20<d.MA50; close50=c<d.MA50
    trend=(m10break.astype(int)+close20.astype(int)+m20break.astype(int)*2+close50.astype(int))>=2
    touch=h>=d.BBUpper; cycle=False; cycles=np.zeros(n,bool)
    for i in range(n):
        if bool(touch.iloc[i]) if pd.notna(touch.iloc[i]) else False: cycle=True
        cooling_now=d.RSI.iloc[i]<55 and c.iloc[i]<d.BBBasis.iloc[i]
        recover=d.MA10.iloc[i]>d.MA20.iloc[i] and c.iloc[i]>d.MA20.iloc[i] and d.RSI.iloc[i]<60
        if cooling_now or recover: cycle=False
        cycles[i]=cycle
    firsts=np.zeros(n,bool); seconds=np.zeros(n,bool); thirds=np.zeros(n,bool)
    stage=0; lastsell=None
    for i in range(n):
        cooldown=lastsell is None or i-lastsell>=15
        bull=bool(d["상승정렬"].iloc[i]); ht=bool(heat.iloc[i]); mom=bool(momentum.iloc[i]); tr=bool(trend.iloc[i]); cyc=bool(cycles[i])
        f_bull=bull and ht and mom
        f_other=(not bull) and ((ht and mom) or (ht and tr) or (mom and tr))
        f=cyc and stage<1 and cooldown and (f_bull or f_other)
        s=cyc and stage==1 and cooldown and ((ht and mom and tr) or (bool(m10break.iloc[i]) and bool(close20.iloc[i]) and ht))
        t=stage==2 and cooldown and ((bool(m20break.iloc[i]) and bool(close20.iloc[i])) or (bool(m20break.iloc[i]) and bool(close50.iloc[i])))
        firsts[i],seconds[i],thirds[i]=f,s,t
        if f: stage=1; lastsell=i
        if s: stage=2; lastsell=i
        if t: stage=3; lastsell=i
        if bull and d.RSI.iloc[i]<60 and c.iloc[i]>d.MA20.iloc[i] and c.iloc[i]>d.BBBasis.iloc[i]: stage=0
    d["비중축소"],d["강비중축소"],d["과감한 비중축소"]=firsts,seconds,thirds
    d["과열감시시작"]=touch & ~touch.shift(1,fill_value=False)
    d["고점대비 하락률(%)"]=dd
    return d

def scan_one(ticker, period, extra):
    try:
        df=yf.download(ticker,period=period,interval="1d",auto_adjust=False,progress=False,threads=False)
        if df is None or df.empty: return None, ticker, "가격 데이터 없음", None
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df[~df.index.duplicated(keep="last")]
        now=datetime.now(ZoneInfo("America/New_York"))
        if now.time()<time(16,15) and len(df) and pd.Timestamp(df.index[-1]).date()==now.date(): df=df.iloc[:-1]
        d=compute(df); x=d.iloc[-1]
        names=["웅줍","강웅줍","급락웅줍","급락웅줍+","비중축소","강비중축소","과감한 비중축소"]
        signals=[name for name in names if bool(x[name])]
        prev=d.Close.iloc[-2] if len(d)>1 else np.nan
        row={"종목":ticker,"기준일":pd.Timestamp(d.index[-1]).strftime("%Y-%m-%d"),"종가":round(float(x.Close),2),
             "변동률(%)":round(float((x.Close/prev-1)*100),2) if pd.notna(prev) else np.nan,
             "RSI":round(float(x.RSI),2) if pd.notna(x.RSI) else np.nan,
             "BB %B":round(float(x["BB%B"]),3),"고점대비 하락률(%)":round(float(x["고점대비 하락률(%)"].iloc[-1] if isinstance(x["고점대비 하락률(%)"],pd.Series) else x["고점대비 하락률(%)"]),2),
             "MA20":round(float(x.MA20),2) if pd.notna(x.MA20) else np.nan,"MA50":round(float(x.MA50),2) if pd.notna(x.MA50) else np.nan,
             "MA200":round(float(x.MA200),2) if pd.notna(x.MA200) else np.nan,"상승정렬":bool(x["상승정렬"]),"신호":", ".join(signals) if signals else "신호 없음","_signals":signals}
        detail=None
        if extra:
            detail={"종목":ticker,"뉴스":f"https://news.google.com/search?q={ticker}%20stock","Yahoo":f"https://finance.yahoo.com/quote/{ticker}/","평균목표가":None,"애널리스트 의견":"정보 없음"}
            try:
                info=yf.Ticker(ticker).info
                detail["평균목표가"]=info.get("targetMeanPrice")
                detail["애널리스트 의견"]=str(info.get("recommendationKey","정보 없음")).upper()
            except Exception: pass
        return row,ticker,None,detail
    except Exception as e: return None,ticker,str(e)[:160],None

def run_scan(tickers, years, workers, extra):
    rows=[]; errors=[]; details=[]
    bar=st.progress(0); status=st.empty()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(scan_one,t,f"{years}y",extra) for t in tickers]
        for i,f in enumerate(as_completed(fs),1):
            row,ticker,err,detail=f.result()
            if row: rows.append(row)
            else: errors.append({"종목":ticker,"오류":err})
            if detail: details.append(detail)
            bar.progress(i/len(fs),text=f"조회 중… {i}/{len(fs)}")
            status.caption(f"성공 {len(rows)}개 · 실패 {len(errors)}개")
    bar.empty(); status.empty()
    return pd.DataFrame(rows),pd.DataFrame(errors),pd.DataFrame(details)

if "results" not in st.session_state: st.session_state.results=None
if st.button("🔍 나스닥 100 전체 검사 시작",type="primary",use_container_width=True):
    try:
        with st.spinner("종목 목록을 불러오는 중…"): tickers=get_tickers()
        st.info(f"검사 대상: {len(tickers)}개 종목")
        with st.spinner("가격과 V12 신호를 계산 중입니다…"):
            r,e,dt=run_scan(tickers,years,workers,extra)
        st.session_state.results,st.session_state.errors,st.session_state.details=r,e,dt
        st.success("검사가 끝났습니다.")
    except Exception as e: st.error(f"검사 오류: {e}")

r=st.session_state.results
if r is not None:
    if r.empty: st.warning("유효한 가격 데이터가 없습니다.")
    else:
        buys=["웅줍","강웅줍","급락웅줍","급락웅줍+"]
        sells=["비중축소","강비중축소","과감한 비중축소"]
        bm=r["_signals"].apply(lambda a:any(x in buys for x in a)); sm=r["_signals"].apply(lambda a:any(x in sells for x in a))
        st.subheader("검사 요약")
        a,b,c,d=st.columns(4); a.metric("유효 종목",len(r)); b.metric("매수 신호",int(bm.sum())); c.metric("매도 신호",int(sm.sum())); d.metric("신호 없음",int((r["신호"]=="신호 없음").sum()))
        st.subheader("매수 신호")
        st.dataframe(r[bm].drop(columns="_signals"),use_container_width=True,hide_index=True) if bm.any() else st.info("마지막 확정 일봉에서 매수 신호가 없습니다.")
        st.subheader("매도 신호")
        st.dataframe(r[sm].drop(columns="_signals"),use_container_width=True,hide_index=True) if sm.any() else st.info("마지막 확정 일봉에서 매도 신호가 없습니다.")
        with st.expander("전체 종목 결과"): st.dataframe(r.drop(columns="_signals"),use_container_width=True,hide_index=True)
        st.download_button("결과 CSV 다운로드",r.drop(columns="_signals").to_csv(index=False).encode("utf-8-sig"),"woongdung_v12_scan.csv","text/csv",use_container_width=True)
        dt=st.session_state.get("details")
        if dt is not None and not dt.empty:
            st.subheader("뉴스 · 애널리스트 정보")
            mp=dt.set_index("종목").to_dict("index")
            show=r[r["신호"]!="신호 없음"]["종목"].tolist() or r["종목"].head(10).tolist()
            for t in show:
                q=mp.get(t,{}); target=q.get("평균목표가")
                targettxt=f"${float(target):,.2f}" if target else "정보 없음"
                st.markdown(f"**{t}** · 평균 목표가: {targettxt} · 의견: {q.get('애널리스트 의견','정보 없음')}  \n[Google 뉴스]({q.get('뉴스','#')}) · [Yahoo Finance]({q.get('Yahoo','#')})")
        errors=st.session_state.get("errors")
        if errors is not None and not errors.empty:
            with st.expander(f"조회 실패 종목 ({len(errors)}개)"): st.dataframe(errors,use_container_width=True,hide_index=True)
st.divider()
st.caption("주의: 투자 권유가 아닌 참고용 신호 스캐너입니다. TradingView와 데이터 제공처의 가격·세션·조정 방식 차이로 신호가 다를 수 있습니다. 실제 체결·수수료·슬리피지를 재현하는 전략테스터는 포함하지 않습니다.")
