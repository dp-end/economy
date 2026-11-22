"""MCP araçlarının gövdeleri (PDF Bölüm 22: 15 araç). Sunucu bunları yalnızca kaydeder (server.py).

Her araç:
  - girişi doğrular, geçersiz girişte ToolError ile anlaşılır Türkçe hata verir (çökmez),
  - yalnızca deterministik modül fonksiyonlarını çağırır (LLM hesap yapmaz),
  - 'asof' / 'end' sonrasındaki veriyi döndürmez (geleceğe bakma yok),
  - JSON'a çevrilebilir sözlük döndürür (tarihler 'YYYY-MM-DD', boş değer None).
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from mcp.server.mcpserver.exceptions import ToolError

from src.backtest.engine import COST_ONE_WAY, benchmarks, calculate_performance, load_returns
from src.backtest.engine import run_backtest as _engine_backtest
from src.backtest.strategies import WARMUP
from src.data.pit_fundamentals import pit_asof
from src.data.macro import CPI_RELEASE_DAY, get_macro_context as _macro
from src.data.news_tags import CATEGORY_RULES, get_news_events as _news_query, load_news_events
from src.data.prices import get_index_history as _index, get_stock_history as _stock
from src.data.speech import SOURCES
from src.data.speech_features import get_speech_features as _speech, load_speech_features
from src.data.universe import load_universe
from src.events.technical_events import HARNESS_EVENTS
from src.events.calendar_patterns import WEEKDAYS, _data as _calendar_data, welch
from src.harness.decision_log import dumps, load_analysis
from src.harness.evidence import _variants, analysis_day, run_analysis
from src.indicators.features import full_features
from src.indicators.support_resistance import sr_features

MAX_ROWS = 260                 # bir yanıtta en fazla satır (yaklaşık 1 yıl); fazlası kesilir ve uyarı verilir
MAX_NEWS = 50
INDICATORS = ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "rsi_14", "macd", "macd_signal", "macd_hist",
              "bb_mid", "bb_upper", "bb_lower", "bb_bandwidth", "bb_pctb", "atr_14", "kama", "kama_slope",
              "supertrend", "st_dir", "st_flip", "tenkan", "kijun", "cloud_top", "cloud_bottom", "cloud_state",
              "vol_z_20", "ret_1d", "ret_5d", "vol_20d"]
HORIZONS = (1, 3, 5, 10)


# ---------- ortak doğrulama ----------
def to_json(x):
    """Pandas/numpy değerlerini düz JSON tiplerine çevirir (NaN -> None)."""
    return json.loads(dumps(x).replace("NaN", "null"))


def _ticker(t: str, core_only: bool = False) -> str:
    t = str(t or "").strip().upper()
    u = load_universe()
    allowed = set(u[u["is_core"] == 1]["ticker"]) if core_only else set(u["ticker"])
    if t not in allowed:
        raise ToolError(f"bilinmeyen hisse kodu: '{t}'. Geçerli kodlar: {', '.join(sorted(allowed))}")
    return t


def _date(d, name: str, default=None) -> pd.Timestamp:
    if d in (None, ""):
        if default is None:
            raise ToolError(f"'{name}' gerekli (YYYY-AA-GG)")
        return pd.Timestamp(default)
    try:
        return pd.Timestamp(d).normalize()
    except (ValueError, TypeError):
        raise ToolError(f"'{name}' tarih değil: '{d}' (beklenen biçim YYYY-AA-GG)")


def _range(start, end):
    last = _index("XU100").index[-1]
    s = _date(start, "start", "2021-11-01")
    e = min(_date(end, "end", last), last)
    if s > e:
        raise ToolError(f"start ({s.date()}) end'den ({e.date()}) sonra olamaz")
    return s, e


def _asof(asof):
    last = _index("XU100").index[-1]
    d = _date(asof, "asof", last)
    if d > last:
        raise ToolError(f"asof ({d.date()}) son veri gününden ({last.date()}) sonra; gelecek verisi yok")
    try:
        return analysis_day(d)
    except ValueError as e:
        raise ToolError(str(e))


def _rows(df: pd.DataFrame) -> dict:
    """Satırları kayıt listesine çevirir; MAX_ROWS'u aşarsa son MAX_ROWS satır döner."""
    warn = ""
    if len(df) > MAX_ROWS:
        warn = f"{len(df)} satır vardı, son {MAX_ROWS} satır döndü; daha kısa aralık isteyin"
        df = df.iloc[-MAX_ROWS:]
    d = df.reset_index()
    d = d.rename(columns={d.columns[0]: "date"})
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return dict(rows=to_json(d.to_dict("records")), n_rows=len(d), truncated_warning=warn)


# ---------- 1-2: fiyat ----------
def get_stock_history(ticker: str, start: str = None, end: str = None) -> dict:
    """Günlük High/Low/Close/Volume/Turnover_TL ve kaynak bilgisi."""
    t = _ticker(ticker)
    s, e = _range(start, end)
    df = _stock(t).loc[s:e]
    if df.empty:
        raise ToolError(f"{t} için {s.date()}-{e.date()} aralığında fiyat yok")
    meta = dict(ticker=t, source="İş Yatırım (herkese açık uç nokta), günlük; Open yok",
                start=str(df.index[0].date()), end=str(df.index[-1].date()))
    return dict(meta=meta, **_rows(df))


def get_index_history(index: str = "XU100", start: str = None, end: str = None) -> dict:
    """Kıyas endeksi kapanışı (yalnızca XU100)."""
    if str(index).upper() != "XU100":
        raise ToolError(f"bilinmeyen endeks: '{index}'. Yalnızca XU100 var")
    s, e = _range(start, end)
    df = _index("XU100").loc[s:e]
    return dict(meta=dict(index="XU100", source="İş Yatırım END_DEGER"), **_rows(df))


# ---------- 3-5: teknik ----------
def calculate_indicators(ticker: str, start: str = None, end: str = None, indicators: list[str] = None) -> dict:
    """Seçilen göstergelerin günlük değerleri (varsayılan: hepsi). Göstergeler yalnızca geçmiş veriden hesaplanır."""
    t = _ticker(ticker, core_only=True)
    cols = indicators or INDICATORS
    bad = [c for c in cols if c not in INDICATORS]
    if bad:
        raise ToolError(f"bilinmeyen gösterge: {bad}. Geçerli: {', '.join(INDICATORS)}")
    s, e = _range(start, end)
    df = full_features(t).loc[s:e, ["Close"] + list(cols)]
    return dict(meta=dict(ticker=t, indicators=list(cols)), **_rows(df))


def detect_reversal_events(ticker: str, start: str = None, end: str = None, event: str = None,
                           event_set: str = "all") -> dict:
    """Zaman damgalı teknik olaylar ve olay anındaki gösterge durumları. İleri getiri yalnızca penceresi 'end'
    tarihine kadar kapanmışsa verilir. event_set='harness' yalnızca harness'ın kullandığı 12 türü döndürür."""
    t = _ticker(ticker, core_only=True)
    s, e = _range(start, end)
    te = pd.read_parquet("reports/technical_events.parquet")
    if event_set not in ("all", "harness"):
        raise ToolError("event_set 'all' ya da 'harness' olmalı")
    if event_set == "harness":
        te = te[te["event"].isin(HARNESS_EVENTS)]
    if event and event not in set(te["event"]):
        raise ToolError(f"bilinmeyen olay: '{event}'. Geçerli: {', '.join(sorted(te['event'].unique()))}")
    ev = te[(te["ticker"] == t) & te["date"].between(s, e) & ((te["event"] == event) if event else True)].copy()
    days = _index("XU100").index
    for h in HORIZONS:
        closes = days[np.minimum(days.searchsorted(ev["date"]) + h, len(days) - 1)]
        open_ = closes > e                                                # pencere 'end' sonrasına taşıyor
        ev.loc[open_, [f"fwd_{h}", f"exc_{h}"]] = np.nan
    summary = (ev.groupby("event").agg(n=("date", "size"), direction=("direction", "first"),
                                       hit_5=("exc_5", lambda x: (x.dropna() > 0).mean() if x.notna().any() else None),
                                       mean_exc_5=("exc_5", "mean"))
               .reset_index().to_dict("records"))
    ev["date"] = ev["date"].dt.strftime("%Y-%m-%d")
    return dict(meta=dict(ticker=t, start=str(s.date()), end=str(e.date()), event_set=event_set,
                          note="exc_h = h gün sonra XU100'e göre fazla getiri"),
                events=to_json(ev.tail(MAX_ROWS).to_dict("records")), summary=to_json(summary), n_events=len(ev))


def find_support_resistance(ticker: str, date: str = None, tolerance: float = 0.015) -> dict:
    """Analiz gününde (o güne kadar onaylanmış pivotlardan) en yakın destek/direnç, dokunuş sayısı ve yaş."""
    t = _ticker(ticker, core_only=True)
    if not 0.001 <= float(tolerance) <= 0.1:
        raise ToolError("tolerance 0,001 ile 0,1 arasında olmalı")
    d = _asof(date)
    df = _stock(t).loc[:d]
    sr = sr_features(df, tol=float(tolerance)).iloc[-1]
    close = float(df["Close"].iloc[-1])
    out = {k: sr[k] for k in sr.index}
    out.update(ticker=t, date=str(d.date()), close=close, tolerance=float(tolerance),
               dist_support=(close / out["support"] - 1) if pd.notna(out.get("support")) else None,
               dist_resistance=(out["resistance"] / close - 1) if pd.notna(out.get("resistance")) else None)
    return to_json(out)


# ---------- 6-7: sektör ve takvim ----------
def rank_sector_relative_strength(sector: str, horizon: int = 10, date: str = None) -> dict:
    """Akran grubu içinde getiri, akran ortancasına göre gecikme (kendisi hariç) ve sıra."""
    u = load_universe()
    groups = sorted(u["peer_group"].unique())
    match = [g for g in groups if g.lower() == str(sector).strip().lower()]
    if not match:
        raise ToolError(f"bilinmeyen sektör grubu: '{sector}'. Geçerli: {', '.join(groups)}")
    if int(horizon) not in (5, 10, 20):
        raise ToolError("horizon 5, 10 ya da 20 olmalı")
    g, h, d = match[0], int(horizon), _asof(date)
    members = u[u["peer_group"] == g]["ticker"].tolist()
    ret = pd.Series({m: _stock(m)["Close"].loc[:d].pipe(lambda s: s.iloc[-1] / s.iloc[-1 - h] - 1) for m in members})
    rows = []
    for m in members:
        peers = ret.drop(m)
        lag = ret[m] - peers.median() if len(peers) else None
        rows.append(dict(ticker=m, core=bool(u.set_index("ticker").loc[m, "is_core"]), ret=ret[m], lag=lag))
    out = pd.DataFrame(rows).sort_values("ret", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    warn = f"grup küçük ({len(members)} hisse), ortanca zayıf" if len(members) < 3 else ""
    return to_json(dict(sector=g, horizon=h, date=str(d.date()), sector_median=ret.median(),
                        members=out.to_dict("records"), warning=warn))


def test_weekday_pattern(ticker: str, pattern: str = "weekday", start: str = None, end: str = None) -> dict:
    """Haftanın günü ya da art arda düşüş/yükseliş sonrası getiri. Koşulsuz getiriyle karşılaştırma,
    Welch t testi, ilk/ikinci yarı tutarlılığı ve maliyet notu. pattern: 'weekday', 'down_streak:3', 'up_streak:2'."""
    cal = _calendar_data()
    if str(ticker).upper() == "PORTFOLIO":
        name, r = "PORTFOLIO", cal["ret"].mean(axis=1)
    else:
        name = _ticker(ticker, core_only=True)
        r = cal["ret"][name]
    s, e = _range(start, end)
    valid = cal["valid"].loc[s:e]
    r = r.loc[s:e]
    half = r.index[len(r) // 2]

    def compare(mask):
        a, b = r[valid & mask].dropna(), r[valid & ~mask].dropna()
        if len(a) < 10:
            return dict(n=len(a), warning="az gözlem")
        t, p = welch(a, b)
        f, sn = r.index < half, r.index >= half
        diff = lambda m: (r[valid & mask & m].mean() - r[valid & ~mask & m].mean()) * 100
        return dict(n=len(a), mean=a.mean() * 100, median=a.median() * 100, unconditional=b.mean() * 100,
                    diff_pt=(a.mean() - b.mean()) * 100, t=t, p=p, first_half_diff=diff(f), second_half_diff=diff(sn),
                    same_sign_halves=bool(np.sign(diff(f)) == np.sign(diff(sn))))

    if pattern == "weekday":
        dow = pd.Series(r.index.day_of_week, index=r.index)
        res = {WEEKDAYS[d]: compare(dow == d) for d in range(5)}
        tests = 5
    else:
        try:
            kind, k = pattern.split(":")
            k = int(k)
            assert kind in ("down_streak", "up_streak") and 1 <= k <= 6
        except (ValueError, AssertionError):
            raise ToolError("pattern: 'weekday', 'down_streak:K' ya da 'up_streak:K' (K 1-6)")
        sign = (r < 0) if kind == "down_streak" else (r > 0)
        run = sign.astype(int).groupby((~sign).cumsum()).cumsum()
        res = {f"{kind}:{k} sonrası ertesi gün": compare(run.shift(1) >= k)}
        tests = 1
    return to_json(dict(ticker=name, pattern=pattern, start=str(s.date()), end=str(e.date()), results=res,
                        tests_in_this_call=tests,
                        warnings=["Çoklu test riski: birçok desen denenirse şansla anlamlı görünen çıkar; "
                                  "Adım 6'da 150 hisse-gün testinden FDR sonrası 6 kaldı.",
                                  f"Gidiş-dönüş maliyet ~%{2 * COST_ONE_WAY * 100:.2f}; günlük farklar çoğu zaman bundan küçük."]))


# ---------- 8-9: bilanço ----------
FUND_NOTE = ("Değerler analiz gününde bilinebilenlerdir: enflasyon muhasebesi (TMS 29) uygulayan şirketlerin sonradan "
             "düzeltilmiş 2022/4-2023/3 rakamları ancak yayınlandıkları günden sonra kullanılır; yıllık ve çeyreklik "
             "değişimler aynı birimdeki iki rakamla hesaplanır. TL tutarları analiz gününde görünen son raporun "
             "parasıyla (2022/9 ve öncesi nominal); birimler unit_flow (gelir) ve unit_stock (bilanço) alanlarında, "
             "ör. 'TL@2026-09' Eylül 2026 TL'si. Boş değerlerin nedeni pit_dropped alanında. tercile: açıklama "
             "günündeki sınıf; eşikler bir önceki çeyrekte o güne kadar açıklanmış en az 15 hissenin yıllık net kâr "
             "değişiminin 1/3 ve 2/3 noktaları (sektör içi değil). Tanımlar: FAVÖK = brüt kâr − genel yönetim − "
             "pazarlama − Ar-Ge giderleri + amortisman (şirketlerin açıkladığı tanım); net borç = finansal borç − "
             "nakit − kısa vadeli finansal yatırım; *_ttm son 12 ay; roa ve roe_ttm son 12 ay net kâr / dönem sonu "
             "varlık ve özkaynak; nakit dönüşümü = işletme nakit akışı / FAVÖK. Banka, sigorta, aracı kurum ve KCHOL "
             "(banka konsolide eden holding) için FAVÖK, net borç, likidite ve nakit akışı hesaplanmaz. valuation: F/K = "
             "piyasa değeri / son 12 ay net kâr, PD/DD = piyasa değeri / ana ortaklık özkaynağı, FD/FAVÖK = (piyasa "
             "değeri + net borç + azınlık payları) / son 12 ay FAVÖK; kâr, FAVÖK ve özkaynak analiz gününe kadar "
             "yayınlanmış son TÜFE ile o günün parasına taşınır; pe_reported raporun parasıyla (ekranda görünen); "
             "payda ≤ 0 ise boş.")


@lru_cache(maxsize=1)
def _valuation() -> pd.DataFrame:
    return pd.read_parquet("reports/valuation_daily.parquet")


def _visible_quarters(t: str, d: pd.Timestamp) -> pd.DataFrame:
    """Analiz gününe kadar açıklanmış çeyrekler: o gün bilinebilen değerler (src/data/pit_fundamentals.pit_asof;
    TL tutarları o gün görünen son raporun parasıyla) + açıklama gününde hesaplanan 30 hisse içi sıralama."""
    q = pit_asof(t, d)
    if q.empty:
        return q
    ev = pd.read_parquet("reports/earnings_events.parquet")
    ev = ev.loc[ev["ticker"] == t, ["period_end", "publish_time", "tercile", "financial"]]
    return q.merge(ev, on="period_end", how="left").sort_values("period_end").reset_index(drop=True)


def get_fundamental_metrics(ticker: str, period: str = None, asof: str = None) -> dict:
    """Analiz gününde kamuya açık olan son bilanço (ya da istenen dönem) ve yayın zamanı."""
    t = _ticker(ticker, core_only=True)
    d = _asof(asof)
    q = _visible_quarters(t, d)
    if q.empty:
        raise ToolError(f"{t} için {d.date()} itibarıyla açıklanmış bilanço yok")
    if period:
        p = _date(period, "period")
        q = q[q["period_end"] == p]
        if q.empty:
            raise ToolError(f"{t} {p.date()} dönemi {d.date()} itibarıyla açıklanmamış ya da veride yok")
    r = q.iloc[-1]
    cols = ["period_end", "publish_time", "event_day", "net_income", "revenue", "gross_profit", "op_income", "equity",
            "total_assets", "ni_yoy", "ni_qoq", "roe_q", "leverage", "net_margin", "loss", "tercile", "financial",
            # Adım 1c
            "ebitda", "ebitda_ttm", "revenue_ttm", "ocf_ttm", "fcf_ttm", "net_debt", "roa", "roe_ttm",
            "ebitda_margin_ttm", "ebitda_yoy", "nd_ebitda", "debt_to_equity", "current_ratio", "quick_ratio",
            "cash_conversion",
            "unit_flow", "unit_stock", "pit_dropped"]
    out = {c: r[c] for c in cols if c in r.index}
    # Adım 1d: analiz günündeki değerleme (kâr ve özkaynak o gün yayınlanmış son TÜFE ile o günün parasına taşınmış)
    v = _valuation()
    v = v[(v["ticker"] == t) & (v["date"] <= d)]
    if len(v) and not period:
        last = v.iloc[-1]
        out["valuation"] = {k: last[k] for k in ("date", "market_cap", "pe", "pb", "ev_ebitda", "pe_reported",
                                                 "cpi_month", "val_dropped")}
    out.update(ticker=t, asof=str(d.date()), source="Fintables (elle indirilen Excel), yayın zamanı KAP",
               note="Banka/sigorta/aracı kurumda ciro ve brüt kâr yoktur; ortak ölçüler net kâr, özkaynak, toplam varlık. "
                    + FUND_NOTE)
    return to_json(out)


def calculate_fundamental_change(ticker: str, periods: int = 4, asof: str = None) -> dict:
    """Son 'periods' açıklanmış çeyreğin çeyreklik ve yıllık değişimleri (yalnızca analiz gününde görünür olanlar)."""
    t = _ticker(ticker, core_only=True)
    if not 1 <= int(periods) <= 20:
        raise ToolError("periods 1 ile 20 arasında olmalı")
    d = _asof(asof)
    q = _visible_quarters(t, d).tail(int(periods))
    if q.empty:
        raise ToolError(f"{t} için {d.date()} itibarıyla açıklanmış bilanço yok")
    q = q.assign(roe_change=q["roe_q"].diff(), margin_change=q["net_margin"].diff() if "net_margin" in q else np.nan)
    cols = ["period_end", "event_day", "net_income", "ni_yoy", "ni_qoq", "roe_q", "roe_change", "net_margin",
            "margin_change", "leverage", "loss", "tercile",
            "ebitda", "ebitda_yoy", "ebitda_margin_q", "nd_ebitda", "current_ratio", "cash_conversion", "roa",
            "unit_flow", "unit_stock", "pit_dropped"]
    return to_json(dict(ticker=t, asof=str(d.date()), quarters=q[[c for c in cols if c in q]].to_dict("records"),
                        note=FUND_NOTE))


# ---------- 10-12: makro, haber, konuşma ----------
def get_macro_context(start: str = None, end: str = None) -> dict:
    """Kur, faiz, enflasyon ve Fed aralığı; her gün yalnızca o gün yayınlanmış değerler."""
    s, e = _range(start, end)
    df = _macro(s, e)
    meta = dict(source="TCMB EVDS (USD/TRY, EUR/TRY, politika faizi, TÜFE), Federal Reserve (hedef aralık)",
                publication=f"TÜFE izleyen ayın {CPI_RELEASE_DAY}'ünde görünür; kur ve faiz kendi gününde; "
                            "Fed kararı ertesi gün görünür")
    return dict(meta=meta, **_rows(df))


def get_news_events(ticker: str = None, topic: str = None, start: str = None, end: str = None,
                    asof: str = None) -> dict:
    """KAP ve RSS haber olayları (yalnızca başlık/özet, kaynak bağlantısı ve zaman)."""
    t = _ticker(ticker, core_only=True) if ticker else None
    if topic and topic not in CATEGORY_RULES and topic != "diger":
        raise ToolError(f"bilinmeyen konu: '{topic}'. Geçerli: {', '.join(list(CATEGORY_RULES) + ['diger'])}")
    s = _date(start, "start", "2021-11-01")
    e = _date(end, "end", pd.Timestamp.today().normalize())
    a = _asof(asof) if asof else None
    n = _news_query(load_news_events(), ticker=t, topic=topic, start=s, end=e, asof=a)
    total = len(n)
    n = n.tail(MAX_NEWS)
    cols = ["source", "publish_time", "event_day", "ticker", "scope", "category", "categories", "title", "summary", "url"]
    return to_json(dict(n_total=total, n_returned=len(n), events=n[cols].to_dict("records"),
                        note="Haber bağlamdır; sayısal veri yerine geçmez. Kaynak ve zaman her kayıtta.",
                        warning="haber kapsamı seyrek" if total < 3 else ""))


def get_speech_features(source_id: str = None) -> dict:
    """Konuşma kaynağının konu, duruş, belirsizlik ve doğrulanmış/doğrulanmamış iddiaları. source_id yoksa liste."""
    if not source_id:
        return dict(sources=[dict(source_id=k, speaker=v["speaker"], publish_time=v["publish_time"], kind=v["kind"])
                             for k, v in SOURCES.items()])
    if source_id not in SOURCES:
        raise ToolError(f"bilinmeyen konuşma kaynağı: '{source_id}'. Geçerli: {', '.join(SOURCES)}")
    return to_json(_speech(load_speech_features(), source_id))


# ---------- 13-15: backtest, kalite, kanıt ----------
def run_backtest(strategy: str, start: str = None, end: str = None, cost: float = COST_ONE_WAY) -> dict:
    """Strateji varyantının dönem performansı (sinyal 1 gün kaydırılır, maliyet dahil) ve işlem özeti."""
    variants = _variants()
    if strategy not in variants:
        raise ToolError(f"bilinmeyen strateji: '{strategy}'. Geçerli: {', '.join(variants)}")
    if not 0 <= float(cost) <= 0.01:
        raise ToolError("cost (tek yön) 0 ile 0,01 arasında olmalı")
    s, e = _range(start, end)
    ret = load_returns().iloc[WARMUP:].loc[s:e]
    if len(ret) < 20:
        raise ToolError(f"backtest için en az 20 işlem günü gerekli (ısınma sonrası ilk gün {load_returns().index[WARMUP].date()})")
    bt = _engine_backtest(variants[strategy].reindex(ret.index), ret, cost=float(cost))
    bm = benchmarks(ret)
    perf = calculate_performance(bt, bm["esit_agirlik"])
    perf["xu100_getiri"] = float((1 + bm["xu100"].fillna(0)).prod() - 1)
    perf["xu100_farki"] = perf["toplam_getiri"] - perf["xu100_getiri"]          # model hesaplamasın, araç hesaplar
    perf["esit_agirlik_getiri"], perf["esit_agirlik_farki"] = perf.pop("kiyas_getiri"), perf.pop("fark")
    w = variants[strategy].reindex(ret.index)
    last = w.iloc[-1]
    return to_json(dict(strategy=strategy, start=str(ret.index[0].date()), end=str(ret.index[-1].date()),
                        cost_one_way=float(cost), performance=perf,
                        last_positions=last[last > 0].round(4).to_dict(),
                        note="Sinyal t kapanışında, işlem t+1 getirisine uygulanır; universe 30 çekirdek hisse. "
                             "Getiriler oran (2.41 = %241). esit_agirlik = 30 hissenin eşit ağırlıklı al-tut kıyası."))


def check_data_quality(ticker: str, asof: str = None) -> dict:
    """Analiz paketi için kalite kapısı: kesin engeller, güvensiz durumlar, yumuşak uyarılar."""
    t = str(ticker or "").strip().upper()
    d = _asof(asof)
    r = run_analysis(t, d)
    return to_json(dict(ticker=t, asof=str(d.date()), **r["gate"]))


def build_evidence_bundle(ticker: str = None, asof: str = None, analysis_id: str = None) -> dict:
    """Tam gözlem seti + committed evidence + kurallı eylem + kapı. analysis_id verilirse günlükten okunur."""
    if analysis_id:
        try:
            a = load_analysis(analysis_id)
        except ValueError as e:
            raise ToolError(str(e))
        return to_json(dict(analysis_id=analysis_id, ticker=a["ticker"], asof=a["asof"], action=a["action"],
                            secondary=a["secondary"], gate=a["gate"], committed=a["committed"],
                            observed=a["observed"], from_log=True))
    t = _ticker(ticker, core_only=True)
    d = _asof(asof)
    r = run_analysis(t, d)
    return to_json(dict(ticker=t, asof=str(d.date()), action=r["action"], secondary=r["secondary"], gate=r["gate"],
                        committed=r["committed"], observed=r["observed"], history=r["history"],
                        backtest=r["backtest"], warning=r["warning"], from_log=False))


TOOLS = [get_stock_history, get_index_history, calculate_indicators, detect_reversal_events, find_support_resistance,
         rank_sector_relative_strength, test_weekday_pattern, get_fundamental_metrics, calculate_fundamental_change,
         get_macro_context, get_news_events, get_speech_features, run_backtest, check_data_quality,
         build_evidence_bundle]
