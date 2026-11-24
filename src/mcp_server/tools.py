"""MCP araçlarının gövdeleri (PDF Bölüm 22: 15 araç). Sunucu bunları yalnızca kaydeder (server.py).

Her araç:
  - girişi doğrular, geçersiz girişte ToolError ile anlaşılır Türkçe hata verir (çökmez),
  - yalnızca deterministik modül fonksiyonlarını çağırır (LLM hesap yapmaz),
  - 'asof' / 'end' sonrasındaki veriyi döndürmez (geleceğe bakma yok),
  - harness bir tarih tavanı koyduysa (analiz tarihi) o günden sonrasını hiç göstermez; daha ileri bir tarih
    istenirse tavana kırpar ve çıktıya 'date_warnings' yazar. Tavanı model değiştiremez (araç parametresi değil),
  - JSON'a çevrilebilir sözlük döndürür (tarihler 'YYYY-MM-DD', boş değer None).
"""
import functools
import json
import os
import re
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import pandas as pd
from mcp.server.mcpserver.exceptions import ToolError

from src.backtest.engine import COST_ONE_WAY, benchmarks, calculate_performance, load_returns, sector_neutral, trade_log
from src.backtest.engine import run_backtest as _engine_backtest
from src.backtest.strategies import WARMUP
from src.data.pit_fundamentals import pit_asof
from src.data.macro import CPI_RELEASE_DAY, get_macro_context as _macro
from src.data.news_tags import CATEGORY_RULES, get_news_events as _news_query, load_news_events
from src.data.prices import get_index_history as _index, get_stock_history as _stock
from src.data.speech import SOURCES
from src.data.speech_features import get_speech_features as _speech, load_speech_features
from src.data.universe import load_universe
from src.events.technical_events import DIRECTION, HARNESS_EVENTS, TOUCH_DIRECTION, touch_table
from src.events.calendar_patterns import WEEKDAYS, _data as _calendar_data, welch
from src.events.sector_catchup import catchup_history, catchup_summary
from src.indicators.relative import sector_lag
from src.harness.decision_log import dumps, load_analysis
from src.harness.evidence import _variants, analysis_day, run_analysis, speech_known_day
from src.indicators.features import full_features
from src.indicators.support_resistance import sr_features

MAX_ROWS = 260                 # bir yanıtta en fazla satır (yaklaşık 1 yıl); fazlası kesilir ve uyarı verilir
MAX_NEWS = 50
INDICATORS = ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "rsi_14", "macd", "macd_signal", "macd_hist",
              "bb_mid", "bb_upper", "bb_lower", "bb_bandwidth", "bb_pctb", "atr_14", "kama", "kama_slope",
              "supertrend", "st_dir", "st_flip", "tenkan", "kijun", "cloud_top", "cloud_bottom", "cloud_state",
              "vol_z_20", "ret_1d", "ret_5d", "vol_20d"]
HORIZONS = (1, 3, 5, 10)
MIN_EVENTS = 10        # bundan az olayda örneklem küçük uyarısı (catch-up ve olay koşul özeti)
CEILING_ENV = "BIST_DATA_CEILING"


# ---------- tarih tavanı (analiz tarihi) ----------
_CEILING = None        # with data_ceiling(...) ile konur; süreç içi tek analiz olduğu için düz değişken yeterli
_NOTES = []            # o anki araç çağrısının tarih uyarıları


@contextmanager
def data_ceiling(day):
    """Harness'ın tavanı: bu blok içinde hiçbir araç 'day' sonrasındaki veriyi göstermez.
    Örnek: with data_ceiling("2026-09-10"): ...  (ayrı süreçteki MCP sunucusu için BIST_DATA_CEILING)."""
    global _CEILING
    old, _CEILING = _CEILING, (pd.Timestamp(day).normalize() if day else None)
    try:
        yield
    finally:
        _CEILING = old


def _active_ceiling():
    """Konulmuş tavan (takvim günü) ya da None. Öncelik: data_ceiling bloğu > ortam değişkeni."""
    if _CEILING is not None:
        return _CEILING
    env = os.environ.get(CEILING_ENV, "").strip()
    return pd.Timestamp(env).normalize() if env else None


def ceiling() -> pd.Timestamp:
    """Araçların görebileceği son işlem günü: tavan varsa ondan önceki son işlem günü, yoksa son veri günü."""
    last = _index("XU100").index[-1]
    c = _active_ceiling()
    if c is None or c >= last:
        return last
    try:
        return analysis_day(c)
    except ValueError as e:
        raise ToolError(str(e))


def _note(msg: str):
    _NOTES.append(msg)


def _guarded(fn):
    """Her aracın sarmalı: uyarıları sıfırlar; tavan varsa çıktıya data_ceiling ve date_warnings ekler.
    Tavan yoksa çıktı eskisiyle aynıdır."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _NOTES.clear()
        out = fn(*args, **kwargs)
        if _active_ceiling() is not None and isinstance(out, dict):
            out = dict(out, data_ceiling=str(ceiling().date()), date_warnings=list(_NOTES))
        return out
    return wrapper


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
    top = ceiling()
    s = _date(start, "start", "2021-11-01")
    e = _date(end, "end", top)
    if e > top:
        if _active_ceiling() is not None:
            _note(f"istenen 'end' analiz tarihinden sonraydı; {top.date()} tarihine kırpıldı")
        e = top
    if s > e:
        if _active_ceiling() is not None and s > top:
            raise ToolError(f"start ({s.date()}) analiz tarihinden ({top.date()}) sonra; bu dönemin verisi yok")
        raise ToolError(f"start ({s.date()}) end'den ({e.date()}) sonra olamaz")
    return s, e


def _asof(asof, name: str = "asof"):
    top = ceiling()
    d = _date(asof, name, top)
    if d > top:
        if _active_ceiling() is None:
            raise ToolError(f"{name} ({d.date()}) son veri gününden ({top.date()}) sonra; gelecek verisi yok")
        _note(f"istenen '{name}' analiz tarihinden sonraydı; {top.date()} tarihine kırpıldı")
        d = top
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
@_guarded
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


@_guarded
def get_index_history(index: str = "XU100", start: str = None, end: str = None) -> dict:
    """Kıyas endeksi kapanışı (yalnızca XU100)."""
    if str(index).upper() != "XU100":
        raise ToolError(f"bilinmeyen endeks: '{index}'. Yalnızca XU100 var")
    s, e = _range(start, end)
    df = _index("XU100").loc[s:e]
    return dict(meta=dict(index="XU100", source="İş Yatırım END_DEGER"), **_rows(df))


# ---------- 3-5: teknik ----------
@_guarded
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


# Koşul dili (olay günü kapanışındaki durum, src/events/technical_events.states ile aynı tanımlar). '+' = ve.
COND_TOKENS = {
    "rsi_dip": ("rsi_state", "dip"), "rsi_tepe": ("rsi_state", "tepe"), "rsi_notr": ("rsi_state", "notr"),
    "hacim_yuksek": ("vol_recent", True), "hacim_normal": ("vol_recent", False),
    "st_yukari": ("st_state", "yukari"), "st_asagi": ("st_state", "asagi"),
    "bulut_ustu": ("cloud_pos", "ustunde"), "bulut_alti": ("cloud_pos", "altinda"), "bulut_ici": ("cloud_pos", "icinde"),
    "kama_yukselen": ("kama_state", "yukselen"), "kama_dusen": ("kama_state", "dusen"), "kama_yatay": ("kama_state", "yatay"),
    "trend_yukari": ("regime", "uptrend"), "trend_asagi": ("regime", "downtrend"),
}
DEFAULT_CONDITIONS = {1: ["rsi_dip", "hacim_yuksek", "rsi_dip+hacim_yuksek", "st_asagi", "bulut_alti", "st_asagi+bulut_alti"],
                      -1: ["rsi_tepe", "hacim_yuksek", "rsi_tepe+hacim_yuksek", "st_yukari", "bulut_ustu",
                           "st_yukari+bulut_ustu"]}
COND_NOTE = ("Koşullar olay günü kapanışındaki durum: rsi_dip/rsi_tepe = son 3 günde RSI(14) ≤ 35 / ≥ 65; hacim_yuksek = "
             "son 3 günde hacim z-skoru > 1; st_* = Supertrend yönü; bulut_* = fiyatın Ichimoku bulutuna göre yeri; "
             "kama_* = 5 günlük KAMA değişimi > +%0,5 / < −%0,5 / arası; trend_* = kapanış SMA200 üstünde/altında. "
             "success: h. gün kapanışı olay günü kapanışından yüksek (yükseliş olayı; düşüş olayında düşük). "
             "mean_pct/median_pct ham getiri, mean_exc_pct XU100'e göre fazla getiri (yüzde). n_incomplete: penceresi "
             "analiz tarihini aşan, henüz bilinmediği için istatistiğe katılmayan olaylar.")


def _cond_mask(ev: pd.DataFrame, cond: str) -> pd.Series:
    m = pd.Series(True, index=ev.index)
    for tok in str(cond).split("+"):
        tok = tok.strip().lower()
        if tok not in COND_TOKENS:
            raise ToolError(f"bilinmeyen koşul: '{tok}'. Geçerli: {', '.join(COND_TOKENS)} ('+' ile birleştirilir)")
        col, val = COND_TOKENS[tok]
        m &= ev[col] == val
    return m


def _by_condition(ev: pd.DataFrame, direction: int, conditions: list) -> list:
    """Her koşul grubu için ufuk başına başarı/başarısızlık sayısı ve ortalama/medyan dönüş gücü (yüzde)."""
    groups = [("hepsi", pd.Series(True, index=ev.index))] + [(c, _cond_mask(ev, c)) for c in conditions]
    out = []
    for name, m in groups:
        g, hs = ev[m], []
        for h in HORIZONS:
            x, xe = g[f"fwd_{h}"].dropna().astype(float), g[f"exc_{h}"].dropna().astype(float)
            ok = int((direction * x > 0).sum())
            hs.append(dict(horizon_days=h, n=int(len(x)), n_incomplete=int(g[f"fwd_{h}"].isna().sum()),
                           success=ok, failure=int(len(x)) - ok,
                           mean_pct=x.mean() * 100 if len(x) else None, median_pct=x.median() * 100 if len(x) else None,
                           mean_exc_pct=xe.mean() * 100 if len(xe) else None))
        row = dict(condition=name, n_events=int(len(g)), horizons=hs)
        n5 = hs[HORIZONS.index(5)]["n"]
        if n5 < MIN_EVENTS:
            row["warning"] = f"5 günlük penceresi tamamlanmış yalnızca {n5} olay (< {MIN_EVENTS}); sonuç zayıf"
        out.append(row)
    return out


@_guarded
def detect_reversal_events(ticker: str, start: str = None, end: str = None, event: str = None,
                           event_set: str = "all", conditions: list[str] = None) -> dict:
    """Zaman damgalı teknik olaylar ve olay anındaki gösterge durumları. İleri getiri yalnızca penceresi 'end'
    tarihine kadar kapanmışsa verilir. event_set='harness' yalnızca harness'ın kullandığı 12 türü döndürür.
    event='bb_lower_touch'/'bb_upper_touch': gün içinde banda dokunma (seri başı, en az 5 gün arayla). Tek bir event
    verilince by_condition: tekli ve birleşik koşullara göre başarı sayısı ve 1/3/5/10 günlük dönüş gücü;
    conditions ör. ['rsi_dip+hacim_yuksek', 'st_asagi+bulut_alti'] (verilmezse varsayılan set)."""
    t = _ticker(ticker, core_only=True)
    s, e = _range(start, end)
    if event_set not in ("all", "harness"):
        raise ToolError("event_set 'all' ya da 'harness' olmalı")
    te = pd.read_parquet("reports/technical_events.parquet")
    known = set(te["event"]) | (set(TOUCH_DIRECTION) if event_set == "all" else set())
    if event_set == "harness":
        known &= set(HARNESS_EVENTS)
    if event and event not in known:
        raise ToolError(f"bilinmeyen olay: '{event}'. Geçerli: {', '.join(sorted(known))}")
    if conditions and not event:
        raise ToolError("conditions için tek bir event verin (ör. event='bb_lower_touch')")
    te = te[(te["ticker"] == t) & te["event"].isin(known)]
    touches, touch_days = touch_table(t)
    if event_set == "all":
        te = pd.concat([te, touches.reset_index()], ignore_index=True)
    ev = te[te["date"].between(s, e) & ((te["event"] == event) if event else True)].copy()
    days = _index("XU100").index
    for h in HORIZONS:
        closes = days[np.minimum(days.searchsorted(ev["date"]) + h, len(days) - 1)]
        open_ = closes > e                                                # pencere 'end' sonrasına taşıyor
        ev.loc[open_, [f"fwd_{h}", f"exc_{h}"]] = np.nan
    summary = (ev.groupby("event").agg(n=("date", "size"), direction=("direction", "first"),
                                       hit_5=("exc_5", lambda x: (x.dropna() > 0).mean() if x.notna().any() else None),
                                       mean_exc_5=("exc_5", "mean"))
               .reset_index().to_dict("records"))
    meta = dict(ticker=t, start=str(s.date()), end=str(e.date()), event_set=event_set,
                note="exc_h = h gün sonra XU100'e göre fazla getiri")
    extra = {}
    if event:
        direction = TOUCH_DIRECTION.get(event, DIRECTION.get(event))
        extra["by_condition"] = to_json(_by_condition(ev, direction, conditions or DEFAULT_CONDITIONS[direction]))
        extra["condition_note"] = COND_NOTE
        if event in TOUCH_DIRECTION:
            meta["touch_days"] = int(touch_days.loc[s:e, event].sum())     # bilgi: her gün ayrı sayılsaydı
            meta["touch_rule"] = ("dokunuş: gün içi en düşük ≤ alt bant (üst: en yüksek ≥ üst bant); olay = art arda "
                                  "dokunulan günlerin ilki, olaylar arası en az 5 işlem günü")
    ev["date"] = ev["date"].dt.strftime("%Y-%m-%d")
    return dict(meta=meta, events=to_json(ev.tail(MAX_ROWS).to_dict("records")), summary=to_json(summary),
                n_events=len(ev), **extra)


@_guarded
def find_support_resistance(ticker: str, date: str = None, tolerance: float = 0.015) -> dict:
    """Analiz gününde (o güne kadar onaylanmış pivotlardan) en yakın destek/direnç, dokunuş sayısı ve yaş."""
    t = _ticker(ticker, core_only=True)
    if not 0.001 <= float(tolerance) <= 0.1:
        raise ToolError("tolerance 0,001 ile 0,1 arasında olmalı")
    d = _asof(date, "date")
    df = _stock(t).loc[:d]
    sr = sr_features(df, tol=float(tolerance)).iloc[-1]
    close = float(df["Close"].iloc[-1])
    out = {k: sr[k] for k in sr.index}
    out.update(ticker=t, date=str(d.date()), close=close, tolerance=float(tolerance),
               dist_support=(close / out["support"] - 1) if pd.notna(out.get("support")) else None,
               dist_resistance=(out["resistance"] / close - 1) if pd.notna(out.get("resistance")) else None)
    return to_json(out)


# ---------- 6-7: sektör ve takvim ----------
# Türkçe sektör adları (soru Türkçe sorulur). Karşılaştırma büyük/küçük harf ve Türkçe karaktere duyarsız.
SECTOR_ALIASES = {
    "Automotive & Durables": ["otomotiv", "dayanıklı tüketim", "beyaz eşya"],
    "Banking": ["banka", "bankalar", "bankacılık"],
    "Chemicals & Energy": ["kimya", "petrokimya", "rafineri", "kimya ve enerji"],
    "Construction RE & Utilities": ["inşaat", "gayrimenkul", "gyo", "elektrik", "kamu hizmetleri"],
    "Healthcare": ["sağlık", "ilaç"],
    "Metals & Mining": ["metal", "demir çelik", "çelik", "madencilik", "maden", "metal ve madencilik"],
    "Other Financials": ["holding", "sigorta", "aracı kurum", "diğer finansal"],
    "Retail & Consumer": ["perakende", "tüketim", "gıda", "gıda ve içecek"],
    "Tech Telecom & Defense": ["teknoloji", "telekom", "telekomünikasyon", "savunma"],
    "Transportation": ["ulaştırma", "ulaşım", "havayolu", "havacılık"],
}


def _fold(s: str) -> str:
    s = str(s).strip().replace("İ", "i").replace("I", "ı").lower()
    return s.translate(str.maketrans("çğıöşü", "cgiosu"))


def _peer_group(sector: str = None, tickers: list = None) -> tuple:
    """(grup adı, hisseler). tickers verilirse o liste (en az 2 hisse) grup olur; yoksa sektör adı (İngilizce ya da
    Türkçe) çözülür."""
    u = load_universe()
    if tickers:
        members = list(dict.fromkeys(_ticker(t) for t in tickers))
        if len(members) < 2:
            raise ToolError("tickers en az 2 farklı hisse olmalı")
        return (str(sector).strip() if sector else "özel grup: " + ", ".join(members)), members
    if not sector:
        raise ToolError("sector ya da tickers gerekli")
    groups = sorted(u["peer_group"].unique())
    names = {_fold(g): g for g in groups}
    names.update({_fold(a): g for g, al in SECTOR_ALIASES.items() for a in al})
    g = names.get(_fold(sector))
    if g is None:
        raise ToolError(f"bilinmeyen sektör grubu: '{sector}'. Geçerli: {', '.join(groups)} "
                        f"(Türkçe adlar da olur, ör. perakende, banka) ya da tickers=[...] verin")
    return g, u[u["peer_group"] == g]["ticker"].tolist()


@_guarded
def rank_sector_relative_strength(sector: str = None, horizon: int = 10, date: str = None,
                                  tickers: list[str] = None) -> dict:
    """Akran grubu içinde h günlük getiri ve sıra. lag = getiri - kendisi HARİÇ akran ortancası (ana ölçü);
    gap_to_sector_median = getiri - grubun tümünün (kendisi dahil) ortancası. Grup: sector (Türkçe ad da olur)
    ya da tickers=[...] listesi."""
    g, members = _peer_group(sector, tickers)
    if int(horizon) not in (5, 10, 20):
        raise ToolError("horizon 5, 10 ya da 20 olmalı")
    h, d = int(horizon), _asof(date, "date")
    core = load_universe().set_index("ticker")["is_core"]
    ret = pd.Series({m: _stock(m)["Close"].loc[:d].pipe(lambda s: s.iloc[-1] / s.iloc[-1 - h] - 1) for m in members})
    rows = []
    for m in members:
        peers = ret.drop(m)
        lag = ret[m] - peers.median() if len(peers) else None
        rows.append(dict(ticker=m, core=bool(core.loc[m]), ret=ret[m], lag=lag,
                         gap_to_sector_median=ret[m] - ret.median()))
    out = pd.DataFrame(rows).sort_values("ret", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    warn = f"grup küçük ({len(members)} hisse), ortanca zayıf" if len(members) < 3 else ""
    return to_json(dict(sector=g, horizon=h, date=str(d.date()), sector_median=ret.median(),
                        most_behind=out.iloc[-1]["ticker"], members=out.to_dict("records"), warning=warn,
                        note="lag: kendisi hariç akran ortancasına fark (R2 araştırmasıyla aynı tanım). "
                             "gap_to_sector_median: kendisi dahil grup ortancasına fark. Oranlar (0.05 = %5)."))


WEEK_RE = re.compile(r"^([DU.]+)-([DU]+)(\.*)$")
STABLE_P, STABLE_MIN_N = 0.05, 10      # kararlılık kuralı: örneklem içi p < 0,05, dışında aynı yön ve en az 10 gözlem
PATTERN_HELP = ("pattern: 'weekday', 'down_streak:K', 'up_streak:K' (K 1-6) ya da 'week:DDD-UU' (Pzt..Cum 5 gün; "
                "D düşüş, U yükseliş, . fark etmez; tire koşul günlerini sonuç günlerinden ayırır, ör. 'week:D...-U' "
                "Pazartesi düşüş -> Cuma, 'week:D-U...' Pazartesi düşüş -> Salı)")


def _parse_week(pattern: str) -> tuple:
    """'week:DDD-UU' -> (koşul [(gün, D/U)], sonuç [(gün, D/U)]). Gün 0 = Pazartesi."""
    m = WEEK_RE.match(pattern.split(":", 1)[1].replace(" ", "").upper())
    if not m or sum(len(x) for x in m.groups()) != 5 or not set(m.group(1)) - {"."}:
        raise ToolError(PATTERN_HELP)
    k = len(m.group(1))
    return ([(i, c) for i, c in enumerate(m.group(1)) if c != "."],
            [(k + i, c) for i, c in enumerate(m.group(2))])


def _full_weeks(r: pd.Series, valid: pd.Series) -> tuple:
    """Pzt-Cum 5 günü de olan ve hiçbir günü tatil sonrası/deprem olmayan haftalar (satır = haftanın Pazartesi'si,
    sütun 0-4 günlük getiri) ve dışarıda kalan (eksik, tatilli ya da dönem sınırına taşan) hafta sayısı."""
    x = pd.DataFrame({"r": r, "v": valid.reindex(r.index).fillna(False).astype(bool)})
    rows, idx, skipped = [], [], 0
    for _, g in x.groupby(x.index.to_period("W-FRI")):
        if list(g.index.day_of_week) != [0, 1, 2, 3, 4] or not g["v"].all() or g["r"].isna().any():
            skipped += 1
            continue
        rows.append(g["r"].to_numpy())
        idx.append(g.index[0])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=range(5)), skipped


def _week_stats(W: pd.DataFrame, skipped: int, cond: list, out: list) -> dict:
    """Koşul haftalarında sonuç günlerinin toplam getirisi (%) ve diğer haftalarla karşılaştırma."""
    match = lambda i, c: (W[i] < 0) if c == "D" else (W[i] > 0)
    m, hit_all = pd.Series(True, index=W.index), pd.Series(True, index=W.index)
    for i, c in cond:
        m &= match(i, c)
    for i, c in out:
        hit_all &= match(i, c)
    cum = (1 + W[[i for i, _ in out]]).prod(axis=1) - 1
    signs = {c for _, c in out}
    sign = (1 if signs == {"U"} else -1) if len(signs) == 1 else None
    a, b = cum[m], cum[~m]
    res = dict(n_weeks=len(W), excluded_weeks=skipped, n=int(m.sum()),
               full_pattern_count=int((m & hit_all).sum()),
               full_pattern_expected=float(len(W) * np.prod([match(i, c).mean() for i, c in cond + out])) if len(W) else 0.0)
    if len(a):
        res.update(mean=a.mean() * 100, median=a.median() * 100,
                   unconditional=b.mean() * 100 if len(b) else None,
                   diff_pt=(a.mean() - b.mean()) * 100 if len(b) else None,
                   hit_rate=float((sign * a > 0).mean()) if sign else None,
                   all_days_hit_rate=float(hit_all[m].mean()),
                   mean_after_cost=(sign * a.mean() - 2 * COST_ONE_WAY) * 100 if sign else None)
    if len(a) >= 3 and len(b) >= 3:
        res["t"], res["p"] = welch(a, b)
    if len(a) < STABLE_MIN_N:
        res["warning"] = "az gözlem"
    return res


def _stability(i: dict, o: dict) -> dict:
    """Kural: örneklem içinde p < 0,05 VE örneklem dışında aynı yönde fark VE en az 10 gözlem -> 'kararlı'."""
    reasons, p = [], i.get("p")
    if p is None:
        reasons.append("örneklem içinde test yapılamadı (az gözlem)")
    elif p >= STABLE_P:
        reasons.append(f"örneklem içinde anlamlı değil (p={p:.2f})")
    if o.get("n", 0) < STABLE_MIN_N:
        reasons.append(f"örneklem dışında az gözlem ({o.get('n', 0)} < {STABLE_MIN_N})")
    di, do = i.get("diff_pt"), o.get("diff_pt")
    if di is not None and do is not None and np.sign(di) != np.sign(do):
        reasons.append("örneklem dışında fark ters yönde")
    return dict(verdict="kararsız" if reasons else "kararlı", reasons=reasons)


@_guarded
def test_weekday_pattern(ticker: str, pattern: str = "weekday", start: str = None, end: str = None,
                         oos_start: str = None) -> dict:
    """Takvim deseni: haftanın günü, art arda düşüş/yükseliş sonrası ertesi gün ya da hafta içi dizi
    ('week:DDD-UU' = Pzt-Sal-Çar düşüş -> Per+Cum yükseliş). Koşulsuz getiriyle karşılaştırma, Welch t testi,
    maliyet. oos_start verilirse örneklem içi [start, oos_start) ve dışı [oos_start, end] ayrı hesaplanır ve
    kurala göre 'kararlı'/'kararsız' kararı verilir. pattern: 'weekday', 'down_streak:3', 'up_streak:2', 'week:...'."""
    cal = _calendar_data()
    if str(ticker).upper() == "PORTFOLIO":
        name, r = "PORTFOLIO", cal["ret"].mean(axis=1)
    else:
        name = _ticker(ticker, core_only=True)
        r = cal["ret"][name]
    s, e = _range(start, end)
    valid = cal["valid"].loc[s:e]
    r = r.loc[s:e]
    if oos_start:
        o = _date(oos_start, "oos_start")
        if not s < o <= e:
            raise ToolError(f"oos_start ({o.date()}) start ({s.date()}) ile end ({e.date()}) arasında olmalı")
        parts = {"in_sample": r.index < o, "out_of_sample": r.index >= o}
    else:
        parts = {"results": np.ones(len(r), dtype=bool)}

    def compare(mask, part):
        rr, vv, mm = r[part], valid[part], mask[part]
        a, b = rr[vv & mm].dropna(), rr[vv & ~mm].dropna()
        if len(a) < 10:
            return dict(n=len(a), warning="az gözlem")
        t, p = welch(a, b)
        half = rr.index[len(rr) // 2]
        f, sn = rr.index < half, rr.index >= half
        diff = lambda h: (rr[vv & mm & h].mean() - rr[vv & ~mm & h].mean()) * 100
        return dict(n=len(a), mean=a.mean() * 100, median=a.median() * 100, unconditional=b.mean() * 100,
                    diff_pt=(a.mean() - b.mean()) * 100, t=t, p=p, first_half_diff=diff(f), second_half_diff=diff(sn),
                    same_sign_halves=bool(np.sign(diff(f)) == np.sign(diff(sn))))

    if pattern == "weekday":
        dow = pd.Series(r.index.day_of_week, index=r.index)
        run = lambda part: {WEEKDAYS[d]: compare(dow == d, part) for d in range(5)}
        tests = 5
    elif str(pattern).lower().startswith("week:"):
        cond, out = _parse_week(pattern)
        desc = lambda xs: ", ".join(f"{WEEKDAYS[i]} {'düşüş' if c == 'D' else 'yükseliş'}" for i, c in xs)
        label = f"koşul: {desc(cond)} -> sonuç: {desc(out)}"
        run = lambda part: {label: _week_stats(*_full_weeks(r[part], valid), cond, out)}
        tests = 1
    else:
        try:
            kind, k = pattern.split(":")
            k = int(k)
            assert kind in ("down_streak", "up_streak") and 1 <= k <= 6
        except (ValueError, AssertionError):
            raise ToolError(PATTERN_HELP)
        sign = (r < 0) if kind == "down_streak" else (r > 0)
        streak = sign.astype(int).groupby((~sign).cumsum()).cumsum()
        run = lambda part: {f"{kind}:{k} sonrası ertesi gün": compare(streak.shift(1) >= k, part)}
        tests = 1
    body = {key: run(part) for key, part in parts.items()}
    if oos_start:
        body["stability"] = {k: _stability(body["in_sample"][k], body["out_of_sample"][k]) for k in body["in_sample"]}
        body["stability_rule"] = (f"kararlı = örneklem içinde p < {STABLE_P} VE örneklem dışında fark aynı yönde VE "
                                  f"en az {STABLE_MIN_N} gözlem; biri tutmazsa kararsız")
    return to_json(dict(ticker=name, pattern=pattern, start=str(s.date()), end=str(e.date()),
                        oos_start=str(o.date()) if oos_start else None, **body, tests_in_this_call=tests,
                        note="Getiriler yüzde (1.36 = %1,36). week desenlerinde sonuç = sonuç günlerinin toplam "
                             "getirisi; unconditional = koşulun olmadığı haftalar; full_pattern_expected = günler "
                             "bağımsız olsaydı tam dizinin beklenen sayısı; mean_after_cost = desen yönünde işlem "
                             "(son koşul günü kapanışında gir, son sonuç günü kapanışında çık; D için açığa satış "
                             "varsayımı), gidiş-dönüş maliyet düşülmüş.",
                        warnings=["Çoklu test riski: birçok desen denenirse şansla anlamlı görünen çıkar; R3'te 243 "
                                  "testin yalnızca 18'i düzeltme (Benjamini-Hochberg) sonrası kaldı.",
                                  f"Gidiş-dönüş maliyet ~%{2 * COST_ONE_WAY * 100:.2f}; günlük farklar çoğu zaman "
                                  "bundan küçük."]))


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


@_guarded
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


@_guarded
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
@_guarded
def get_macro_context(start: str = None, end: str = None) -> dict:
    """Kur, faiz, enflasyon ve Fed aralığı; her gün yalnızca o gün yayınlanmış değerler."""
    s, e = _range(start, end)
    df = _macro(s, e)
    meta = dict(source="TCMB EVDS (USD/TRY, EUR/TRY, politika faizi, TÜFE), Federal Reserve (hedef aralık)",
                publication=f"TÜFE izleyen ayın {CPI_RELEASE_DAY}'ünde görünür; kur ve faiz kendi gününde; "
                            "Fed kararı ertesi gün görünür")
    return dict(meta=meta, **_rows(df))


@_guarded
def get_news_events(ticker: str = None, topic: str = None, start: str = None, end: str = None,
                    asof: str = None) -> dict:
    """KAP ve RSS haber olayları (yalnızca başlık/özet, kaynak bağlantısı ve zaman)."""
    t = _ticker(ticker, core_only=True) if ticker else None
    if topic and topic not in CATEGORY_RULES and topic != "diger":
        raise ToolError(f"bilinmeyen konu: '{topic}'. Geçerli: {', '.join(list(CATEGORY_RULES) + ['diger'])}")
    s = _date(start, "start", "2021-11-01")
    if _active_ceiling() is None:
        e = _date(end, "end", pd.Timestamp.today().normalize())
        a = _asof(asof) if asof else None
    else:                                   # tavan varsa: en geç tavan günü, yalnızca o gün kapanışında bilinenler
        top = ceiling()
        e = _date(end, "end", top)
        if e > top:
            _note(f"istenen 'end' analiz tarihinden sonraydı; {top.date()} tarihine kırpıldı")
            e = top
        a = _asof(asof) if asof else top
    n = _news_query(load_news_events(), ticker=t, topic=topic, start=s, end=e, asof=a)
    total = len(n)
    n = n.tail(MAX_NEWS)
    cols = ["source", "publish_time", "event_day", "ticker", "scope", "category", "categories", "title", "summary", "url"]
    return to_json(dict(n_total=total, n_returned=len(n), events=n[cols].to_dict("records"),
                        note="Haber bağlamdır; sayısal veri yerine geçmez. Kaynak ve zaman her kayıtta.",
                        warning="haber kapsamı seyrek" if total < 3 else ""))


@_guarded
def get_speech_features(source_id: str = None) -> dict:
    """Konuşma kaynağının konu, duruş, belirsizlik ve doğrulanmış/doğrulanmamış iddiaları. source_id yoksa liste."""
    sp = load_speech_features()
    visible = list(SOURCES)
    if _active_ceiling() is not None:       # analiz tarihinde henüz bilinmeyen kaynak hiç yokmuş gibi davranılır
        known = set(sp.loc[speech_known_day(sp) <= ceiling(), "source_id"])
        visible = [k for k in SOURCES if k in known]
    if not source_id:
        return dict(sources=[dict(source_id=k, speaker=SOURCES[k]["speaker"], publish_time=SOURCES[k]["publish_time"],
                                  kind=SOURCES[k]["kind"]) for k in visible])
    if source_id not in visible:
        raise ToolError(f"bilinmeyen konuşma kaynağı: '{source_id}'. Geçerli: {', '.join(visible) or '(yok)'}")
    return to_json(_speech(sp, source_id))


# ---------- 13-15: backtest, kalite, kanıt ----------
@_guarded
def run_backtest(strategy: str, start: str = None, end: str = None, cost: float = COST_ONE_WAY,
                 sector: str = None, tickers: list[str] = None, ticker: str = None, lookback: int = 20,
                 threshold: float = None) -> dict:
    """Strateji varyantının dönem performansı (sinyal 1 gün kaydırılır, maliyet dahil) ve işlem özeti.
    strategy='catch_up': akran grubunda (sector ya da tickers) geçmiş geride kalmaların 5/10/20 günde kapanıp
    kapanmadığı; ticker = incelenen hisse (verilmezse en geride olan), threshold verilmezse hissenin bugünkü farkı."""
    variants = _variants()
    if strategy != "catch_up" and strategy not in variants:
        raise ToolError(f"bilinmeyen strateji: '{strategy}'. Geçerli: {', '.join(list(variants) + ['catch_up'])}")
    if not 0 <= float(cost) <= 0.01:
        raise ToolError("cost (tek yön) 0 ile 0,01 arasında olmalı")
    if strategy == "catch_up":
        return _catch_up(start, end, float(cost), sector, tickers, ticker, int(lookback), threshold)
    if sector or tickers or ticker or threshold is not None or int(lookback) != 20:
        raise ToolError("sector, tickers, ticker, lookback ve threshold yalnızca strategy='catch_up' için")
    s, e = _range(start, end)
    ret = load_returns().iloc[WARMUP:].loc[s:e]
    if len(ret) < 20:
        raise ToolError(f"backtest için en az 20 işlem günü gerekli (ısınma sonrası ilk gün {load_returns().index[WARMUP].date()})")
    w = variants[strategy].reindex(ret.index)
    bt = _engine_backtest(w, ret, cost=float(cost))
    bm = benchmarks(ret)
    trades = trade_log(w, ret, cost=float(cost))
    perf = calculate_performance(bt, bm["esit_agirlik"], sector_neutral(w, ret), trades)
    perf["xu100_getiri"] = float((1 + bm["xu100"].fillna(0)).prod() - 1)
    perf["xu100_farki"] = perf["toplam_getiri"] - perf["xu100_getiri"]          # model hesaplamasın, araç hesaplar
    perf["esit_agirlik_getiri"], perf["esit_agirlik_farki"] = perf.pop("kiyas_getiri"), perf.pop("fark")
    last = w.iloc[-1]
    recent = trades.tail(10).assign(entry=lambda d: d["entry"].dt.strftime("%Y-%m-%d"),
                                    exit=lambda d: d["exit"].dt.strftime("%Y-%m-%d"))
    return to_json(dict(strategy=strategy, start=str(ret.index[0].date()), end=str(ret.index[-1].date()),
                        cost_one_way=float(cost), performance=perf,
                        last_positions=last[last > 0].round(4).to_dict(), recent_trades=recent.to_dict("records"),
                        note="Sinyal t kapanışında, işlem t+1 getirisine uygulanır; universe 30 çekirdek hisse. "
                             "Getiriler oran (2.41 = %241). esit_agirlik = 30 hissenin eşit ağırlıklı al-tut kıyası. "
                             "sektor_kiyas = stratejinin sektör ağırlıklarıyla, sektör içinde eşit ağırlıklı kıyas "
                             "(fark = sektör içi hisse seçimi). islem_* = kapanmış işlemler (giriş-çıkış, maliyet dahil)."))


MAX_FP_ROWS = 20


def _catch_up(start, end, cost, sector, tickers, ticker, lookback, threshold) -> dict:
    """run_backtest(strategy='catch_up') gövdesi. Olay tanımı ve sonuç: src/events/sector_catchup.catchup_history."""
    if lookback not in (5, 10, 20):
        raise ToolError("lookback 5, 10 ya da 20 olmalı")
    g, members = _peer_group(sector, tickers)
    s, e = _range(start, end)
    panel = pd.DataFrame({m: _stock(m)["Close"] for m in members}).loc[:e]
    lag_now = sector_lag(panel, {"grup": members}, lookback).iloc[-1]
    focus = _ticker(ticker) if ticker else lag_now.idxmin()
    if focus not in members:
        raise ToolError(f"ticker '{focus}' grupta yok: {', '.join(members)}")
    if threshold is None:
        if not lag_now[focus] < 0:
            raise ToolError(f"{focus} {e.date()} itibarıyla akranlarının gerisinde değil "
                            f"(fark {lag_now[focus]:+.4f}); threshold verin")
        thr, thr_src = float(-lag_now[focus]), "hissenin bugünkü farkı"
    else:
        thr, thr_src = float(threshold), "istekte verilen"
        if not 0.005 <= thr <= 0.5:
            raise ToolError("threshold 0,005 ile 0,5 arasında olmalı (oran; 0.10 = 10 puan geride)")
    ev = catchup_history(panel, lookback, thr, s, e)
    quake = int(ev["quake"].sum())
    ev = ev[~ev["quake"]]
    rt = 2 * cost

    def scope(x):
        out = dict(n_events=int(len(x)), by_horizon=catchup_summary(x, cost_round_trip=rt))
        done = x["rel_20"].dropna()
        out["false_positive_rate_20"] = float((done <= 0).mean()) if len(done) else None
        fp = x[x["rel_20"] <= 0]
        out["false_positives"] = to_json(fp.assign(date=pd.to_datetime(fp["date"]).dt.strftime("%Y-%m-%d"))
                                         [["date", "ticker", "lag", "rel_20"]].tail(MAX_FP_ROWS).to_dict("records"))
        if len(fp) > MAX_FP_ROWS:
            out["false_positives_note"] = f"{len(fp)} yanlış pozitiften son {MAX_FP_ROWS} tanesi"
        return out

    stock, group = scope(ev[ev["ticker"] == focus]), scope(ev)
    warnings = [f"{name}: yalnızca {x['n_events']} olay (< {MIN_EVENTS}); sonuç zayıf"
                for name, x in (("hisse", stock), ("grup", group)) if x["n_events"] < MIN_EVENTS]
    if quake:
        warnings.append(f"Şubat 2023 deprem penceresindeki {quake} olay istatistik dışı (R2 ile aynı)")
    return to_json(dict(
        strategy="catch_up", sector=g, members=members, start=str(s.date()), end=str(e.date()), ticker=focus,
        lookback=lookback, current_lag=float(lag_now[focus]), threshold=thr, threshold_source=thr_src,
        cost_one_way=cost, stock=stock, group=group, warnings=warnings,
        note=(f"Olay: {lookback} günlük getiri - kendisi hariç akran ortancası ilk kez -threshold ya da altına "
              "indiği gün (aynı hissede en az 10 gün arayla). rel_h: olay günü kapanışından h gün sonrasına "
              "hisse getirisi - kendisi hariç akran ortancası. hit_rate: rel_h > 0 (akranlarını geçti); "
              "full_close_rate: rel_h olay günündeki farkı tamamen kapattı; yanlış pozitif: 20 gün sonra hâlâ "
              "geride (rel_20 <= 0). n_incomplete: penceresi analiz tarihini aştığı için henüz bilinmeyen ve "
              "istatistiğe katılmayan olaylar. mean_rel_after_cost: gidiş-dönüş maliyet düşülmüş. Oranlar "
              "(0.05 = 5 puan). Geçmiş sonuç gelecek garantisi değildir.")))


@_guarded
def check_data_quality(ticker: str, asof: str = None) -> dict:
    """Analiz paketi için kalite kapısı: kesin engeller, güvensiz durumlar, yumuşak uyarılar."""
    t = str(ticker or "").strip().upper()
    d = _asof(asof)
    r = run_analysis(t, d)
    return to_json(dict(ticker=t, asof=str(d.date()), **r["gate"]))


@_guarded
def build_evidence_bundle(ticker: str = None, asof: str = None, analysis_id: str = None) -> dict:
    """Tam gözlem seti + committed evidence + kurallı eylem + kapı. analysis_id verilirse günlükten okunur."""
    if analysis_id:
        try:
            a = load_analysis(analysis_id)
        except ValueError as e:
            raise ToolError(str(e))
        if _active_ceiling() is not None and pd.Timestamp(a["asof"]) > ceiling():
            raise ToolError("bu analiz kaydı analiz tarihinden sonraki bir güne ait; o tarihte yoktu")
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
