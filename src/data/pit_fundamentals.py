"""Zamanında bilinen (point-in-time) temel veri katmanı. Adım 1a, 2026-11-22.

Sorun. Fintables, enflasyon muhasebesi (TMS 29) uygulayan şirketlerin rakamlarını ilk yayınlandıkları
gibi saklamıyor. 2026-11-22'de yayınlanmış rakamlarla doğrulanan kurallar (testler:
tests/test_pit_fundamentals.py):

  * TMS 29 uygulamayan 6 şirket (NOT_RESTATED) ve diğer 24 şirketin 2022/9 ve önceki dönemleri:
    açıklandığı gibi, nominal TL.
  * 24 şirketin 2022/12 rakamları (gelir + bilanço) ile 2023/3-2023/9 gelir/nakit akış rakamları:
    sonradan düzeltilmiş sürümler. Bunlar ilk kez BİR YIL SONRAKİ raporda karşılaştırma olarak
    yayınlandı. Aynı dönemlerin (2023/3-9) bilançoları ise açıklandığı gibi (hiç düzeltilmedi).
  * 24 şirketin 2023/12 ve sonrası: enflasyon muhasebeli. Bu düzeltilmiş rakamlar ve 2023/12 sonrası
    rakamlar, şirketin son bilanço ayının (U) parasına çevrilmiş:
        fintables = yayınlanan × TÜFE(U) / TÜFE(raporun dönem sonu ayı)

Buna göre her rakamın bir dönem türü (era) vardır:
    A = açıklandığı gibi (birim ailesi 'nominal')
    B = sonradan düzeltilmiş, bir yıl sonra yayınlandı (birim ailesi 'U')
    C = enflasyon muhasebeli, kendi raporunda yayınlandı (birim ailesi 'U')

İki kural (kullanıcı kararı, 2026-11-22):
  1. Bir hesap, girdilerinin elimizdeki sürümü yayınlandıktan sonra kullanılabilir.
     known_from = girdilerin en geç yayın günü (A ve C: kendi raporunun event_day'i; B: bir yıl sonraki
     raporun event_day'i). B rakamları böylece yalnızca ertesi yılın karşılaştırması olarak kullanılır.
  2. Bir hesaptaki bütün girdiler aynı birim ailesinde olmalı. 'nominal' ile 'U' karışamaz.
TL tutarları, yayınlandığı raporun parasıyla gösterilir (TÜFE oranıyla geri çevrilir). Çevirme sabiti
TÜFE(U) analiz gününden sonraya ait olabilir. Bu bilgi sızdırmaz: yalnızca Fintables'ın çevirmesini geri
alır ve sonuç, o gün yayınlanmış rakamın aynısıdır.

Bu modül mevcut tüketicileri (Rapor 4, backtest, harness) henüz değiştirmez. Onlar Adım 1b'de bu katmana
geçecek.

Çalıştırma: python -m src.data.pit_fundamentals    (doğrulama özeti; ağ isteği yok)
"""
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.disclosure_dates import START_PERIOD, legal_deadline
from src.data.fundamentals import FINANCIALS, LABELS, _period_end, _row
from src.data.kap_dates import load_kap_table
from src.data.macro import load_series
from src.data.universe import load_universe

FT_DIR = Path("data/fintables")
CUM_SHEETS = {"flow": "Gelir tablosu (dönemsel)", "stock": "Bilanço"}   # gelir tablosu: dönem başından birikimli

# Fintables verisinde 2022/9 -> 2022/12 sıçraması olmayan, yani rakamları yeniden düzenlenmemiş şirketler.
# Banka (AKBNK) ve sigorta (TURSG) bu dönemde TMS 29 uygulamadı. ENKAI, EREGL, THYAO ve PGSUS'ta da Fintables
# rakamları nominal kalıyor. Liste, testte verideki özkaynak sıçramalarıyla karşılaştırılır.
NOT_RESTATED = frozenset({"AKBNK", "TURSG", "ENKAI", "EREGL", "THYAO", "PGSUS"})
LAST_NOMINAL = pd.Timestamp("2022-09-30")      # bu ve önceki dönemler açıklandığı gibi
FIRST_RESTATED = pd.Timestamp("2022-12-31")    # 2022 yıl sonu: gelir ve bilanço sonradan düzeltilmiş (B)
FIRST_TMS29 = pd.Timestamp("2023-12-31")       # ilk enflasyon muhasebeli rapor (C)
# Mali yılı takvimden farklı şirket: Fintables etiketi "2026/6", gerçek dönem sonu bir ay sonra (31 Temmuz).
FISCAL_OFFSET_MONTHS = {"MAVI": 1}

FLOW_ITEMS = LABELS["income"]      # net_income, revenue, gross_profit, op_income
STOCK_ITEMS = LABELS["balance"]    # equity, total_assets
NO_SALES_LINES = ("revenue", "gross_profit", "op_income")   # banka/sigorta/aracı kurumda anlamsız


def core_tickers() -> list:
    u = load_universe()
    return u[u["is_core"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].tolist()


def restaters() -> frozenset:
    return frozenset(core_tickers()) - NOT_RESTATED


# ---------------------------------------------------------------- dönem türü, birim, yayın günü
def era(ticker: str, period_end, kind: str) -> str:
    """kind: 'flow' (gelir/nakit akış, birikimli) ya da 'stock' (bilanço)."""
    p = pd.Timestamp(period_end)
    if ticker in NOT_RESTATED or p <= LAST_NOMINAL:
        return "A"
    if p >= FIRST_TMS29:
        return "C"
    if p == FIRST_RESTATED:
        return "B"
    return "B" if kind == "flow" else "A"       # 2023/3-9: gelir düzeltilmiş, bilanço açıklandığı gibi


def family(e: str) -> str:
    return "nominal" if e == "A" else "U"


def actual_end(ticker: str, period_end) -> pd.Timestamp:
    """Fintables etiketinin gerçek dönem sonu (MAVI'de bir ay sonra)."""
    k = FISCAL_OFFSET_MONTHS.get(ticker, 0)
    return pd.Timestamp(period_end) + pd.offsets.MonthEnd(k) if k else pd.Timestamp(period_end)


@lru_cache(maxsize=1)
def _cpi() -> pd.Series:
    c = load_series("cpi")
    return c.set_index("data_date")["value"]


def cpi_at(month_end) -> float:
    d = pd.Timestamp(month_end)
    return float(_cpi().loc[pd.Timestamp(d.year, d.month, 1)])


@lru_cache(maxsize=None)
def unit_month(ticker: str):
    """Fintables'ın bu şirketin B/C rakamlarını çevirdiği ay: son bilanço döneminin gerçek sonu (U).
    Yeniden düzenlenmeyen şirketlerde None."""
    if ticker in NOT_RESTATED:
        return None
    st = read_raw(ticker)
    last = st.index[st["equity"].notna()].max()
    return actual_end(ticker, last)


def report_factor(ticker: str, report_period) -> float:
    """'U' birimindeki bir Fintables rakamını, report_period raporunun parasına çeviren çarpan."""
    u = unit_month(ticker)
    return 1.0 if u is None else cpi_at(actual_end(ticker, report_period)) / cpi_at(u)


@lru_cache(maxsize=1)
def _event_days() -> dict:
    k = load_kap_table()
    k = k[k["event_day"].notna()]
    return {(t, pd.Timestamp(p)): pd.Timestamp(d) for t, p, d in zip(k["ticker"], k["period_end"], k["event_day"])}


def event_day(ticker: str, period_end):
    """Raporun piyasa tarafından görülebildiği ilk işlem günü (KAP). Bilinmiyorsa NaT."""
    return _event_days().get((ticker, pd.Timestamp(period_end)), pd.NaT)


def known_from(ticker: str, period_end, e: str):
    """Elimizdeki sürümün yayınlandığı gün. B: bir yıl sonraki raporun event_day'i."""
    p = pd.Timestamp(period_end)
    if e == "B":
        return event_day(ticker, p + pd.offsets.QuarterEnd(4))
    d = event_day(ticker, p)
    if pd.isna(d) and p < START_PERIOD:
        # KAP tablosundan önceki eski raporlar (yalnızca karşılaştırma olarak kullanılır): yasal son tarih üst sınır
        d = legal_deadline(p) + pd.DateOffset(months=FISCAL_OFFSET_MONTHS.get(ticker, 0))
    return d


# ---------------------------------------------------------------- ham veri
@lru_cache(maxsize=None)
def read_raw(ticker: str) -> pd.DataFrame:
    """Fintables'tan birikimli gelir kalemleri (flow) ve bilanço kalemleri (stock), dönem etiketine göre."""
    path = FT_DIR / f"{ticker} (TRY).xlsx"
    out = {}
    for kind, items in (("flow", FLOW_ITEMS), ("stock", STOCK_ITEMS)):
        df = pd.read_excel(path, sheet_name=CUM_SHEETS[kind])
        for name, cands in items.items():
            s = _row(df, cands)
            if s is not None:
                out[name] = s
    res = pd.DataFrame(out)
    res.index = pd.DatetimeIndex([_period_end(c) for c in res.index], name="period_end")
    res = res.sort_index()
    if ticker in FINANCIALS:
        res = res.drop(columns=[c for c in NO_SALES_LINES if c in res.columns])
    return res


# ---------------------------------------------------------------- nicelik tablosu: değer + birim ailesi + yayın günü
def _put(df, name, value, fam, known, why=""):
    df[name] = value
    df[f"{name}__fam"] = fam
    df[f"{name}__known"] = pd.to_datetime(known)
    df[f"{name}__why"] = why


def _shifted(df: pd.DataFrame, name: str, n: int) -> pd.DataFrame:
    """'name' niceliğinin n çeyrek önceki değeri, P satırına hizalı (konuma göre değil tarihe göre)."""
    cols = [name, f"{name}__fam", f"{name}__known", f"{name}__why"]
    src = df[cols].reindex(df.index - pd.offsets.QuarterEnd(n))
    src.index = df.index
    return src.rename(columns={c: c.replace(name, f"{name}_l{n}", 1) for c in cols})


def _combine(df: pd.DataFrame, name: str, fn, inputs: list, rows=None):
    """Girdilerden yeni nicelik. Girdilerden biri eksikse 'eksik', birim aileleri farklıysa 'birim'.
    known = girdilerin en geç yayın günü. rows verilirse yalnızca o satırlar yazılır."""
    rows = df.index if rows is None else rows
    if name not in df.columns:
        _put(df, name, np.nan, None, pd.NaT)
    if len(rows) == 0:
        return
    sub = df.loc[rows]
    vals = pd.concat([sub[i] for i in inputs], axis=1)
    fams = pd.concat([sub[f"{i}__fam"] for i in inputs], axis=1)
    knowns = pd.concat([sub[f"{i}__known"] for i in inputs], axis=1)
    present = vals.notna().all(axis=1)
    fam0 = fams.iloc[:, 0]
    same = fams.eq(fam0, axis=0).all(axis=1) & fam0.notna()
    ok = present & same
    value = fn(*[sub[i] for i in inputs]).replace([np.inf, -np.inf], np.nan).where(ok)
    known = knowns.max(axis=1, skipna=False).where(ok)
    why = pd.Series(np.select([~present, ~same], ["eksik", "birim"], default=""), index=rows)
    df.loc[rows, name] = value
    df.loc[rows, f"{name}__fam"] = fam0.where(ok)
    df.loc[rows, f"{name}__known"] = known
    df.loc[rows, f"{name}__why"] = why


@lru_cache(maxsize=None)
def panel(ticker: str) -> pd.DataFrame:
    """Her dönem etiketi için bütün nicelikler ve ayrıntıları (__fam, __known, __why). Testler ve denetim için."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # sütun sütun kurulum; sonuç kopyalanır
        return _panel(ticker).copy()


def _panel(ticker: str) -> pd.DataFrame:
    raw = read_raw(ticker)
    df = pd.DataFrame(index=raw.index)
    eras_f = [era(ticker, p, "flow") for p in raw.index]
    eras_s = [era(ticker, p, "stock") for p in raw.index]
    df["era_flow"], df["era_stock"] = eras_f, eras_s
    kf = [known_from(ticker, p, e) for p, e in zip(raw.index, eras_f)]
    ks = [known_from(ticker, p, e) for p, e in zip(raw.index, eras_s)]
    flows = [c for c in FLOW_ITEMS if c in raw.columns]
    stocks = [c for c in STOCK_ITEMS if c in raw.columns]
    for c in flows:
        _put(df, f"c_{c}", raw[c], [family(e) for e in eras_f], kf)
    for c in stocks:
        _put(df, f"s_{c}", raw[c], [family(e) for e in eras_s], ks)

    q1 = df.index.month == 3                    # mali yılın ilk çeyreği (MAVI dahil etiketler hizalı)
    fy_start = df.index[q1]
    for c in flows:
        df = df.join(_shifted(df, f"c_{c}", 1))
        # tek çeyrek: ilk çeyrekte birikimli değerin kendisi, diğerlerinde C(P) - C(P-1)
        _put(df, f"q_{c}", np.nan, None, pd.NaT)
        _combine(df, f"q_{c}", lambda a: a, [f"c_{c}"], rows=fy_start)
        _combine(df, f"q_{c}", lambda a, b: a - b, [f"c_{c}", f"c_{c}_l1"], rows=df.index[~q1])
        df = df.join(_shifted(df, f"q_{c}", 1)).join(_shifted(df, f"q_{c}", 4))
        # son 12 ay: yıl sonunda birikimli; diğerlerinde C(P) + geçen yılın yıllığı - C(P-4).
        # Birikimli değerlerle hesaplanır: düzeltilmiş ara dönemler toplamda birbirini götürür (2024/3: 2023 yıllık
        # + 2024/3 - düzeltilmiş 2023/3, hepsi 2024/3 raporuyla bilinir).
        df = df.join(_shifted(df, f"c_{c}", 4))
        fy = pd.Series([p - pd.offsets.QuarterEnd(p.month // 3) for p in df.index], index=df.index)
        prev_fy = df[[f"c_{c}", f"c_{c}__fam", f"c_{c}__known", f"c_{c}__why"]].reindex(fy.to_numpy())
        prev_fy.index = df.index
        df = df.join(prev_fy.rename(columns=lambda x: x.replace(f"c_{c}", f"c_{c}_fy1", 1)))
        _put(df, f"ttm_{c}", np.nan, None, pd.NaT)
        ye = df.index.month == 12
        _combine(df, f"ttm_{c}", lambda a: a, [f"c_{c}"], rows=df.index[ye])
        _combine(df, f"ttm_{c}", lambda a, b, d: a + b - d, [f"c_{c}", f"c_{c}_fy1", f"c_{c}_l4"],
                 rows=df.index[~ye])
    for c in stocks:
        df = df.join(_shifted(df, f"s_{c}", 4))

    # oranlar (aynı birim ailesinde; ölçek birbirini götürür)
    growth = lambda a, b: (a - b) / b.abs()
    _combine(df, "ni_yoy", growth, ["q_net_income", "q_net_income_l4"])
    _combine(df, "ni_qoq", growth, ["q_net_income", "q_net_income_l1"])
    _combine(df, "roe_q", lambda a, b: a / b, ["q_net_income", "s_equity"])
    _combine(df, "leverage", lambda a, b: a / b, ["s_total_assets", "s_equity"])
    if "q_revenue" in df:
        _combine(df, "net_margin", lambda a, b: a / b, ["q_net_income", "q_revenue"])
        _combine(df, "revenue_yoy", growth, ["q_revenue", "q_revenue_l4"])
        df = df.join(_shifted(df, "net_margin", 4))
        _combine(df, "margin_chg", lambda a, b: a - b, ["net_margin", "net_margin_l4"])
    if "q_op_income" in df:
        _combine(df, "op_income_yoy", growth, ["q_op_income", "q_op_income_l4"])
    df = df.join(_shifted(df, "roe_q", 4))
    _combine(df, "roe_chg", lambda a, b: a - b, ["roe_q", "roe_q_l4"])
    return df


# ---------------------------------------------------------------- olay günü görünümü
RATIOS = ["ni_yoy", "ni_qoq", "roe_q", "leverage", "net_margin", "revenue_yoy", "op_income_yoy", "margin_chg",
          "roe_chg"]
LEVELS = {  # çıktı adı -> panel niceliği (TL; yayınlandığı raporun parasıyla)
    "net_income": "q_net_income", "net_income_prev": "q_net_income_l4", "revenue": "q_revenue",
    "gross_profit": "q_gross_profit", "op_income": "q_op_income", "net_income_ttm": "ttm_net_income",
    "equity": "s_equity", "total_assets": "s_total_assets",
}


def pit_quarters(ticker: str) -> pd.DataFrame:
    """Her açıklanmış çeyrek için, açıklama gününde (event_day) bilinebilen değerler.
    Bilinemeyen ya da birim karıştıran değerler boş bırakılır; nedeni 'pit_dropped' sütununda yazılır.
    TL tutarları o raporun parasıyla; 'unit' sütunu birimi söyler."""
    df = panel(ticker)
    rows = []
    for p in df.index:
        ev = event_day(ticker, p)
        if pd.isna(ev):
            continue
        r = df.loc[p]
        u = unit_month(ticker)
        out = dict(ticker=ticker, period_end=p, event_day=ev, era_flow=r["era_flow"], era_stock=r["era_stock"])
        dropped = []
        for name, q in [(n, n) for n in RATIOS] + list(LEVELS.items()):
            if q not in df.columns:
                continue
            v, fam, kn, why = r[q], r[f"{q}__fam"], r[f"{q}__known"], r[f"{q}__why"]
            if pd.isna(v):
                if why:
                    dropped.append(f"{name}:{why}")
                out[name] = np.nan
                continue
            if pd.isna(kn) or kn > ev:
                dropped.append(f"{name}:gec({'?' if pd.isna(kn) else pd.Timestamp(kn).date()})")
                out[name] = np.nan
                continue
            if name in LEVELS and fam == "U":
                v = v * report_factor(ticker, p)
            out[name] = float(v)
        out["loss"] = bool(out["net_income"] < 0) if pd.notna(out.get("net_income")) else None
        # birim: gelir ve bilanço ayrı (2023/3-9'da TMS 29 şirketlerinde bilanço açıklandığı gibi, gelir düzeltilmiş)
        here = f"TL@{actual_end(ticker, p):%Y-%m}"
        out["unit_flow"] = {"A": "TL nominal", "C": here}.get(r["era_flow"], "-") if u is not None else "TL nominal"
        out["unit_stock"] = {"A": "TL nominal", "C": here}.get(r["era_stock"], "-") if u is not None else "TL nominal"
        out["pit_dropped"] = ";".join(dropped)
        rows.append(out)
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def pit_all() -> pd.DataFrame:
    return pd.concat([pit_quarters(t) for t in core_tickers()], ignore_index=True)


# ---------------------------------------------------------------- açıklandığı haliyle tek rakam (denetim, test)
def as_published(ticker: str, period_end, item: str, quarterly: bool = False) -> tuple:
    """Birikimli (ya da tek çeyrek) kalemi, İLK yayınlandığı raporun parasıyla döndürür.
    Sonuç: (değer, birim, yayınlandığı rapor dönemi). B rakamları bir yıl sonraki raporda yayınlandı."""
    p = pd.Timestamp(period_end)
    kind = "stock" if item in STOCK_ITEMS else "flow"
    col = f"{'s' if kind == 'stock' else ('q' if quarterly else 'c')}_{item}"
    pan = panel(ticker)
    v = pan.loc[p, col]
    e = era(ticker, p, kind)
    rep = p + pd.offsets.QuarterEnd(4) if e == "B" else p
    kn, ev = pan.loc[p, f"{col}__known"], event_day(ticker, rep)
    if pd.notna(ev) and (pd.isna(kn) or kn > ev):
        v = np.nan               # ör. 2023/4. çeyrek tek başına: düzeltilmiş 9 ay Kasım 2024'te yayınlandı
    if e == "A":
        return float(v), "TL nominal", p
    return float(v) * report_factor(ticker, rep), f"TL@{actual_end(ticker, rep):%Y-%m}", rep


def in_report_unit(ticker: str, period_end, item: str, report_period, quarterly: bool = False) -> float:
    """Bir dönemin rakamını, başka bir raporun (ör. ertesi yılın karşılaştırması) parasıyla döndürür."""
    p = pd.Timestamp(period_end)
    kind = "stock" if item in STOCK_ITEMS else "flow"
    col = f"{'s' if kind == 'stock' else ('q' if quarterly else 'c')}_{item}"
    v = panel(ticker).loc[p, col]
    return float(v) * (report_factor(ticker, report_period) if family(era(ticker, p, kind)) == "U" else 1.0)


# ---------------------------------------------------------------- veriyle denetim
def era_evidence() -> pd.DataFrame:
    """Dönem türü kurallarının veride görünen izi: özkaynak sıçramaları ve Fintables'ta 2022/4. çeyreğin yokluğu."""
    rows = []
    for t in core_tickers():
        eq = read_raw(t)["equity"]
        quarterly = pd.read_excel(FT_DIR / f"{t} (TRY).xlsx", sheet_name="Gelir tablosu (çeyreklik)")
        ni_q = _row(quarterly, FLOW_ITEMS["net_income"])
        rows.append(dict(
            ticker=t, restated=t not in NOT_RESTATED,
            jump_2022q4=eq.get(pd.Timestamp("2022-12-31")) / eq.get(pd.Timestamp("2022-09-30")),
            jump_2023q4=eq.get(pd.Timestamp("2023-12-31")) / eq.get(pd.Timestamp("2023-09-30")),
            fintables_has_2022q4=bool(ni_q is not None and pd.notna(ni_q.get("2022/12"))),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print("Dönem türü kanıtı (özkaynak oranları):")
    print(era_evidence().round(2).to_string(index=False))
    a = pit_all()
    print(f"\nolay satırı: {len(a)}")
    for col in ("ni_yoy", "ni_qoq", "roe_q", "net_income_ttm"):
        print(f"  {col}: geçerli {int(a[col].notna().sum())}")
    print("\nTMS 29 şirketlerinde 2023 çeyrekleri (ASELS):")
    print(a[(a["ticker"] == "ASELS") & (a["period_end"].between("2022-09-30", "2024-06-30"))]
          [["period_end", "event_day", "era_flow", "era_stock", "unit_flow", "net_income", "ni_yoy", "ni_qoq",
            "pit_dropped"]].to_string(index=False))
