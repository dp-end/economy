"""Günlük piyasa değeri (Adım 1d, 2026-11-22). Kaynak: İş Yatırım HisseTekil uç noktası (src/data/prices.py ile aynı,
herkese açık). Fiyat önbelleği (data/isyatirim) düzeltilmiş kapanışı tutar; değerleme için gereken alanlar ayrı bir
önbelleğe yazılır, fiyat önbelleğine dokunulmaz:

  PD          piyasa değeri = düzeltilmemiş kapanış × ödenmiş sermaye (pay sayısı; BIST'te pay başına 1 TL nominal)
  HG_KAPANIS  düzeltilmemiş kapanış (bedelsiz artırım günü yarıya iner)
  SERMAYE     o günkü ödenmiş sermaye
  HAO_PD      halka açık kısmın piyasa değeri

Doğrulama (testler: tests/test_market_cap.py): PD = HG_KAPANIS × SERMAYE; bedelsiz artırım günlerinde PD sıçramaz;
günlük PD getirisi, temettü günleri dışında düzeltilmiş fiyatın getirisine eşittir.

Çalıştırma: python -m src.data.market_cap      # eksik hisseleri indirir (hisse başına 5 istek), sonra denetler
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.prices import RANGES, URL, _fetch_raw
from src.data.universe import load_universe

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "isyatirim_mcap"
COLS = ["PD", "HG_KAPANIS", "SERMAYE", "HAO_PD"]


def core_tickers() -> list:
    u = load_universe()
    return u[u["is_core"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].tolist()


def fetch(ticker: str, force: bool = False) -> pd.DataFrame:
    """Hissenin günlük piyasa değeri alanları (önbellekten; yoksa İş Yatırım'dan, 0,4 sn aralıklı 5 istek)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    rows = []
    for s, e in RANGES:
        r = requests.get(URL, params={"hisse": ticker, "startdate": s, "enddate": e},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        rows += r.json()["value"]
        time.sleep(0.4)
    if not rows:
        raise ValueError(f"Veri yok: {ticker}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["HGDG_TARIH"], format="%d-%m-%Y")
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated()][COLS + ["HGDG_KAPANIS"]].astype(float)
    df.to_parquet(path)
    return df


TOL = 0.01      # fiyat bölünme oranı ile sermaye oranı arasındaki göreli tolerans


def _split_factor(d: pd.DataFrame) -> pd.Series:
    """Ham fiyat getirisi / düzeltilmiş fiyat getirisi: bedelsiz/bölünme gününde 1/k, temettü gününde 1'e yakın."""
    return (d["HG_KAPANIS"] / d["HG_KAPANIS"].shift(1)) / (d["HGDG_KAPANIS"] / d["HGDG_KAPANIS"].shift(1))


def aligned_capital(ticker: str) -> tuple:
    """Sermayeyi fiyat bölünmesiyle aynı güne hizalar. İş Yatırım bazı bedelsiz artırımlarda sermayeyi fiyattan bir
    işlem günü önce güncellemiş (TURSG 2025-06-03 ve 2026-06-12, BIMAS 2026-05-13): o gün PD = eski fiyat × yeni
    sermaye, yani piyasa değeri bir gün için iki kat görünür. Fiyat bölünmeden gelen sermaye değişimleri (birleşme
    karşılığı pay ihracı, iptal edilen geri alınmış paylar, dönüşüm) gerçektir ve olduğu gibi kalır.
    Döner: (hizalanmış sermaye, düzeltilen günlerin listesi)."""
    d = fetch(ticker)
    s, f = d["SERMAYE"].copy(), _split_factor(d)
    fixes = []
    for i in range(1, len(d)):
        r = s.iloc[i] / s.iloc[i - 1]
        nxt = f.iloc[i + 1] if i + 1 < len(d) else np.nan
        if abs(r - 1) > 1e-9 and abs(f.iloc[i] * r - 1) > TOL and abs(nxt * r - 1) <= TOL:
            fixes.append(dict(date=d.index[i], case="sermaye fiyattan bir gün önce", ratio=r))
            s.iloc[i] = s.iloc[i - 1]                      # değişim ertesi güne (fiyatın bölündüğü güne) ait
        elif abs(r - 1) <= 1e-9 and abs(f.iloc[i] - 1) > TOL and i + 1 < len(d):
            r_next = s.iloc[i + 1] / s.iloc[i]
            if abs(r_next - 1) > 1e-9 and abs(f.iloc[i] * r_next - 1) <= TOL:
                fixes.append(dict(date=d.index[i], case="fiyat sermayeden bir gün önce", ratio=r_next))
                s.iloc[i] = s.iloc[i + 1]
    return s, fixes


def market_cap(ticker: str) -> pd.Series:
    """Günlük piyasa değeri [TL, o günün parası] = düzeltilmemiş kapanış × hizalanmış sermaye. Kapanışta bilinir.
    İş Yatırım'ın PD alanıyla aynıdır; yalnızca sermaye ile fiyatın farklı günlere düştüğü günlerde düzeltilir."""
    d = fetch(ticker)
    s, _ = aligned_capital(ticker)
    return (d["HG_KAPANIS"] * s).rename("market_cap")


def check(ticker: str) -> dict:
    """Denetim: PD = ham kapanış × sermaye; sermaye değişim günlerinde PD sürekli; fiyat önbelleğiyle aynı sürüm."""
    d = fetch(ticker)
    px = _fetch_raw(ticker)["HGDG_KAPANIS"]
    same = d.index.intersection(px.index)
    rel = (d["PD"] / (d["HG_KAPANIS"] * d["SERMAYE"]) - 1).abs()
    s, fixes = aligned_capital(ticker)
    chg = d.index[s.diff().fillna(0) != 0]
    mc_ret = market_cap(ticker).pct_change()
    adj_ret = d["HGDG_KAPANIS"].pct_change()
    split = (_split_factor(d).reindex(chg) * (s / s.shift(1)).reindex(chg) - 1).abs() <= TOL
    jump = (mc_ret - adj_ret).abs().reindex(chg)
    ratio = (d.loc[same, "HGDG_KAPANIS"] / px.loc[same])
    return dict(ticker=ticker, days=len(d), first=d.index.min().date(), last=d.index.max().date(),
                pd_vs_price_x_capital=float(rel.max()),
                adjusted_close_vs_price_cache=("aynı" if np.allclose(ratio, 1, rtol=1e-9) else
                                               # sabit oran: verinin sonundan sonraki bir temettü bütün geçmişi aynı
                                               # katla ölçekler (MGROS 0,9868); getiriler değişmez
                                               f"sabit oran {ratio.iloc[-1]:.4f}" if np.allclose(ratio, ratio.iloc[-1],
                                                                                                rtol=1e-4) else "farklı"),
                capital_changes=len(chg), fixed_days=len(fixes),
                splits=int(split.sum()), max_gap_on_split=float(jump[split].max()) if split.any() else 0.0,
                issuance_or_cancellation=int((~split).sum()))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    todo = [t for t in core_tickers() if not (CACHE / f"{t}.parquet").exists()]
    print(f"indirilecek hisse: {len(todo)} ({len(todo) * len(RANGES)} istek)", flush=True)
    for i, t in enumerate(todo, 1):
        fetch(t)
        print(f"  {i}/{len(todo)} {t}", flush=True)
    print(pd.DataFrame([check(t) for t in core_tickers()]).to_string(index=False))
