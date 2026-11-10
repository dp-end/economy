
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.universe import load_universe

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "isyatirim"
REPORT_DIR = ROOT / "reports"

URL = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
RANGES = [("01-11-2021", "31-10-2022"), ("01-11-2022", "31-10-2023"),
          ("01-11-2023", "31-10-2024"), ("01-11-2024", "31-10-2025"),
          ("01-11-2025", "06-11-2026")]
COLS = ["HGDG_KAPANIS", "HGDG_AOF", "HGDG_MIN", "HGDG_MAX", "HGDG_HACIM", "SERMAYE", "END_DEGER"]


def _fetch_raw(ticker: str, force: bool = False) -> pd.DataFrame:
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
    df = df[~df.index.duplicated()][COLS].astype(float)
    df.to_parquet(path)
    return df


def get_stock_history(ticker: str, force: bool = False) -> pd.DataFrame:
    """Günlük High/Low/Close/Volume (lot) ve TL hacim. Open yok."""
    t = ticker.upper()
    if t not in set(load_universe()["ticker"]):
        raise ValueError(f"Evrende olmayan ticker: {ticker}")
    raw = _fetch_raw(t, force)
    close = raw["HGDG_KAPANIS"]
    df = pd.DataFrame({
        "High": np.maximum(raw["HGDG_MAX"], close),  # kapanış seansı aralık dışı kalabilir
        "Low": np.minimum(raw["HGDG_MIN"], close),
        "Close": close,
        "Volume": (raw["HGDG_HACIM"] / raw["HGDG_AOF"]).replace([np.inf, -np.inf], np.nan),
        "Turnover_TL": raw["HGDG_HACIM"],
    }, index=raw.index)
    df.index.name = "date"
    return df


def get_index_history(index: str = "XU100", force: bool = False) -> pd.DataFrame:
    """XU100 sadece kapanış (İş Yatırım endeks değeri)."""
    if index != "XU100":
        raise ValueError(f"Bilinmeyen endeks: {index}")
    close = _fetch_raw("ASELS", force)["END_DEGER"]
    out = pd.DataFrame({"Close": close})
    out.index.name = "date"
    return out


def check_price_quality(df: pd.DataFrame, name: str, cal=None) -> dict:
    ret = df["Close"].pct_change()
    out = {
        "name": name,
        "rows": len(df),
        "first": df.index.min().date(),
        "last": df.index.max().date(),
        "nan_close": int(df["Close"].isna().sum()),
        "bad_close": int((df["Close"] <= 0).sum()),
        "max_abs_ret": round(float(ret.abs().max()), 3),
        "ret_gt_11pct": int((ret.abs() > 0.11).sum()),  # BIST günlük limit %10
    }
    if "Volume" in df:
        out["missing_volume"] = int((df["Volume"].isna() | (df["Volume"] <= 0)).sum())
    if cal is not None:
        out["missing_vs_calendar"] = len(cal.difference(df.index))
    return out


def download_all(force: bool = False) -> pd.DataFrame:
    xu = get_index_history(force=force)
    rows = [check_price_quality(xu, "XU100")]
    for t in load_universe()["ticker"]:
        try:
            rows.append(check_price_quality(get_stock_history(t, force), t, xu.index))
        except Exception as e:
            rows.append({"name": t, "error": str(e)})
    rep = pd.DataFrame(rows)
    REPORT_DIR.mkdir(exist_ok=True)
    rep.to_csv(REPORT_DIR / "price_quality.csv", index=False)
    return rep


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(download_all().to_string())