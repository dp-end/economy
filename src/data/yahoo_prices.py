
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.universe import load_universe

ROOT = Path(__file__).resolve().parents[2]
PRICE_DIR = ROOT / "data" / "prices"
REPORT_DIR = ROOT / "reports"

START = "2021-11-01"
END = "2026-11-09"  # yfinance'ta bitiş dahil değil
INDEX_SYMBOLS = {"XU100": "XU100.IS"}

# Yahoo'nun düzeltmediği bölünmeler: ticker -> [(bölünme günü, oran)]
# CCOLA: %1000 bedelsiz (1 hisse -> 11 hisse), düşüş veride 2024-08-01'de görünüyor
MANUAL_SPLITS = {"CCOLA": [("2024-08-01", 11.0)]}

# Gerçek olaylar: sıçrama alarmı vermesin ama raporda not olsun
KNOWN_EVENT_DATES = {
    pd.Timestamp("2023-02-15"): "6 Şubat 2023 depremi sonrası borsa kapanışı (9-14 Şubat), yeniden açılış günü",
}


def _download(symbol: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(start=START, end=END, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Veri yok: {symbol}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _load_cached(name: str, symbol: str, force: bool = False) -> pd.DataFrame:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICE_DIR / f"{name}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    df = _download(symbol)
    df.to_parquet(path)
    return df


def clean_prices(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Ham Yahoo verisini temizler (önbellek ham kalır)."""
    df = df.copy()
    df["Volume"] = df["Volume"].astype(float)

    # 1) Yer tutucu satırlar: hacim 0 ve OHLC sabit -> işlem yok
    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    df = df[~((df["Volume"] == 0) & flat)]

    # 2) Yahoo'nun düzeltmediği bölünmeler
    for day, ratio in MANUAL_SPLITS.get(ticker, []):
        before = df.index < pd.Timestamp(day)
        df.loc[before, ["Open", "High", "Low", "Close"]] /= ratio
        df.loc[before, "Volume"] *= ratio

    # 3) Kalan sıfır hacim = eksik hacim verisi
    df["Volume"] = df["Volume"].replace(0, np.nan)
    return df


@lru_cache(maxsize=1)
def market_calendar() -> pd.DatetimeIndex:
    """İşlem günü = hisselerin en az yarısında (yer tutucu olmayan) satır olan gün."""
    uni = load_universe()
    flags = []
    for t, sym in zip(uni["ticker"], uni["yahoo"]):
        df = clean_prices(_load_cached(t, sym), t)
        flags.append(pd.Series(1, index=df.index))
    cnt = pd.concat(flags, axis=1).sum(axis=1)
    cal = cnt[cnt >= 0.5 * len(uni)].index
    return pd.DatetimeIndex(cal, name="date")


def get_index_history(index: str = "XU100", force: bool = False) -> pd.DataFrame:
    if index not in INDEX_SYMBOLS:
        raise ValueError(f"Bilinmeyen endeks: {index}")
    raw = _load_cached(index, INDEX_SYMBOLS[index], force)
    df = clean_prices(raw)
    return df[df.index.isin(market_calendar())]


def get_stock_history(ticker: str, force: bool = False) -> pd.DataFrame:
    uni = load_universe()
    row = uni[uni["ticker"] == ticker.upper()]
    if row.empty:
        raise ValueError(f"Evrende olmayan ticker: {ticker}")
    raw = _load_cached(ticker.upper(), row.iloc[0]["yahoo"], force)
    df = clean_prices(raw, ticker.upper())
    return df[df.index.isin(market_calendar())]


def check_price_quality(df: pd.DataFrame, name: str, cal=None) -> dict:
    ret = df["Close"].pct_change()
    tol = 1e-6 * df["Close"]
    bad_ohlc = (df["High"] < df[["Open", "Close", "Low"]].max(axis=1) - tol) | (
        df["Low"] > df[["Open", "Close", "High"]].min(axis=1) + tol
    )
    jumps = (ret.abs() > 0.12).values
    known = ret.index.isin(list(KNOWN_EVENT_DATES))
    out = {
        "name": name,
        "rows": len(df),
        "first": df.index.min().date(),
        "last": df.index.max().date(),
        "nan_cells": int(df[["Open", "High", "Low", "Close"]].isna().sum().sum()),
        "bad_close": int((df["Close"] <= 0).sum()),
        "missing_volume": int(df["Volume"].isna().sum()),
        "max_abs_ret": round(float(ret.abs().max()), 3),
        "jumps_gt_12pct": int(jumps.sum()),
        "unexplained_jumps": int((jumps & ~known).sum()),
        "bad_ohlc": int(bad_ohlc.sum()),
    }
    if cal is not None:
        out["missing_vs_calendar"] = len(cal.difference(df.index))
    return out


def download_all(force: bool = False) -> pd.DataFrame:
    get_index_history(force=force)
    for t in load_universe()["ticker"]:
        try:
            get_stock_history(t, force)
        except Exception as e:
            print("HATA", t, e)
        time.sleep(0.5)
    market_calendar.cache_clear()
    cal = market_calendar()
    rows = [check_price_quality(get_index_history(), "XU100", cal)]
    for t in load_universe()["ticker"]:
        try:
            rows.append(check_price_quality(get_stock_history(t), t, cal))
        except Exception as e:
            rows.append({"name": t, "error": str(e)})
    rep = pd.DataFrame(rows)
    REPORT_DIR.mkdir(exist_ok=True)
    rep.to_csv(REPORT_DIR / "price_quality.csv", index=False)
    print("İşlem günü sayısı (takvim):", len(cal))
    return rep


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(download_all().to_string())