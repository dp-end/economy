
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - \
           close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper, lower = mid + k * std, mid - k * std
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "bb_bandwidth": (upper - lower) / mid,
        "bb_pctb": (close - lower) / (upper - lower),
    })


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr.iloc[0] = np.nan  # ilk gün önceki kapanış yok
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    c = close.to_numpy(dtype=float)
    change = close.diff(n).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = (change / volatility).to_numpy()
    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out, index=close.index)
    out[n - 1] = c[n - 1]
    for i in range(n, len(c)):
        if np.isnan(sc[i]):
            out[i] = out[i - 1]
        else:
            out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    hl2 = (df["High"] + df["Low"]) / 2
    a = atr(df, period)
    ub = (hl2 + mult * a).to_numpy()
    lb = (hl2 - mult * a).to_numpy()
    c = df["Close"].to_numpy(dtype=float)
    n = len(df)
    fub = np.full(n, np.nan)
    flb = np.full(n, np.nan)
    st = np.full(n, np.nan)
    d = np.full(n, np.nan)
    valid = np.where(~np.isnan(ub))[0]
    if len(valid):
        first = valid[0]
        for i in range(first, n):
            if i == first:
                fub[i], flb[i], d[i] = ub[i], lb[i], 1.0
            else:
                fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
                flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
                if c[i] > fub[i - 1]:
                    d[i] = 1.0
                elif c[i] < flb[i - 1]:
                    d[i] = -1.0
                else:
                    d[i] = d[i - 1]
            st[i] = flb[i] if d[i] == 1.0 else fub[i]
    direction = pd.Series(d, index=df.index)
    flip = np.sign(direction.diff()).fillna(0)  # +1: yükselişe dönüş, -1: düşüşe dönüş
    return pd.DataFrame({"supertrend": st, "st_dir": direction, "st_flip": flip}, index=df.index)


def ichimoku(df: pd.DataFrame, t: int = 9, k: int = 26, b: int = 52) -> pd.DataFrame:
    """Chikou bilerek yok (geleceğe bakar). Bulut, t gününde sadece geçmiş veriyle bilinir."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tenkan = (h.rolling(t).max() + l.rolling(t).min()) / 2
    kijun = (h.rolling(k).max() + l.rolling(k).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(k)
    senkou_b = ((h.rolling(b).max() + l.rolling(b).min()) / 2).shift(k)
    pair = pd.concat([senkou_a, senkou_b], axis=1)
    top = pair.max(axis=1, skipna=False)
    bottom = pair.min(axis=1, skipna=False)
    state = pd.Series(np.where(c > top, 1.0, np.where(c < bottom, -1.0, 0.0)), index=df.index)
    state = state.where(top.notna())  # 1: bulutun üstünde, -1: altında, 0: içinde
    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
        "cloud_top": top, "cloud_bottom": bottom, "cloud_state": state,
    }, index=df.index)


def volume_zscore(volume: pd.Series, n: int = 20) -> pd.Series:
    r = volume.rolling(n, min_periods=int(n * 0.75))
    return (volume - r.mean()) / r.std()


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """High/Low/Close/Volume tablosuna tüm göstergeleri ekler."""
    out = df.copy()
    c = df["Close"]
    for n in (20, 50, 200):
        out[f"sma_{n}"] = sma(c, n)
    for n in (20, 50):
        out[f"ema_{n}"] = ema(c, n)
    out["rsi_14"] = rsi(c, 14)
    out = out.join(macd(c))
    out = out.join(bollinger(c))
    out["atr_14"] = atr(df, 14)
    out["kama"] = kama(c)
    out["kama_slope"] = out["kama"].pct_change()
    out = out.join(supertrend(df))
    out = out.join(ichimoku(df))
    out["vol_z_20"] = volume_zscore(df["Volume"])
    out["ret_1d"] = c.pct_change()
    out["ret_5d"] = c.pct_change(5)
    out["vol_20d"] = out["ret_1d"].rolling(20).std() * np.sqrt(252)
    return out