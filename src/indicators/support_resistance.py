
import numpy as np
import pandas as pd


def find_pivots(df: pd.DataFrame, k: int = 5):
    """Yerel tepe/dip. Pivot i günü, ancak i+k gününde onaylanır."""
    h, l = df["High"], df["Low"]
    ph = h.where(h == h.rolling(2 * k + 1, center=True).max())
    pl = l.where(l == l.rolling(2 * k + 1, center=True).min())
    return ph, pl


def sr_features(df: pd.DataFrame, k: int = 5, lookback: int = 250, tol: float = 0.015) -> pd.DataFrame:
    """Her gün için, o güne kadar onaylanmış pivotlardan en yakın destek ve direnç.
    Seviye gücü = kümedeki pivot sayısı (touches). Yenilik = en son pivottan beri geçen gün."""
    ph, pl = find_pivots(df, k)
    pos = np.concatenate([np.where(ph.notna())[0], np.where(pl.notna())[0]])
    px = np.concatenate([ph.dropna().to_numpy(), pl.dropna().to_numpy()])
    close = df["Close"].to_numpy(dtype=float)
    n = len(df)
    names = ["support", "support_touches", "support_age",
             "resistance", "resistance_touches", "resistance_age"]
    out = {c: np.full(n, np.nan) for c in names}

    for t in range(n):
        m = (pos + k <= t) & (pos >= t - lookback)
        if not m.any():
            continue
        order = np.argsort(px[m])
        p, q = px[m][order], pos[m][order]
        groups = [[(p[0], q[0])]]
        for v, i in zip(p[1:], q[1:]):
            if v <= np.mean([x[0] for x in groups[-1]]) * (1 + tol):
                groups[-1].append((v, i))
            else:
                groups.append([(v, i)])
        centers = np.array([np.mean([x[0] for x in g]) for g in groups])
        counts = np.array([len(g) for g in groups])
        last = np.array([max(x[1] for x in g) for g in groups])

        below = np.where(centers < close[t])[0]
        above = np.where(centers > close[t])[0]
        if len(below):
            j = below[-1]
            out["support"][t], out["support_touches"][t], out["support_age"][t] = centers[j], counts[j], t - last[j]
        if len(above):
            j = above[0]
            out["resistance"][t], out["resistance_touches"][t], out["resistance_age"][t] = centers[j], counts[j], t - last[j]

    res = pd.DataFrame(out, index=df.index)
    res["dist_support"] = df["Close"] / res["support"] - 1       # pozitif: destekten ne kadar yukarıda
    res["dist_resistance"] = res["resistance"] / df["Close"] - 1  # pozitif: dirence ne kadar var
    return res