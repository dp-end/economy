
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.prices import get_index_history,get_stock_history
from src.data.universe import load_universe
from src.indicators.features import full_features

HORIZONS = (1, 3, 5, 10)
DEDUPE_GAP = 5
QUAKE_START = pd.Timestamp("2023-02-06")
QUAKE_END = pd.Timestamp("2023-02-24")

DIRECTION = {  # +1 boğa (yükseliş beklentisi), -1 ayı
    "bb_lower_reentry": 1, "bb_upper_reentry": -1,
    "rsi_oversold_exit": 1, "rsi_overbought_exit": -1,
    "supertrend_up": 1, "supertrend_down": -1,
    "kama_turn_up": 1, "kama_turn_down": -1,
    "cloud_break_up": 1, "cloud_break_down": -1,
    "resistance_break": 1, "support_break": -1,
}


def detect_events(f: pd.DataFrame) -> pd.DataFrame:
    """Her gün için olay var mı (True/False). Sadece geçmiş ve o günkü veriyi kullanır."""
    c = f["Close"]
    rsi, d, s, cs = f["rsi_14"], f["st_dir"], f["kama_slope"], f["cloud_state"]
    res, sup = f["resistance"].shift(1), f["support"].shift(1)
    ev = {
        "bb_lower_reentry": (c.shift(1) < f["bb_lower"].shift(1)) & (c >= f["bb_lower"]),
        "bb_upper_reentry": (c.shift(1) > f["bb_upper"].shift(1)) & (c <= f["bb_upper"]),
        "rsi_oversold_exit": (rsi.shift(1) < 30) & (rsi >= 30),
        "rsi_overbought_exit": (rsi.shift(1) > 70) & (rsi <= 70),
        "supertrend_up": (d.shift(1) < 0) & (d > 0),
        "supertrend_down": (d.shift(1) > 0) & (d < 0),
        "kama_turn_up": (s.shift(1) <= 0) & (s > 0),
        "kama_turn_down": (s.shift(1) >= 0) & (s < 0),
        "cloud_break_up": (cs.shift(1) <= 0) & (cs > 0),
        "cloud_break_down": (cs.shift(1) >= 0) & (cs < 0),
        "resistance_break": (c.shift(1) <= res) & (c > res),
        "support_break": (c.shift(1) >= sup) & (c < sup),
    }
    return pd.DataFrame(ev, index=f.index).fillna(False).astype(bool)


def dedupe(mask: pd.Series, gap: int = DEDUPE_GAP) -> pd.Series:
    """Son tutulan olaydan en fazla `gap` gün sonra gelen olayları at."""
    keep = np.zeros(len(mask), dtype=bool)
    last = -10**9
    for i, v in enumerate(mask.to_numpy()):
        if v and i - last > gap:
            keep[i] = True
            last = i
    return pd.Series(keep, index=mask.index)


def regime(f: pd.DataFrame) -> pd.Series:
    r = pd.Series("unknown", index=f.index)
    ok = f["sma_200"].notna()
    r[ok & (f["Close"] > f["sma_200"])] = "uptrend"
    r[ok & (f["Close"] <= f["sma_200"])] = "downtrend"
    return r


def quake_flags(index: pd.DatetimeIndex, max_h: int = max(HORIZONS)) -> pd.Series:
    """Deprem penceresi: kapanıştan max_h işlem günü önce başlar, 24 Şubat 2023'te biter."""
    pos0 = index.searchsorted(QUAKE_START)
    lo = max(pos0 - max_h, 0)
    flag = (np.arange(len(index)) >= lo) & (index <= QUAKE_END)
    return pd.Series(flag, index=index)


def event_table(ticker: str, xu: pd.Series = None) -> pd.DataFrame:
    f = full_features(ticker)
    xu = get_index_history()["Close"] if xu is None else xu
    c = f["Close"]
    ev = detect_events(f)
    reg, q, vc = regime(f), quake_flags(f.index), f["vol_z_20"] > 1
    fwd = {h: c.shift(-h) / c - 1 for h in HORIZONS}
    exc = {h: fwd[h] - (xu.shift(-h) / xu - 1) for h in HORIZONS}

    parts = []
    for name in ev.columns:
        keep = dedupe(ev[name]).to_numpy()
        idx = f.index[keep]
        if len(idx) == 0:
            continue
        p = pd.DataFrame({
            "ticker": ticker, "event": name, "direction": DIRECTION[name],
            "regime": reg.loc[idx].to_numpy(),
            "vol_confirm": vc.loc[idx].to_numpy(),
            "quake_window": q.loc[idx].to_numpy(),
        }, index=idx)
        for h in HORIZONS:
            p[f"fwd_{h}"] = fwd[h].loc[idx].to_numpy()
            p[f"exc_{h}"] = exc[h].loc[idx].to_numpy()
        parts.append(p)
    out = pd.concat(parts)
    out.index.name = "date"
    return out


@lru_cache(maxsize=1)
def all_events() -> pd.DataFrame:
    u = load_universe()
    core = u[u["is_core"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].tolist()
    xu = get_index_history()["Close"]
    return pd.concat([event_table(t, xu) for t in core]).sort_index()


def baseline() -> pd.DataFrame:
    """Olaysız herhangi bir günün ortalama ham getirisi (%) ve pozitif gün oranı (%). Deprem hariç."""
    u = load_universe()
    core = u[u["is_core"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].tolist()
    vals = {h: [] for h in HORIZONS}
    for t in core:
        c = get_stock_history(t)["Close"]
        q = quake_flags(c.index)
        for h in HORIZONS:
            vals[h].append((c.shift(-h) / c - 1)[~q].dropna())
    out = {}
    for h in HORIZONS:
        s = pd.concat(vals[h])
        out[f"mean_{h}"] = s.mean() * 100
        out[f"hit_{h}"] = (s > 0).mean() * 100
    return pd.Series(out).to_frame("baz_gun").T


def summarize(ev: pd.DataFrame, by=("event",), kind: str = "fwd", horizons=HORIZONS) -> pd.DataFrame:
    """Yön düzeltmeli ortalama getiri (%) ve isabet oranı (%). Deprem penceresi hariç.
    kind='fwd' ham getiri, kind='exc' XU100'e göre fazla getiri."""
    ev = ev[~ev["quake_window"]]
    rows = []
    for keys, g in ev.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["n"] = len(g)
        for h in horizons:
            s = (g["direction"] * g[f"{kind}_{h}"]).dropna()
            row[f"mean_{h}"] = s.mean() * 100
            row[f"hit_{h}"] = (s > 0).mean() * 100
            if h == 5:
                row["t_5"] = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index(list(by))


if __name__ == "__main__":
    ev = all_events()
    ev.reset_index().to_parquet("reports/technical_events.parquet")
    print("toplam olay:", len(ev), "| deprem penceresindeki:", int(ev["quake_window"].sum()))
    pd.set_option("display.width", 200)
    for kind in ("fwd", "exc"):
        s = summarize(ev, by=("event",), kind=kind)
        s.to_csv(f"reports/event_summary_{kind}.csv")
        print(f"\n--- {kind}: yön düzeltmeli ortalama getiri %, isabet %, deprem hariç ---")
        print(s.round(2).to_string())
    summarize(ev, by=("event", "regime"), kind="fwd").to_csv("reports/event_summary_regime.csv")
    print("\n--- baz çizgisi: olaysız herhangi bir gün (ham getiri %, pozitif gün %) ---")
    print(baseline().round(2).to_string())