from functools import lru_cache

import pandas as pd

from src.data.prices import get_stock_history, get_index_history
from src.data.universe import load_universe, build_sector_peer_groups

HORIZONS = (5, 10, 20)


def relative_to_index(panel: pd.DataFrame, xu_close: pd.Series, n: int) -> pd.DataFrame:
    """Hisse n günlük getirisi - XU100 n günlük getirisi."""
    return panel.pct_change(n).sub(xu_close.pct_change(n), axis=0)


def sector_lag(panel: pd.DataFrame, groups: dict, n: int) -> pd.DataFrame:
    """Hisse n günlük getirisi - kendisi HARİÇ sektör ortancası. Negatif = sektörün gerisinde."""
    r = panel.pct_change(n)
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for members in groups.values():
        for t in members:
            peers = [m for m in members if m != t]
            if peers:
                out[t] = r[t] - r[peers].median(axis=1)
    return out


@lru_cache(maxsize=1)
def _all_features() -> dict:
    tickers = load_universe()["ticker"].tolist()
    panel = pd.DataFrame({t: get_stock_history(t)["Close"] for t in tickers})
    xu = get_index_history()["Close"]
    groups, _ = build_sector_peer_groups()
    parts = {}
    for n in HORIZONS:
        parts[f"rel_xu_{n}"] = relative_to_index(panel, xu, n)
        parts[f"sector_lag_{n}"] = sector_lag(panel, groups, n)
    return {t: pd.DataFrame({k: v[t] for k, v in parts.items()}) for t in tickers}


def relative_features(ticker: str) -> pd.DataFrame:
    return _all_features()[ticker.upper()]