import numpy as np
import pandas as pd

from src.data.fundamentals import load_all
from src.data.prices import get_index_history, get_stock_history

START_PERIOD = pd.Timestamp("2022-03-31")
QUAKE_LO, QUAKE_HI = pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-10")
DEADLINE_DAYS = {3: 40, 6: 60, 9: 40, 12: 70}   # dönem ayı -> yasal süre (gün)
MIN_LAG_DAYS = 14                               # açıklama penceresi: dönem sonu + 14 gün ... deadline


def legal_deadline(period_end) -> pd.Timestamp:
    """En geç bu tarihte bilanço kamuya açıktır. Hafta sonuna denk gelirse sonraki iş günü."""
    p = pd.Timestamp(period_end)
    d = p + pd.Timedelta(days=DEADLINE_DAYS[p.month])
    return pd.offsets.BDay().rollforward(d)


def disclosure_table() -> pd.DataFrame:
    """Her (hisse, çeyrek): açıklama penceresinin başı ve yasal son tarih."""
    rows = []
    for t, d in load_all().items():
        for p in d.index[(d.index >= START_PERIOD) & d["net_income"].notna().to_numpy()]:
            rows.append(dict(ticker=t, period_end=p,
                             window_start=p + pd.Timedelta(days=MIN_LAG_DAYS),
                             deadline=legal_deadline(p)))
    return pd.DataFrame(rows)


def visible_asof(table: pd.DataFrame, date) -> pd.DataFrame:
    """Analiz tarihi 'date' iken kullanılabilecek bilançolar (deadline <= date)."""
    return table[table["deadline"] <= pd.Timestamp(date)]


def excess_volume_z(volume: pd.DataFrame, window: int = 60, min_periods: int = 40) -> pd.DataFrame:
    """Hacim z-skoru (sadece geçmiş 60 gün) eksi o günkü tüm hisselerin ortancası."""
    lv = np.log(volume.where(volume > 0))
    past = lv.shift(1).rolling(window, min_periods=min_periods)
    z = (lv - past.mean()) / past.std()
    return z.sub(z.median(axis=1), axis=0)


def season_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True = gün bir açıklama penceresinin içinde (takvim bazlı, tüm hisseler için aynı)."""
    m = pd.Series(False, index=index)
    for y in range(index.min().year - 1, index.max().year + 1):
        for mth in (3, 6, 9, 12):
            p = pd.Timestamp(y, mth, 1) + pd.offsets.MonthEnd(0)
            m |= (index >= p + pd.Timedelta(days=MIN_LAG_DAYS)) & (index <= legal_deadline(p))
    return m


def spike_check() -> pd.DataFrame:
    """Negatif bulgu: hacim / fiyat sıçramaları bilanço sezonunda daha sık mı? (hisse-gün oranları)"""
    tickers = list(load_all())
    px = {t: get_stock_history(t) for t in tickers}
    ex = excess_volume_z(pd.DataFrame({t: px[t]["Volume"] for t in tickers}))
    xu = get_index_history("XU100")["Close"].pct_change()
    absar = pd.DataFrame({t: px[t]["Close"].pct_change() for t in tickers}).sub(xu, axis=0).abs()
    idx = ex.index
    ok = ~((idx >= QUAKE_LO) & (idx <= QUAKE_HI))
    inside = season_mask(idx).to_numpy() & ok
    outside = ~season_mask(idx).to_numpy() & ok
    tests = {"hacim_z>2": ex > 2, "hacim_z>3": ex > 3, "|AR|>%4": absar > 0.04,
             "|AR|>%6": absar > 0.06, "ikisi (z>2, AR>%4)": (ex > 2) & (absar > 0.04)}
    rows = [dict(olcu=k, sezon_ici=v[inside].stack().mean(), sezon_disi=v[outside].stack().mean())
            for k, v in tests.items()]
    out = pd.DataFrame(rows)
    out["oran"] = out["sezon_ici"] / out["sezon_disi"]
    out.attrs["days"] = (int(inside.sum()), int(outside.sum()))
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    tab = disclosure_table()
    tab.to_csv("reports/disclosure_dates.csv", index=False)
    print(f"satır: {len(tab)}  hisse: {tab['ticker'].nunique()}  çeyrek: {tab['period_end'].nunique()}")
    n = tab.groupby("period_end").size()
    print("30 hisseden az veri olan çeyrekler:")
    print(n[n < 30].to_string())
    print("\nASELS son 4 çeyrek:")
    print(tab[tab["ticker"] == "ASELS"].tail(4).to_string(index=False))
    chk = spike_check()
    chk.to_csv("reports/disclosure_spike_check.csv", index=False)
    print(f"\nSıçrama kontrolü (gün sayısı sezon içi / dışı: {chk.attrs['days']}):")
    print(chk.round(4).to_string(index=False))
