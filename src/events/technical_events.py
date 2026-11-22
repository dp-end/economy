
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
    # Adım 3 (22 Kasım 2026): PDF §8'in eksik aileleri
    "bb_mid_hold_up": 1, "bb_mid_hold_down": -1,
    "support_touch": 1, "resistance_touch": -1,
    "cloud_reject_up": 1, "cloud_reject_down": -1,
    "cloud_return_from_below": 1, "cloud_return_from_above": -1,
}   # KAMA-fiyat kesişimi eklenmedi: kama_turn_* ile aynı gün (KAMA eğimi fiyat KAMA'yı geçtiği gün döner; R1 §2)
# Harness yalnızca bu 12 türü kullanır (Adım 3 kararı A: yeni türler önce araştırmada sınanır).
HARNESS_EVENTS = ("bb_lower_reentry", "bb_upper_reentry", "rsi_oversold_exit", "rsi_overbought_exit",
                  "supertrend_up", "supertrend_down", "kama_turn_up", "kama_turn_down",
                  "cloud_break_up", "cloud_break_down", "resistance_break", "support_break")
SR_TOUCH_TOL = 0.015  # destek/direnç kümeleme toleransıyla aynı (src/indicators/support_resistance.py)


def detect_events(f: pd.DataFrame) -> pd.DataFrame:
    """Her gün için olay var mı (True/False). Sadece geçmiş ve o günkü veriyi kullanır."""
    c = f["Close"]
    rsi, d, s, cs = f["rsi_14"], f["st_dir"], f["kama_slope"], f["cloud_state"]
    res, sup = f["resistance"].shift(1), f["support"].shift(1)
    h, l, c1 = f["High"], f["Low"], c.shift(1)
    mid, top, bot = f["bb_mid"], f["cloud_top"], f["cloud_bottom"]
    tol = SR_TOUCH_TOL
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
        # orta bant testi: önceki taraftan gelip orta banda değiyor, kapanış aynı tarafta kalıyor
        "bb_mid_hold_up": (c1 > mid.shift(1)) & (l <= mid) & (c > mid),
        "bb_mid_hold_down": (c1 < mid.shift(1)) & (h >= mid) & (c < mid),
        # destek/direnç teması: dünkü seviyeye %1,5 içinde yaklaşıyor ya da değiyor, kapanış kırmadan kalıyor
        "support_touch": (c1 > sup * (1 + tol)) & (l <= sup * (1 + tol)) & (c > sup),
        "resistance_touch": (c1 < res * (1 - tol)) & (h >= res * (1 - tol)) & (c < res),
        # bulut reddi: bulutun bir tarafında kalıp gün içinde buluta değiyor
        "cloud_reject_up": (cs.shift(1) > 0) & (cs > 0) & (l <= top),
        "cloud_reject_down": (cs.shift(1) < 0) & (cs < 0) & (h >= bot),
        # buluta geri dönüş
        "cloud_return_from_below": (cs.shift(1) < 0) & (cs == 0),
        "cloud_return_from_above": (cs.shift(1) > 0) & (cs == 0),
    }
    return pd.DataFrame(ev, index=f.index).fillna(False).astype(bool)


KAMA_FLAT = 0.5      # % (5 günlük KAMA değişimi); günleri kabaca üçe böler, getiriye bakılarak seçilmedi
RSI_LOW, RSI_HIGH = 35, 65
RECENT = 3           # "son 3 gün" = t-2..t
K2_APPLICABLE = {"bb_lower_reentry", "bb_upper_reentry", "rsi_oversold_exit", "rsi_overbought_exit",
                 "kama_turn_up", "kama_turn_down", "resistance_break", "support_break",
                 "bb_mid_hold_up", "bb_mid_hold_down", "support_touch", "resistance_touch"}   # Supertrend/bulut olaylarında
#                                                                       K2 tanım gereği imkânsız


def states(f: pd.DataFrame) -> pd.DataFrame:
    """Olay günü kapanışındaki gösterge durumları (PDF §8 örnek olay kaydı). Yalnızca o güne kadarki veri.
    rsi_state: son 3 günde RSI ≤ 35 'dip', ≥ 65 'tepe', yoksa 'notr' (dönüş günü RSI çoktan yükselmeye başlamış olur).
    kama_state: 5 günlük KAMA değişimi > +0.5% 'yukselen', < −0.5% 'dusen', arası 'yatay'.
    kama_turn: 5 günlük KAMA değişimi eksi 5 gün önceki 5 günlük değişim (+ = eğim yukarı dönüyor)."""
    rsi = f["rsi_14"]
    rsi_min, rsi_max = rsi.rolling(RECENT).min(), rsi.rolling(RECENT).max()
    rsi_state = np.select([rsi_min <= RSI_LOW, rsi_max >= RSI_HIGH], ["dip", "tepe"], "notr")
    kama5 = (f["kama"] / f["kama"].shift(5) - 1) * 100
    kama_state = np.select([kama5 > KAMA_FLAT, kama5 < -KAMA_FLAT, kama5.notna()], ["yukselen", "dusen", "yatay"],
                           "bilinmiyor")
    st_state = np.select([f["st_dir"] > 0, f["st_dir"] < 0], ["yukari", "asagi"], "bilinmiyor")
    cloud_pos = np.select([f["cloud_state"] > 0, f["cloud_state"] < 0, f["cloud_state"] == 0],
                          ["ustunde", "altinda", "icinde"], "bilinmiyor")
    out = pd.DataFrame({"rsi_state": rsi_state, "kama_state": kama_state, "kama_turn": kama5 - kama5.shift(5),
                        "st_state": st_state, "cloud_pos": cloud_pos,
                        "vol_recent": f["vol_z_20"].rolling(RECENT).max() > 1}, index=f.index)
    out.loc[rsi_min.isna(), "rsi_state"] = "bilinmiyor"
    return out


def combined(direction: int, st: pd.DataFrame) -> pd.DataFrame:
    """Önceden yazılmış birleşik koşullar (Adım 2, 22 Kasım 2026, getirilere bakmadan; PDF §13 ASELS örneği).
    Onaylar (ayı olayında aynası): RSI son 3 günde dip (tepe); hacim son 3 günde z > 1; KAMA eğimi olay yönüne dönüyor.
    K1 destekli = en az 2 onay; k1_all3 = üçü birden (PDF'in harfiyen koşulu, yalnız sayılır).
    K2 trende karşı = Supertrend olayın tersi yönde ve fiyat bulutun olaya ters tarafında."""
    bull = direction > 0
    c_rsi = st["rsi_state"] == ("dip" if bull else "tepe")
    c_vol = st["vol_recent"]
    c_kama = (st["kama_turn"] > 0) if bull else (st["kama_turn"] < 0)
    score = c_rsi.astype(int) + c_vol.astype(int) + c_kama.astype(int)
    k2 = ((st["st_state"] == "asagi") & (st["cloud_pos"] == "altinda")) if bull else \
        ((st["st_state"] == "yukari") & (st["cloud_pos"] == "ustunde"))
    return pd.DataFrame({"conf_rsi": c_rsi, "conf_vol": c_vol, "conf_kama": c_kama, "conf_score": score,
                         "k1_confirmed": score >= 2, "k1_all3": score == 3, "k2_against_trend": k2}, index=st.index)


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
    stt = states(f)
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
        p = p.join(stt.loc[idx]).join(combined(DIRECTION[name], stt.loc[idx]))
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