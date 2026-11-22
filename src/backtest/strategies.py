
from functools import lru_cache

import numpy as np
import pandas as pd

from src.backtest.engine import (COST_ONE_WAY, benchmarks, calculate_performance, load_returns,
                                 run_backtest, sector_neutral, split_halves, trade_log)
from src.data.fundamentals import load_all
from src.data.macro import macro_panel
from src.data.prices import get_index_history
from src.events.earnings_reaction import earnings_events
from src.events.news_reaction import news_flags
from src.indicators.features import full_features

TOP_K = 10                 # portföyde en fazla hisse; her biri 1/TOP_K ağırlık, boş kalan pay nakit
WARMUP = 250               # ilk 250 gün göstergelerin ısınması için değerlendirme dışı
SHOCK_LIMIT, SHOCK_DAYS = 0.095, 10      # son 10 günde ±%9,5 (BIST günlük limiti) hareket = şok
TREND_SMA = 200            # piyasa trend filtresi: XU100 200 günlük ortalamasının altında mı
FX_WINDOW, FX_LOOKBACK, FX_PCT = 20, 250, 0.90   # kur: 20 günlük artış, kendi son 250 gününün %90'lık diliminde mi
RATE_LOOKBACK = 126        # faiz döngüsü: politika faizi ~6 ay öncesinden yüksek mi
COSTS = (0.0, COST_ONE_WAY, 2 * COST_ONE_WAY)
WINDOW = 126              # Adım 5: kayan pencere ve walk-forward test penceresi (~6 ay)
TRAIN = 252               # walk-forward seçim penceresi (~12 ay, test penceresinden hemen önce)
POS_NEWS_BONUS = 0.5      # E: son 10 günde olumlu KAP haberi olan hisse sıralamada bu kadar öne alınır (sonuçtan önce sabitlendi)


@lru_cache(maxsize=1)
def panels() -> dict:
    """Tarih x hisse tabloları. Her hücre yalnızca o gün kapanışta bilinen veriyi içerir."""
    tickers = list(load_all())
    f = {t: full_features(t) for t in tickers}
    col = lambda c: pd.DataFrame({t: f[t][c] for t in tickers})
    close, ret = col("Close"), col("Close").pct_change()
    p = dict(close=close, ret=ret, st_dir=col("st_dir"), sma_50=col("sma_50"), kama_slope=col("kama_slope"),
             cloud=col("cloud_state"), rel=col("rel_xu_20"), sector_lag=col("sector_lag_10"), atr=col("atr_14"))
    p["shock"] = (ret.abs() >= SHOCK_LIMIT).astype(float).rolling(SHOCK_DAYS).sum() >= 1
    # son YAYINLANMIŞ bilançonun sıralaması (event_day'den itibaren görünür)
    ev = earnings_events().dropna(subset=["event_day"]).sort_values("event_day")
    weak = pd.DataFrame(np.nan, index=close.index, columns=tickers)
    for t, g in ev.groupby("ticker"):
        s = g.drop_duplicates("event_day", keep="last").set_index("event_day")["tercile"].map(lambda x: float(x == "alt"))
        weak[t] = s.reindex(close.index.union(s.index)).ffill().reindex(close.index)
    p["weak_earnings"] = weak.fillna(0).astype(bool)
    p["neg_news"], p["pos_news"] = news_flags(close.index, tickers)     # event_day'den itibaren 10 işlem günü
    return p


def weekly(mask_or_weights: pd.DataFrame) -> pd.DataFrame:
    """Haftalık dengeleme: haftanın son işlem günündeki karar bir sonraki haftanın sonuna kadar taşınır."""
    idx = mask_or_weights.index
    week = idx.to_series().dt.to_period("W")
    last = (week != week.shift(-1)).to_numpy()
    return mask_or_weights.where(pd.Series(last, index=idx), axis=0).ffill().fillna(0.0)


def pick_top(eligible: pd.DataFrame, score: pd.DataFrame, k: int = TOP_K, weight_by: pd.DataFrame = None) -> pd.DataFrame:
    """Uygun hisseler içinden skora göre en iyi k. Ağırlık 1/k (eşit) ya da weight_by ile orantılı, toplam n/k."""
    s = score.where(eligible)
    chosen = s.rank(axis=1, ascending=False, method="first") <= k
    chosen &= s.notna()
    if weight_by is None:
        return chosen.astype(float) / k
    raw = weight_by.where(chosen)
    return raw.div(raw.sum(axis=1), axis=0).fillna(0.0).mul(chosen.sum(axis=1) / k, axis=0)


def macro_scale(index: pd.DatetimeIndex, use_rate: bool = False) -> pd.Series:
    """1.0 ya da 0.5: piyasa trendi zayıfsa veya kur kendi geçmişine göre hızlı artıyorsa yarı ağırlık."""
    xu = get_index_history("XU100")["Close"].reindex(index)
    m = macro_panel().reindex(index)
    weak_trend = xu < xu.rolling(TREND_SMA).mean()
    fx = m["usdtry"].pct_change(FX_WINDOW)
    fx_stress = fx >= fx.rolling(FX_LOOKBACK).quantile(FX_PCT).shift(1)       # eşik dünkü geçmişten
    risk = weak_trend | fx_stress
    if use_rate:
        risk |= m["policy_rate"] > m["policy_rate"].shift(RATE_LOOKBACK)      # sıkılaşma dönemi
    return pd.Series(np.where(risk, 0.5, 1.0), index=index)


def generate_strategy_variants() -> dict:
    """Ad -> hedef ağırlık tablosu (t günü kapanış kararı). Kurallar tarih ve sabit fiyat seviyesi içermez."""
    p = panels()
    trend_ok = (p["st_dir"] > 0) & (p["close"] > p["sma_50"])
    bear = (p["kama_slope"] < 0) | (p["cloud"] < 0)
    a_ok = trend_ok & ~bear
    b_ok = a_ok & ~p["shock"]
    laggard = p["sector_lag"].rank(axis=1, pct=True) <= 1 / 3                 # sektörüne göre en geride kalan üçte bir
    rel_rank = p["rel"].rank(axis=1, pct=True)
    b_score = rel_rank + laggard.astype(float)                                # geride kalanlar sıralamada öne alınır
    c_ok = b_ok & ~p["weak_earnings"]
    idx = p["close"].index
    inv_vol = 1 / (p["atr"] / p["close"])
    e_ok = c_ok & ~p["neg_news"]                                              # olumsuz haberli hisse dışarıda
    e_score = b_score + POS_NEWS_BONUS * p["pos_news"].astype(float)
    w = {
        "A_teknik": pick_top(a_ok, rel_rank),
        "B_sektor": pick_top(b_ok, b_score),
        "C_temel": pick_top(c_ok, b_score),
        "D_makro": pick_top(c_ok, b_score).mul(macro_scale(idx), axis=0),
        "D_makro_oynaklik": pick_top(c_ok, b_score, weight_by=inv_vol).mul(macro_scale(idx), axis=0),
        "D_makro_faiz": pick_top(c_ok, b_score).mul(macro_scale(idx, use_rate=True), axis=0),
        "E_haber": pick_top(e_ok, e_score).mul(macro_scale(idx), axis=0),
    }
    return {k: weekly(v) for k, v in w.items()}


def weekday_test_weights() -> pd.DataFrame:
    """Adım 6 kontrolü: Çarşamba kapanışında 30 hisseyi eşit ağırlık al, Perşembe kapanışında sat."""
    ret = load_returns()
    w = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    w[ret.index.day_of_week == 2] = 1 / ret.shape[1]
    return w


def evaluate(weights: dict = None, cost: float = COST_ONE_WAY) -> pd.DataFrame:
    """Her strateji: tüm dönem, ilk yarı, ikinci yarı. Kıyas: eşit ağırlıklı al-tut ve XU100."""
    ret = load_returns().iloc[WARMUP:]
    bm = benchmarks(ret)
    rows = []
    for name, w in (weights or generate_strategy_variants()).items():
        w = w.reindex(ret.index)
        bt = run_backtest(w, ret, cost=cost)
        sec = sector_neutral(w, ret)
        first, second = split_halves(bt)
        for part, x in (("tum", bt), ("ilk_yari", first), ("ikinci_yari", second)):
            tr = trade_log(w.loc[x.index], ret.loc[x.index], cost=cost)       # pencere içindeki işlemler
            r = calculate_performance(x, bm["esit_agirlik"], sec, tr)
            r["xu100_fark"] = r["toplam_getiri"] - float((1 + bm["xu100"].reindex(x.index).fillna(0)).prod() - 1)
            rows.append({"strateji": name, "donem": part, **r})
    return pd.DataFrame(rows).set_index(["strateji", "donem"])


def windows(index: pd.DatetimeIndex, size: int = WINDOW) -> list:
    """Değerlendirme dönemini art arda, çakışmayan pencerelere böler (son kısa parça atılır)."""
    return [index[i:i + size] for i in range(0, len(index) - size + 1, size)]


def rolling_table(weights: dict = None, cost: float = COST_ONE_WAY) -> pd.DataFrame:
    """Her strateji x her ~6 aylık pencere: getiri ve XU100, eşit ağırlık, sektör nötr kıyasa göre fark."""
    ret = load_returns().iloc[WARMUP:]
    bm = benchmarks(ret)
    rows = []
    for name, w in (weights or generate_strategy_variants()).items():
        w = w.reindex(ret.index)
        net = run_backtest(w, ret, cost=cost)["net"]
        sec = sector_neutral(w, ret)
        for k, ix in enumerate(windows(ret.index)):
            tot = lambda x: float((1 + x.loc[ix].fillna(0)).prod() - 1)
            rows.append(dict(strateji=name, pencere=k + 1, bas=ix[0].date(), bit=ix[-1].date(), getiri=tot(net),
                             xu100=tot(bm["xu100"]), esit_agirlik=tot(bm["esit_agirlik"]), sektor_notr=tot(sec)))
    d = pd.DataFrame(rows)
    for b in ("xu100", "esit_agirlik", "sektor_notr"):
        d[f"fark_{b}"] = d["getiri"] - d[b]
    return d


def walk_forward(weights: dict = None, cost: float = COST_ONE_WAY, train: int = TRAIN) -> tuple:
    """Kayan seçim (Adım 5): her test penceresinin başında, hemen önceki `train` günde net Sharpe'ı en yüksek
    strateji seçilir ve pencere boyunca onunla gidilir. Seçim yalnızca pencere başından önceki getirileri kullanır.
    Birleşik ağırlıklarla tek backtest (geçiş maliyetleri dahil). Döner: (pencere tablosu, özet tablo)."""
    weights = weights or generate_strategy_variants()
    ret = load_returns().iloc[WARMUP:]
    bm = benchmarks(ret)
    nets = {n: run_backtest(w.reindex(ret.index), ret, cost=cost)["net"] for n, w in weights.items()}
    wins = windows(ret.index)
    first = next(i for i, ix in enumerate(wins) if ret.index.get_loc(ix[0]) >= train)
    stitched = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    rows = []
    for k in range(first, len(wins)):
        ix = wins[k]
        pos = ret.index.get_loc(ix[0])
        past = ret.index[pos - train:pos]                                   # pencere başından ÖNCEKİ günler
        sh = {n: x.loc[past].mean() / x.loc[past].std(ddof=1) * np.sqrt(252) for n, x in nets.items()}
        pick = max(sh, key=sh.get)
        stitched.loc[ix] = weights[pick].reindex(ix).fillna(0.0)
        rows.append(dict(pencere=k + 1, bas=ix[0].date(), bit=ix[-1].date(), secilen=pick,
                         egitim_sharpe=sh[pick], secilen_getiri=float((1 + nets[pick].loc[ix]).prod() - 1),
                         en_iyi_sonradan=max(nets, key=lambda n: (1 + nets[n].loc[ix]).prod()),
                         xu100=float((1 + bm["xu100"].loc[ix].fillna(0)).prod() - 1)))
    test_ix = ret.index[ret.index.get_loc(wins[first][0]):ret.index.get_loc(wins[-1][-1]) + 1]
    # pencere başında hedef ağırlık değişir; ilk günün getirisi bir önceki günün kararıyla alınır (kaydırma)
    bt = run_backtest(stitched.loc[test_ix], ret.loc[test_ix], cost=cost)
    tot = lambda x: float((1 + x.reindex(test_ix).fillna(0)).prod() - 1)
    summ = [dict(yontem="walk-forward secim", toplam_getiri=tot(bt["net"]),
                 sharpe=float(bt["net"].mean() / bt["net"].std(ddof=1) * np.sqrt(252)),
                 max_dusus=float(((1 + bt["net"]).cumprod() / (1 + bt["net"]).cumprod().cummax() - 1).min()))]
    for n, x in nets.items():
        xs = x.reindex(test_ix)
        summ.append(dict(yontem=f"sabit {n}", toplam_getiri=tot(xs), sharpe=float(xs.mean() / xs.std(ddof=1) * np.sqrt(252)),
                         max_dusus=float(((1 + xs).cumprod() / (1 + xs).cumprod().cummax() - 1).min())))
    for b in ("xu100", "esit_agirlik"):
        xs = bm[b].reindex(test_ix).fillna(0)
        summ.append(dict(yontem=f"kiyas {b}", toplam_getiri=tot(xs), sharpe=float(xs.mean() / xs.std(ddof=1) * np.sqrt(252)),
                         max_dusus=float(((1 + xs).cumprod() / (1 + xs).cumprod().cummax() - 1).min())))
    return pd.DataFrame(rows), pd.DataFrame(summ)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ret = load_returns().iloc[WARMUP:]
    bm = benchmarks(ret)
    print(f"değerlendirme: {ret.index.min().date()} - {ret.index.max().date()} ({len(ret)} gün)")
    for part, x in (("tüm", bm), ("ilk yarı", bm.iloc[:len(bm) // 2]), ("ikinci yarı", bm.iloc[len(bm) // 2:])):
        tot = (1 + x.fillna(0)).prod() - 1
        print(f"kıyas {part:12s} XU100 %{tot['xu100'] * 100:,.0f}   eşit ağırlık %{tot['esit_agirlik'] * 100:,.0f}")
    res = evaluate()
    res.to_csv("reports/backtest_strategies.csv")
    variants = generate_strategy_variants()
    ret_e = load_returns().iloc[WARMUP:]
    pd.concat([trade_log(w.reindex(ret_e.index), ret_e).assign(strateji=n) for n, w in variants.items()]) \
        .to_csv("reports/backtest_trades.csv", index=False)
    rolling_table(variants).to_csv("reports/backtest_rolling.csv", index=False)
    wf, wf_sum = walk_forward(variants)
    wf.to_csv("reports/backtest_walkforward.csv", index=False)
    wf_sum.to_csv("reports/backtest_walkforward_summary.csv", index=False)
    print("\nWalk-forward seçim:\n" + wf.round(3).to_string() + "\n" + wf_sum.round(3).to_string())
    cols = ["toplam_getiri", "sharpe", "max_dusus", "piyasada_gun", "islem_sayisi", "toplam_maliyet", "fark", "bilgi_orani", "xu100_fark"]
    print("\nStratejiler (maliyet tek yön %0,15; 'fark' = eşit ağırlıklı al-tut kıyasına göre toplam getiri farkı):")
    print(res[cols].astype(float).round(2).to_string())
    print("\nMaliyet duyarlılığı (tüm dönem toplam getiri):")
    sens = pd.DataFrame({f"maliyet_{c:.4f}": evaluate(cost=c).xs("tum", level="donem")["toplam_getiri"] for c in COSTS})
    sens.to_csv("reports/backtest_cost_sensitivity.csv")
    print(sens.astype(float).round(2).to_string())
    print("\nÇarşamba al - Perşembe sat (30 hisse eşit ağırlık):")
    wk = {"Car_Per": weekday_test_weights()}
    for c in COSTS:
        r = evaluate(wk, cost=c)
        print(f"  maliyet {c:.4f}:", r[["toplam_getiri", "sharpe", "islem_sayisi"]].astype(float).round(2).to_dict("index"))
