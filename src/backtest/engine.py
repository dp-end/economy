
import numpy as np
import pandas as pd

from src.data.fundamentals import load_all
from src.data.prices import get_index_history, get_stock_history

COST_ONE_WAY = 0.0015      # tek yön işlem maliyeti (komisyon + kayma); gidiş-dönüş %0,30
MIN_TRADES = 30            # bundan az işlemli sonuç "az sinyal" uyarısı alır
DAYS = 252


def load_returns(tickers=None) -> pd.DataFrame:
    """Günlük kapanıştan kapanışa getiriler (tarih x hisse). Varsayılan: 30 çekirdek hisse."""
    tickers = tickers or list(load_all())
    return pd.DataFrame({t: get_stock_history(t)["Close"] for t in tickers}).pct_change()


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost: float = COST_ONE_WAY, shift: int = 1) -> pd.DataFrame:
    """weights: t günü KAPANIŞINDA verilen hedef ağırlıklar (0..1, satır toplamı <= 1; kalan nakit).
    shift=1: t günü kararı t+1 günü getirisini alır. shift=0 YANLIŞTIR, sadece kasıtlı hata testi için."""
    w = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    if (w < -1e-12).any().any() or (w.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("ağırlıklar 0..1 aralığında ve satır toplamı en fazla 1 olmalı")
    held = w.shift(shift).fillna(0.0)                    # o gün gerçekten taşınan pozisyon
    prev = held.shift(1).fillna(0.0)
    out = pd.DataFrame(index=returns.index)
    out["gross"] = (held * returns.fillna(0.0)).sum(axis=1)
    out["turnover"] = (held - prev).abs().sum(axis=1)    # her gün hedef ağırlığa dönülür (kayma ihmal edilir)
    out["cost"] = out["turnover"] * cost
    out["net"] = out["gross"] - out["cost"]
    out["exposure"] = held.sum(axis=1)
    out["n_pos"] = (held > 0).sum(axis=1)
    out["entries"] = ((held > 0) & (prev == 0)).sum(axis=1)
    return out


def sector_returns(index: pd.DatetimeIndex) -> tuple:
    """Akran grubu başına eşit ağırlıklı günlük getiri (evrendeki tüm hisseler, çekirdek dışı akranlar dahil;
    bankacılıkta tek çekirdek hisse olduğu için). Döner: (grup x tarih getirileri, hisse -> grup)."""
    from src.data.universe import load_universe
    u = load_universe()
    grp = u.set_index("ticker")["peer_group"]
    r = pd.DataFrame({t: get_stock_history(t)["Close"] for t in grp.index}).pct_change().reindex(index)
    return r.T.groupby(grp).mean().T, grp


def sector_neutral(weights: pd.DataFrame, returns: pd.DataFrame, shift: int = 1) -> pd.Series:
    """Sektör nötr kıyas (Adım 5): her gün stratejinin sektör ağırlıklarını aynen alır, ama her sektörün içinde
    o sektörün tüm hisselerini eşit ağırlıkla tutar. Strateji ile farkı = sektör İÇİ hisse seçimi. Maliyetsiz."""
    w = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    held = w.shift(shift).fillna(0.0)
    sec_ret, grp = sector_returns(returns.index)
    sec_w = held.T.groupby(grp.reindex(held.columns)).sum().T
    return (sec_w * sec_ret[sec_w.columns].fillna(0.0)).sum(axis=1)


def trade_log(weights: pd.DataFrame, returns: pd.DataFrame, cost: float = COST_ONE_WAY, shift: int = 1) -> pd.DataFrame:
    """İşlem günlüğü (Adım 5): bir hissenin kesintisiz taşındığı her dönem bir işlem. Getiri = taşınan günlerin
    birleşik getirisi, giriş ve çıkışta tek yön maliyet; işlem içi ağırlık değişimleri (makro yarılama) yok sayılır.
    Veri sonunda hâlâ açık olan pozisyon 'acik' işaretlenir."""
    w = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    held = w.shift(shift).fillna(0.0) > 0
    xu = get_index_history("XU100")["Close"].pct_change().reindex(returns.index).fillna(0.0)
    rows = []
    for t in held.columns:
        h = held[t].to_numpy()
        if not h.any():
            continue
        start = np.flatnonzero(h & ~np.r_[False, h[:-1]])
        end = np.flatnonzero(h & ~np.r_[h[1:], False])
        r = returns[t].fillna(0.0).to_numpy()
        for a, b in zip(start, end):
            gross = float(np.prod(1 + r[a:b + 1]) - 1)
            rows.append(dict(ticker=t, entry=returns.index[a], exit=returns.index[b], days=int(b - a + 1),
                             gross=gross, net=(1 + gross) * (1 - cost) ** 2 - 1,
                             xu100=float(np.prod(1 + xu.iloc[a:b + 1]) - 1), open=bool(b == len(h) - 1)))
    return pd.DataFrame(rows, columns=["ticker", "entry", "exit", "days", "gross", "net", "xu100", "open"])


def trade_stats(trades: pd.DataFrame) -> dict:
    """Kapanmış işlemlerden: sayı, kazanma oranı, ortalama/ortanca net getiri, XU100'ü geçme oranı, ortalama süre."""
    c = trades[~trades["open"]]
    if c.empty:
        return dict(islem_kapanan=0)
    return dict(islem_kapanan=len(c), islem_kazanma_orani=float((c["net"] > 0).mean()),
                islem_ort_getiri=float(c["net"].mean()), islem_ortanca_getiri=float(c["net"].median()),
                islem_xu100_gecme_orani=float((c["net"] > c["xu100"]).mean()), islem_ort_gun=float(c["days"].mean()))


def max_drawdown(ret: pd.Series) -> float:
    eq = (1 + ret.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calculate_performance(bt: pd.DataFrame, bench: pd.Series = None, sector: pd.Series = None,
                          trades: pd.DataFrame = None) -> dict:
    """Net getiriden özet ölçüler. bench verilirse aynı günlerdeki kıyas getirisine göre fark; sector verilirse
    sektör nötr kıyasa göre fark; trades verilirse işlem bazında istatistikler (Adım 5)."""
    net = bt["net"].fillna(0)
    years = len(net) / DAYS
    total = float((1 + net).prod() - 1)
    vol = float(net.std(ddof=1) * np.sqrt(DAYS))
    active = net[bt["exposure"] > 0]
    res = dict(
        toplam_getiri=total,
        yillik_getiri=(1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan,
        yillik_oynaklik=vol,
        sharpe=float(net.mean() / net.std(ddof=1) * np.sqrt(DAYS)) if net.std(ddof=1) > 0 else np.nan,
        max_dusus=max_drawdown(net),
        kazanan_gun=float((active > 0).mean()) if len(active) else np.nan,
        piyasada_gun=float((bt["exposure"] > 0).mean()),
        islem_sayisi=int(bt["entries"].sum()),
        toplam_maliyet=float(bt["cost"].sum()),
    )
    if bench is not None:
        b = bench.reindex(net.index).fillna(0)
        diff = net - b
        res["kiyas_getiri"] = float((1 + b).prod() - 1)
        res["fark"] = total - res["kiyas_getiri"]
        res["bilgi_orani"] = float(diff.mean() / diff.std(ddof=1) * np.sqrt(DAYS)) if diff.std(ddof=1) > 0 else np.nan
    if sector is not None:
        res["sektor_kiyas_getiri"] = float((1 + sector.reindex(net.index).fillna(0)).prod() - 1)
        res["sektor_farki"] = total - res["sektor_kiyas_getiri"]
    if trades is not None:
        res.update(trade_stats(trades))
    res["uyari"] = "az sinyal" if res["islem_sayisi"] < MIN_TRADES else ""
    return res


def benchmarks(returns: pd.DataFrame) -> pd.DataFrame:
    """Kıyaslar: XU100 ve eşit ağırlıklı al-tut (30 hisse, maliyetsiz)."""
    xu = get_index_history("XU100")["Close"].pct_change().reindex(returns.index)
    return pd.DataFrame({"xu100": xu, "esit_agirlik": returns.mean(axis=1)})


def split_halves(bt: pd.DataFrame):
    """İlk yarı (örneklem içi) ve ikinci yarı (örneklem dışı)."""
    h = len(bt) // 2
    return bt.iloc[:h], bt.iloc[h:]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    ret = load_returns()
    bm = benchmarks(ret)
    equal = pd.DataFrame(1 / ret.shape[1], index=ret.index, columns=ret.columns)
    rows = {}
    rows["eşit ağırlık al-tut (maliyetli)"] = calculate_performance(run_backtest(equal, ret), bm["xu100"])
    cheat = (ret > 0).astype(float)
    cheat = cheat.div(cheat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    rows["KASITLI HATA: bugünün getirisini bilen sinyal, kaydırma yok"] = calculate_performance(run_backtest(cheat, ret, shift=0), bm["xu100"])
    rows["aynı sinyal, doğru kaydırma (1 gün)"] = calculate_performance(run_backtest(cheat, ret, shift=1), bm["xu100"])
    rows["aynı sinyal, kaydırma var, maliyet 0"] = calculate_performance(run_backtest(cheat, ret, cost=0.0), bm["xu100"])
    for name, r in rows.items():
        print(f"{name}\n    toplam %{r['toplam_getiri'] * 100:,.0f}   yıllık %{r['yillik_getiri'] * 100:,.0f}   sharpe {r['sharpe']:.2f}   "
              f"en büyük düşüş %{r['max_dusus'] * 100:.0f}   işlem {r['islem_sayisi']}   maliyet %{r['toplam_maliyet'] * 100:.0f}   "
              f"XU100'e göre fark %{r['fark'] * 100:,.0f}")
    print(f"\nKıyas: XU100 toplam %{((1 + bm['xu100'].fillna(0)).prod() - 1) * 100:,.0f}   "
          f"eşit ağırlık (maliyetsiz) %{((1 + bm['esit_agirlik'].fillna(0)).prod() - 1) * 100:,.0f}")