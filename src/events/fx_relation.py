"""Kur ile hisse getirisi arasındaki geçmiş ilişki (Yerel model 1.6, örnek soru S5: "dolar yükseliyor, X kesin yükselir").

İki ölçü (kullanıcı kararı 2a, 24 Kasım 2026):
  1) Günlük beta: hissenin XU100'e göre fazla getirisi (%) ~ USD/TRY günlük değişimi (%), Newey-West (HAC, 5 gecikme)
     standart hata. R5'in sektör betasıyla (src/reports/r5_macro_context.fx_betas) aynı yöntem, hisse düzeyinde.
  2) Aylık karşılaştırma: USD/TRY'nin dönemin medyanından hızlı arttığı aylar ile diğer aylarda hissenin ortalama
     getirisi ve XU100'e göre farkı; aylık korelasyon. Ay sonu kapanışları; ilk ay dönemin ilk gününe göre, son ay
     kısmi olabilir.
Yalnızca verilen serilerin [start, end] aralığı kullanılır (çağıran 'end' sonrasını vermez).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

HAC_LAGS = 5


def monthly_changes(s: pd.Series) -> pd.Series:
    """Ay sonu değerlerinin bir önceki aya göre değişimi; ilk ay serinin ilk gözlemine göre."""
    m = s.dropna().resample("ME").last()
    return m / m.shift(1).fillna(s.dropna().iloc[0]) - 1


def fx_relation(close: pd.Series, xu: pd.Series, usdtry: pd.Series) -> dict:
    idx = xu.dropna().index
    close, usdtry = close.reindex(idx), usdtry.reindex(idx)
    d = pd.DataFrame({"r": close.pct_change() * 100, "x": xu.reindex(idx).pct_change() * 100,
                      "du": usdtry.pct_change() * 100}).dropna()
    d["ex"] = d["r"] - d["x"]
    daily = {}
    for name, y in (("excess_vs_xu100", "ex"), ("raw", "r")):
        res = sm.OLS(d[y], sm.add_constant(d["du"])).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        daily[name] = dict(beta=float(res.params.iloc[1]), t=float(res.tvalues.iloc[1]), p=float(res.pvalues.iloc[1]))
    daily["n_days"] = int(len(d))

    m = pd.DataFrame({"r": monthly_changes(close), "x": monthly_changes(xu.reindex(idx)),
                      "du": monthly_changes(usdtry)}).dropna() * 100
    fast = m["du"] > m["du"].median()
    grp = lambda g: dict(n_months=int(len(g)), mean_ret_pct=float(g["r"].mean()),
                         mean_excess_vs_xu100_pct=float((g["r"] - g["x"]).mean()),
                         mean_usdtry_chg_pct=float(g["du"].mean()))
    monthly = dict(n_months=int(len(m)), usdtry_months_up=int((m["du"] > 0).sum()),
                   usdtry_months_down=int((m["du"] < 0).sum()), median_usdtry_chg_pct=float(m["du"].median()),
                   fast_fx_months=grp(m[fast]), other_months=grp(m[~fast]),
                   corr_ret_vs_usdtry=float(m["r"].corr(m["du"])),
                   corr_excess_vs_usdtry=float((m["r"] - m["x"]).corr(m["du"])))
    return dict(daily=daily, monthly=monthly)
