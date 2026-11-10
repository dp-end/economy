
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.universe import load_universe

FT_DIR = Path("data/fintables")
FINANCIALS = {"AKBNK", "ISMEN", "TURSG"}   # gelir tablosu yapısı farklı: ciro / brüt kar anlamsız

SHEETS = {"income": "Gelir tablosu (çeyreklik)", "balance": "Bilanço"}
LABELS = {  # aday etiketler, öncelik sırasıyla (küçük harf, Türkçe düzeltilmiş)
    "income": {
        "net_income": ["ana ortaklık payları", "grubun karı (zararı)", "dönem net karı veya zararı"],
        "revenue": ["satış gelirleri"],
        "gross_profit": ["brüt kar (zarar)"],
        "op_income": ["faaliyet karı (zararı)"],
    },
    "balance": {
        "equity": ["ana ortaklığa ait özkaynaklar", "toplam özkaynaklar", "özkaynak toplamı", "özkaynaklar"],
        "total_assets": ["toplam varlıklar", "varlıklar toplamı"],
    },
}


def _norm(s) -> str:
    """Türkçe büyük harfleri düzgün küçült: İ->i, I->ı."""
    return str(s).replace("İ", "i").replace("I", "ı").lower().strip()


def _period_end(col) -> pd.Timestamp:
    """'2026/9' -> 2026-09-30"""
    y, m = str(col).split("/")
    return pd.Timestamp(int(y), int(m), 1) + pd.offsets.MonthEnd(0)


def _row(df: pd.DataFrame, candidates):
    labels = df.iloc[:, 0].map(_norm).to_numpy()
    for c in candidates:
        for i in np.where(labels == c)[0]:
            s = pd.to_numeric(df.iloc[i, 1:], errors="coerce")
            if s.notna().any():
                return s
    return None


def load_statements(ticker: str) -> pd.DataFrame:
    """Tek hisse: dönem sonu tarihi -> temel kalemler (çeyreklik, tek çeyrek değerleri)."""
    path = FT_DIR / f"{ticker} (TRY).xlsx"
    out = {}
    for kind, sheet in SHEETS.items():
        df = pd.read_excel(path, sheet_name=sheet)
        for name, cands in LABELS[kind].items():
            s = _row(df, cands)
            if s is not None:
                out[name] = s
    res = pd.DataFrame(out)
    res.index = [_period_end(c) for c in res.index]
    res.index.name = "period_end"
    res = res.sort_index()
    if ticker in FINANCIALS:
        res = res.drop(columns=[c for c in ("revenue", "gross_profit", "op_income") if c in res.columns])
    return res


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ni = d["net_income"]
    for name, off in (("yoy", pd.offsets.QuarterEnd(4)), ("qoq", pd.offsets.QuarterEnd(1))):
        prev = ni.copy()
        prev.index = prev.index + off          # eski değeri bir sonraki karşılaştırma tarihine taşı
        prev = prev.reindex(d.index)
        d[f"ni_{name}"] = ((ni - prev) / prev.abs()).replace([np.inf, -np.inf], np.nan)
    d["roe_q"] = ni / d["equity"]
    d["leverage"] = d["total_assets"] / d["equity"]
    if "revenue" in d.columns:
        d["net_margin"] = ni / d["revenue"]
    d["loss"] = ni < 0
    return d


@lru_cache(maxsize=1)
def load_all() -> dict:
    u = load_universe()
    core = u[u["is_core"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].tolist()
    return {t: add_ratios(load_statements(t)) for t in core}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    allf = load_all()
    rows = []
    for t, d in allf.items():
        rows.append(dict(
            ticker=t, n=len(d), ilk=d.index.min().date(), son=d.index.max().date(),
            net_income=d["net_income"].notna().sum(), equity=d["equity"].notna().sum(),
            assets=d["total_assets"].notna().sum(),
            revenue=int(d["revenue"].notna().sum()) if "revenue" in d.columns else "-"))
    print(pd.DataFrame(rows).to_string(index=False))
    for t in ("ASELS", "AKBNK", "TURSG"):
        s = allf[t].tail(4)[["net_income", "equity", "ni_yoy", "ni_qoq", "roe_q", "leverage"]].copy()
        s[["net_income", "equity"]] /= 1e9
        print(f"\n{t} (net_income ve equity milyar TL):")
        print(s.round(3).to_string())
    pd.concat(allf, names=["ticker"]).reset_index().to_parquet("reports/fundamentals.parquet")