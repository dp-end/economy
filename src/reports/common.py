"""Araştırma raporları için ortak yardımcılar (Adım 15).

Raporlar reports/ altındaki hazır (commit'lenmiş) çıktıları okur, analizleri yeniden çalıştırmaz.
Sayılar ve tablolar koddan gelir; yorum paragrafları reports/research/notes/<rapor>.md dosyasından okunur,
böylece script yeniden çalışınca elle yazılmış yorumlar silinmez.
"""
import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

OUT = Path("reports/research")
FIG = OUT / "fig"
NOTES = OUT / "notes"
DISCLAIMER = ("Educational research output for CSE-481. Historical evidence only: not a forecast, "
              "not investment advice. Returns are nominal TRY.")

# Varsayılan doğrulanmış palet (dataviz referansı), açık zemin
C = dict(s1="#2a78d6", s2="#eb6834", pos="#2a78d6", neg="#e34948", mid="#f0efec", text="#0b0b0b",
         text2="#52514e", muted="#8a8984", grid="#e4e3df", surface="#fcfcfb")


# ---------- dosya damgası ----------
def stamp(paths) -> str:
    """Kaynak dosyaların tarih ve kısa SHA-256 değeri: rapor hangi çıktıdan üretildi?"""
    rows = []
    for p in map(Path, paths):
        if p.exists():
            ts = pd.Timestamp(p.stat().st_mtime, unit="s").floor("s")
            rows.append(dict(file=f"`{p}`", modified=str(ts), sha256=hashlib.sha256(p.read_bytes()).hexdigest()[:12]))
        else:
            rows.append(dict(file=f"`{p}`", modified="missing", sha256="-"))
    return md_table(pd.DataFrame(rows), index=False)


# ---------- tablo ----------
def _cell(v, fmt: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return fmt.format(v)
    return str(v)


def md_table(df: pd.DataFrame, fmt: dict = None, index: bool = True, default: str = "{:.2f}") -> str:
    """DataFrame -> GitHub Markdown tablosu. fmt: {sütun: '{:.1f}'}; NaN '–' olur."""
    fmt = fmt or {}
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    num = [pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---:" if n else ":---" for n in num) + "|"]
    for i in range(len(d)):             # iterrows kullanılmaz: karışık satırda tam sayıları ondalığa çevirir
        lines.append("| " + " | ".join(_cell(d[c].iat[i], fmt.get(c, default)) for c in d.columns) + " |")
    return "\n".join(lines)


# ---------- yorum notları ----------
def load_notes(name: str) -> dict:
    """notes/<name>.md içindeki '<!-- anahtar -->' işaretleriyle ayrılmış bölümler."""
    p = NOTES / name
    if not p.exists():
        return {}
    parts = re.split(r"^<!--\s*(\w+)\s*-->\s*$", p.read_text(encoding="utf-8"), flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def note(notes: dict, key: str) -> str:
    return notes.get(key) or f"> _(interpretation note `{key}` not written yet)_"


# ---------- istatistik ----------
def t_test(x) -> tuple:
    """Tek örneklem t testi (ortalama = 0). (t, p); 3'ten az gözlemde NaN."""
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan, np.nan
    r = stats.ttest_1samp(x, 0.0)
    return float(r.statistic), float(r.pvalue)


def clustered_t(values: pd.Series, dates: pd.Series) -> tuple:
    """Aynı gün birçok hissede olay olabilir; gözlemler bağımsız değildir. Önce gün ortalaması, sonra t testi."""
    by_day = pd.Series(values.to_numpy(), index=pd.to_datetime(dates.to_numpy())).groupby(level=0).mean()
    t, p = t_test(by_day)
    return t, p, len(by_day)


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q değerleri (NaN korunur)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


# ---------- grafik ----------
def style(ax, grid_axis: str = "x"):
    ax.set_facecolor(C["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["text2"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def save_fig(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(C["surface"])
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"fig/{name}"


def write(name: str, text: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(text.rstrip() + "\n", encoding="utf-8")
    return p
