"""Sayısal halüsinasyon kontrolü: LLM metnindeki her sayı, modele verilen araç çıktısında/kanıtta var mı?

Kural: metindeki sayı, bağlamdaki bir sayının aynı ondalığa yuvarlanmış hâline (ya da %/oran çevrimine)
eşitse desteklidir. Tarihler (2026-10-27, 27.10.2026, '27 Ekim 2026'), yıllar, kanıt kimlikleri (E1, E2) ve 10'dan küçük
tam sayılar (sayım, gün) sayılmaz.
"""
import json
import re

NUM = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,]\d+)*(?![\w])")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4}")
EVID = re.compile(r"\bE\d+\b")
MONTHS = ("Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık|January|February|March|April|"
          "May|June|July|August|September|October|November|December")
# "27 Ekim", "7, 8, 23, 26 Ekim", "26 ve 28 Ekim", "Ekim 2026", "28 Ekim 2026'da"
WRITTEN_DATE = re.compile(rf"\b\d{{1,2}}(?:\s*,\s*\d{{1,2}})*(?:\s+(?:ve|and)\s+\d{{1,2}})?\s+(?={MONTHS})", re.I)
YEAR = re.compile(r"(?<![\d.,])(?:19[89]\d|20[0-3]\d)(?![\d.,]\d)")


def parse_number(tok: str):
    """'81,6' '81.6' '1.234,5' '3,415' -> float. Türkçe ve İngilizce yazımı birlikte ele alır."""
    t = tok.replace("−", "-")
    if "," in t and "." in t:                       # 1.234,5 (TR) ya da 1,234.5 (EN)
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".")
    elif t.count(".") > 1:
        t = t.replace(".", "")
    try:
        return float(t), (len(t.split(".")[1]) if "." in t else 0)
    except ValueError:
        return None, 0


def numbers_in(text: str) -> list:
    """(sayı, ondalık basamak, orijinal yazım) listesi. Tarih ve kanıt kimlikleri atılır."""
    clean = YEAR.sub(" ", WRITTEN_DATE.sub(" ", EVID.sub(" ", DATE.sub(" ", text or ""))))
    out = []
    for m in NUM.finditer(clean):
        v, d = parse_number(m.group())
        if v is not None:
            out.append((v, d, m.group()))
    return out


def context_numbers(*objs) -> list:
    """Bağlam nesnelerindeki (sözlük, liste, metin) tüm sayılar."""
    vals = []
    for o in objs:
        s = o if isinstance(o, str) else json.dumps(o, ensure_ascii=False, default=str)
        vals += [v for v, _, _ in numbers_in(s)]
        for m in re.finditer(r'"[^"]*":\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)', s):   # JSON'daki tam hassasiyetli sayılar
            vals.append(float(m.group(1)))
    return vals


SCALES = (1, 100, 1 / 100, 1e-3, 1e-6, 1e-9)   # oran<->yüzde, bin / milyon / milyar çevirmesi


def _candidates(allowed: list):
    for a in allowed:
        for k in SCALES:
            if k < 1e-2 and abs(a) < 1e3:            # ölçek çevirmesi yalnızca büyük sayılarda anlamlı
                continue
            yield abs(a * k)


def classify(x: float, d: int, allowed: list) -> str:
    """'destekli': bağlamdaki bir sayının doğru yuvarlanmışı. 'yuvarlama': son basamakta en fazla 1 fark
    (örn. 49,959 -> 49,95 kesme). 'uydurma': bağlamda karşılığı yok."""
    ax, unit = abs(x), 10 ** (-d)
    near = False
    for c in _candidates(allowed):
        r = round(c, d)
        if r == round(ax, d):
            return "destekli"
        if abs(c - ax) <= unit + 1e-12:
            near = True
    return "yuvarlama" if near else "uydurma"


def supported(x: float, d: int, allowed: list) -> bool:
    return classify(x, d, allowed) == "destekli"


def check_numbers(text: str, *context) -> dict:
    """{'n_numbers', 'unsupported': uydurma yazımlar, 'rounding': yuvarlama hataları, 'rate', 'rounding_rate'}"""
    allowed = context_numbers(*context)
    nums = [(v, d, raw) for v, d, raw in numbers_in(text) if not (d == 0 and abs(v) < 10)]
    cls = [(raw, classify(v, d, allowed)) for v, d, raw in nums]
    bad = [raw for raw, c in cls if c == "uydurma"]
    rnd = [raw for raw, c in cls if c == "yuvarlama"]
    n = len(nums)
    return dict(n_numbers=n, unsupported=bad, rounding=rnd, rate=len(bad) / n if n else 0.0,
                rounding_rate=len(rnd) / n if n else 0.0)
