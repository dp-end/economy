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


def supported(x: float, d: int, allowed: list) -> bool:
    for a in allowed:
        for cand in (a, a * 100, a / 100, abs(a), abs(a) * 100):
            if round(cand, d) == round(x, d) or round(abs(cand), d) == round(abs(x), d):
                return True
    return False


def check_numbers(text: str, *context) -> dict:
    """{'n_numbers', 'unsupported': [yazım,...], 'rate'} - rate: desteksiz sayı oranı."""
    allowed = context_numbers(*context)
    nums = [(v, d, raw) for v, d, raw in numbers_in(text) if not (d == 0 and abs(v) < 10)]
    bad = [raw for v, d, raw in nums if not supported(v, d, allowed)]
    return dict(n_numbers=len(nums), unsupported=bad, rate=(len(bad) / len(nums)) if nums else 0.0)
