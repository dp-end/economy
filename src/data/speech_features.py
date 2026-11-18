"""Konuşma özellikleri (kurallarla, LLM yok): konu, sayısal iddia, iddia doğrulama, duruş, belirsizlik.

İlke: uzman/kurum yorumu doğrulanmış gerçek değildir. Sayısal iddialar deterministik veriyle
(makro katman, Adım 8) karşılaştırılır; yalnızca VERIFIED olanlar kanıt setine girebilir.
Diğer cümleler "yorum" ya da "doğrulanamayan iddia" olarak bağlam kalır.
"""
import re

import numpy as np
import pandas as pd

from src.data.macro import fed_table, load_series
from src.data.news_tags import tr_lower
from src.data.speech import SOURCES, load_speech_segments, quality

FEATURES_FILE = "reports/speech_features.parquet"
CLAIMS_FILE = "reports/speech_claims.csv"
TOLERANCE = 0.05 + 1e-9    # tek ondalığa yuvarlama payı (puan): 29,77 -> "yüzde 29,8" doğru, 29,73 değil
HISTORY_DAYS = 365         # tarihî değer kontrolü için geriye bakılan süre

TOPICS = {
    "faiz": [r"faiz", r"para politika", r"\bppk\b", r"\brates?\b", r"monetary policy", r"federal funds",
             r"policy stance"],
    "enflasyon": [r"enflasyon", r"fiyat", r"inflation", r"\bprices?\b"],
    "buyume": [r"büyüme", r"talep", r"gsyh", r"\bgrowth", r"\bgdp\b", r"economic activity", r"spending",
               r"investment"],
    "istihdam": [r"istihdam", r"işsizlik", r"\bjobs?\b", r"labor market", r"unemployment", r"payroll",
                 r"employment", r"jobless"],
    "kur": [r"\bkur\b", r"döviz", r"dolar", r"\bdollar", r"exchange rate"],
    "enerji_emtia": [r"petrol", r"enerji", r"doğal gaz", r"emtia", r"energy", r"\boil\b", r"commodit"],
    "jeopolitik": [r"jeopolitik", r"savaş", r"çatışma", r"geopolitic", r"\bwar\b", r"conflict", r"tariff"],
    "kredi_finansal": [r"kredi", r"finansal koşul", r"credit", r"financial conditions", r"lending"],
    "beklenti": [r"beklenti", r"expectations"],
}
HAWKISH = [r"sıkı para politika", r"sıkı duruş", r"sıkılaş", r"ihtiyatlı", r"yukarı yönlü risk", r"faiz artır",
           r"restrictive", r"tighten", r"\bhikes?\b", r"rais\w* rates", r"elevated", r"too high", r"upside risk",
           r"readiness to act", r"price stability", r"remov\w* .{0,20}accommodation"]
DOVISH = [r"faiz indir", r"indirim", r"gevşe", r"aşağı yönlü risk", r"\bcuts?\b", r"\beas(e|ing)\b",
          r"lower rates", r"accommodative", r"downside risk", r"cooling"]
UNCERTAIN = [r"belirsiz", r"\brisk", r"olabilir", r"muhtemel", r"ihtiyat", r"uncertain", r"\bmay\b", r"\bmight\b",
             r"\bcould\b", r"unclear", r"volatil"]
FORECAST_WORDS = [r"tahmin", r"öngör", r"hedef", r"beklenti", r"bekliyor", r"\bgoal", r"target", r"forecast",
                  r"project", r"expect", r"objective"]
# Doğrulanabilir değişkenler: cümlede bu desen + yüzde değeri varsa makro veriyle karşılaştırılır.
VARIABLES = {
    "tcmb_politika_faizi": [r"politika faiz"],
    "tufe_yillik": [r"yıllık tüketici enflasyonu"],
    "fed_faiz_araligi": [r"federal funds", r"fed funds"],
}

NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
NUM_WORDS.update({w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())})
FRACTIONS = {("a", "half"): 0.5, ("one", "half"): 0.5, ("a", "quarter"): 0.25, ("one", "quarter"): 0.25,
             ("three", "quarters"): 0.75}


def low(text: str) -> str:
    return tr_lower(text)


def has_any(text: str, patterns: list) -> bool:
    t = low(text)
    return any(re.search(p, t) for p in patterns)


def count_any(text: str, patterns: list) -> int:
    t = low(text)
    return sum(len(re.findall(p, t)) for p in patterns)


# ---------- sayıları okumak ----------
def _parse_en_number(tok: list, i: int):
    """tok[i]'den başlayan sayı: '2.4', 'two', 'twenty five', 'three and three quarters', 'two point five'."""
    t = tok[i]
    if re.fullmatch(r"\d+(\.\d+)?", t):
        v, j = float(t), i + 1
    elif t in NUM_WORDS:
        v, j = float(NUM_WORDS[t]), i + 1
        if v >= 20 and j < len(tok) and tok[j] in NUM_WORDS and NUM_WORDS[tok[j]] < 10:
            v, j = v + NUM_WORDS[tok[j]], j + 1
    else:
        return None
    if j + 2 < len(tok) + 1 and j < len(tok) and tok[j] == "and" and tuple(tok[j + 1:j + 3]) in FRACTIONS:
        v, j = v + FRACTIONS[tuple(tok[j + 1:j + 3])], j + 3
    elif j < len(tok) and tok[j] == "point":
        digits, k = "", j + 1
        while k < len(tok) and (tok[k] in NUM_WORDS and NUM_WORDS[tok[k]] < 10 or tok[k].isdigit()):
            digits += str(NUM_WORDS.get(tok[k], tok[k]))
            k += 1
        if digits:
            v, j = float(f"{int(v)}.{digits}"), k
    return v, j


def percents_en(text: str) -> list:
    """İngilizce metindeki yüzde değerleri. 'X to Y percent' aralığında iki değer de alınır."""
    t = text.lower().replace("¾", " and three quarters").replace("½", " and a half").replace("¼", " and a quarter")
    t = t.replace("-", " ").replace("%", " percent ")
    tok = re.findall(r"\d+(?:\.\d+)?|[a-z]+", t)
    out, i, pending = [], 0, []
    while i < len(tok):
        p = _parse_en_number(tok, i)
        if p is None:
            pending, i = ([] if tok[i] != "to" else pending), i + 1
            continue
        v, j = p
        if j < len(tok) and tok[j] == "percent":
            out += pending + [v]
            pending = []
        elif j < len(tok) and tok[j] == "to":
            pending = [v]
        else:
            pending = []
        i = j
    return out


def percents_tr(text: str) -> list:
    """Türkçe metindeki yüzde değerleri: 'yüzde 29,8', '%36'."""
    return [float(x.replace(",", ".")) for x in re.findall(r"(?:yüzde|%)\s*(\d+(?:,\d+)?)", low(text))]


def percents(text: str, language: str) -> list:
    return percents_tr(text) if language == "tr" else percents_en(text)


# ---------- doğrulama ----------
def tool_values(variable: str, asof) -> tuple:
    """(analiz anında bilinen değer, son HISTORY_DAYS günde görülen değerler). Yalnızca yayınlanmış veri."""
    asof = pd.Timestamp(asof)
    if variable == "fed_faiz_araligi":
        df = fed_table().rename(columns={"available_date": "avail"})
        df["value"] = list(zip(df["fed_lower"], df["fed_upper"]))
    elif variable == "tcmb_politika_faizi":
        df = load_series("policy_rate").rename(columns={"available_date": "avail"})
    else:                                                   # tufe_yillik
        df = load_series("cpi").sort_values("data_date")
        df["value"] = df["value"].pct_change(12) * 100
        df = df.rename(columns={"available_date": "avail"}).dropna(subset=["value"])
    known = df[df["avail"] <= asof].sort_values("avail")
    if known.empty:
        return None, []
    hist = known[known["avail"] >= asof - pd.Timedelta(days=HISTORY_DAYS)]["value"].tolist()
    return known["value"].iloc[-1], hist


def _close(a, b) -> bool:
    if isinstance(b, tuple):
        return isinstance(a, tuple) and all(abs(x - y) <= TOLERANCE for x, y in zip(a, b))
    return abs(a - b) <= TOLERANCE


def verify(text: str, values: list, asof) -> dict:
    """Bir cümledeki sayısal iddianın durumu.
    VERIFIED: analiz anındaki değere eşit. CONSISTENT_WITH_HISTORY: son bir yıldaki bir değere eşit (geçmişten söz ediyor).
    CONTRADICTED: veriyle uyuşmuyor. FORECAST_OR_TARGET: tahmin/hedef, doğrulanamaz. NO_TOOL: bu sayı için veri aracı yok."""
    var = next((v for v, pats in VARIABLES.items() if has_any(text, pats)), None)
    if var is None:
        status = "FORECAST_OR_TARGET" if has_any(text, FORECAST_WORDS) else "NO_TOOL"
        return dict(variable=None, status=status, tool_value=None)
    now, hist = tool_values(var, asof)
    cands = [tuple(values[i:i + 2]) for i in range(len(values) - 1)] if var == "fed_faiz_araligi" else values
    if now is None:
        status = "NO_TOOL"
    elif any(_close(c, now) for c in cands):
        status = "VERIFIED"
    elif any(_close(c, h) for c in cands for h in hist):
        status = "CONSISTENT_WITH_HISTORY"
    elif has_any(text, FORECAST_WORDS):
        status = "FORECAST_OR_TARGET"
    else:
        status = "CONTRADICTED"
    return dict(variable=var, status=status, tool_value=str(now))


# ---------- tablo ----------
def segment_features(seg: pd.DataFrame) -> pd.DataFrame:
    """Her cümle: konular, şahin/güvercin sayısı, belirsizlik, sayısal değerler ve segment türü."""
    out = seg.copy()
    out["topics"] = out["text"].map(lambda t: "|".join(k for k, p in TOPICS.items() if has_any(t, p)))
    out["hawkish"] = out["text"].map(lambda t: count_any(t, HAWKISH))
    out["dovish"] = out["text"].map(lambda t: count_any(t, DOVISH))
    out["uncertain"] = out["text"].map(lambda t: has_any(t, UNCERTAIN))
    out["values"] = [percents(t, lang) for t, lang in zip(out["text"], out["language"])]
    checks = [verify(t, v, p) if v else dict(variable=None, status=None, tool_value=None)
              for t, v, p in zip(out["text"], out["values"], out["publish_time"])]
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(checks)], axis=1)
    out["segment_type"] = np.select(
        [out["status"] == "VERIFIED", out["status"].notna()], ["dogrulanmis_olgu", "sayisal_iddia"], "yorum")
    out["in_evidence"] = out["status"] == "VERIFIED"
    return out


def stance(hawk: int, dove: int) -> tuple:
    """(-1..1) puan ve etiket. Pozitif = şahin (sıkı para politikası yönlü)."""
    if hawk + dove == 0:
        return 0.0, "notr"
    s = (hawk - dove) / (hawk + dove)
    return round(s, 2), "sahin" if s > 0.2 else "guvercin" if s < -0.2 else "notr"


def get_speech_features(feats: pd.DataFrame, source_id: str) -> dict:
    """MCP aracı get_speech_features'ın çıktısı: kaynak bilgisi, konu, duruş, belirsizlik, iddialar, uyarılar."""
    f = feats[feats["source_id"] == source_id]
    if f.empty:
        raise ValueError(f"bilinmeyen konuşma kaynağı: {source_id} (geçerli: {', '.join(SOURCES)})")
    src = SOURCES[source_id]
    score, label = stance(int(f["hawkish"].sum()), int(f["dovish"].sum()))
    q = quality(source_id, f)
    warnings = [w for w in [q["warning"], src["note"]] if w]
    if f["event_day"].isna().all():
        warnings.append("yayın, fiyat verisinin son gününden sonra: yalnızca güncel bağlam, geçmiş testlere girmez")
    topics = f["topics"].str.split("|").explode()
    claims = f[f["status"].notna()]
    return dict(
        source_id=source_id, speaker=src["speaker"], channel=src["channel"], title=src["title"],
        publish_time=str(src["publish_time"]), media_url=src["media_url"], official_text_url=src["official_text_url"],
        method=q["method"], match_rate=q["match_rate"], n_segments=int(len(f)),
        topics=topics[topics != ""].value_counts().to_dict(),
        stance_score=score, stance=label, uncertainty_share=round(float(f["uncertain"].mean()), 3),
        claims=claims[["segment_no", "start", "end", "text", "values", "variable", "status", "tool_value",
                       "in_evidence"]].to_dict("records"),
        claim_status=claims["status"].value_counts().to_dict(),
        warnings=warnings,
    )


def load_speech_features(path=FEATURES_FILE) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)
    feats = segment_features(load_speech_segments())
    feats.to_parquet(FEATURES_FILE, index=False)
    claims = feats[feats["status"].notna()]
    claims[["source_id", "segment_no", "start", "end", "values", "variable", "status", "tool_value", "text"]] \
        .to_csv(CLAIMS_FILE, index=False)
    for sid in SOURCES:
        r = get_speech_features(feats, sid)
        print(f"\n===== {sid} | {r['speaker']} | {r['publish_time']}")
        print(f"yöntem: {r['method']}  örtüşme: {r['match_rate']}  cümle: {r['n_segments']}")
        print(f"duruş: {r['stance']} ({r['stance_score']})  belirsizlik oranı: {r['uncertainty_share']}")
        print("konular:", r["topics"])
        print("iddia durumu:", r["claim_status"])
        print("uyarılar:", r["warnings"])
        v = claims[(claims["source_id"] == sid) & claims["variable"].notna()]
        print("veriyle karşılaştırılan iddialar:")
        print(v[["values", "variable", "status", "tool_value", "text"]].to_string(index=False))
