"""v4 ince ayar veri seti (27 Kasım 2026, kullanıcı kararları: hata-düzelt, S2 cümleleri, S4/S5 kuralı).

v3 değerlendirmesinde (reports/local_llm/eval_v3) v3'ün 28 yanlışının 20'si ANALYSIS_UNSAFE idi: v3 verisindeki araç
hatalı 24 konuşmanın hepsi ANALYSIS_UNSAFE ile bitiyordu, hatayı düzeltip yeniden deneyen örnek yoktu (istem de 'araç
hatası -> ANALYSIS_UNSAFE' diyordu). v4 verisi v3'ün üstüne dört yerde farklıdır:
  1. Hata-düzelt (60 konuşma, kategorilere dağılır): ilk uygun çağrı bilerek bozulur (yanlış strateji/olay/koşul/
     gösterge adı, bozuk desen, aralık dışı oos_start, ters tarih, .IS ekli kod, çeyrek adı 'Q2'), gerçek araç hatası
     döner, sonraki tur düzeltilmiş çağrıdır, cevap normaldir. Bozuk çağrı kayıpta yoktur (masks.py, 'train': False).
     Hatalı istek konuşmaları (bilinmeyen hisse/endeks/sektör, gelecek tarih) yine ANALYSIS_UNSAFE ile biter.
  2. S2 desenleri: desen kodundan otomatik cümle (gün adları, 'haftanın ilk/son iki günü', 'ertesi gün'; üç sözcük
     ailesi, üç cümle yapısı), 8 yerine bütün geçerli tek-yönlü koşul/sonuç desenleri; örneklem dışı dönemi önce veren
     ve 'üst üste N seans' kalıpları. S2 80 -> 100 konuşma.
  3. S5: kısa dönem (60 işlem gününden az; araç kur-hisse ilişkisini hesaplamaz) açıkça sorulursa INVESTIGATE
     (templates.ans_s5_fx); dönemsiz kalıplar artar (araç çağrısı veri döneminin başından, 2024-01-01).
  4. İstem (question_mode.SYSTEM): düzeltilebilir hata -> yeniden çağır; dönem yoksa 2024-01-01; bilanço sorusunda
     yalnızca üç eylem; ilişki hesaplanamadıysa iddia desteklenmiyor.
heldout_v3 kalıpları eğitime girmez (test eder); v4 yeni bir görülmemiş setle ölçülür (src/finetune/heldout_v4.py).

Çıktı: data/finetune_v4/ (git dışında). Çalıştırma: python -m src.finetune.dataset_v4
"""
import argparse
import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

from src.finetune import build_dataset as B
from src.finetune import heldout as H
from src.finetune import templates as TP
from src.finetune.dataset_v3 import COUNTS_V3, SamplerV3

OUT_V4 = Path("data/finetune_v4")
SEED_V4 = 20261127
COUNTS_V4 = dict(COUNTS_V3, s2=100, fix=60)
CARD_V4 = Path("reports/local_llm/dataset_card_v4.md")

# ---------------------------------------------------------------- 2. S2 desen cümleleri
DAYS = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma")
ABL = ("Pazartesi'den", "Salı'dan", "Çarşamba'dan", "Perşembe'den", "Cuma'dan")
DAT = ("Pazartesi'ye", "Salı'ya", "Çarşamba'ya", "Perşembe'ye", "Cuma'ya")
NOUN = ({"D": "düşüş", "U": "yükseliş"}, {"D": "kayıp", "U": "kazanç"}, {"D": "eksi kapanış", "U": "artı kapanış"})
PART = ({"D": "düştüğü", "U": "yükseldiği"}, {"D": "değer kaybettiği", "U": "değer kazandığı"},
        {"D": "eksi kapandığı", "U": "artı kapandığı"})
CLASSIC = tuple(TP.WEEK_PATTERNS)          # öğretmenin ve v1-v3'ün 8 deseni; daha sık seçilir
CLASSIC_WEIGHT = 8


def week_runs() -> list:
    """(a, b, X, c, d, Y): koşul günleri a..b yönü X, sonuç günleri c..d yönü Y (b < c); koşul ve sonuç 1-3 gün."""
    out = []
    for a in range(5):
        for b in range(a, min(a + 3, 5)):
            for c in range(b + 1, 5):
                for d in range(c, min(c + 3, 5)):
                    out += [(a, b, x, c, d, y) for x in "DU" for y in "DU"]
    return out


def week_code(a, b, x, c, d, y) -> str:
    """'week:' sonrası kod: koşul dizisi c karakter (Pzt..sonuçtan önceki gün), tire, sonuç dizisi 5 - c karakter."""
    return "." * a + x * (b - a + 1) + "." * (c - b - 1) + "-" + y * (d - c + 1) + "." * (4 - d)


def run_names(a: int, b: int, after: int = None) -> list:
    """Gün aralığının ifadeleri. after: koşulun son günü (sonuç hemen ertesi günse 'ertesi gün' de olur)."""
    n = b - a + 1
    if n == 1:
        out = [DAYS[a], f"{DAYS[a]} günü"]
        if after is not None and a == after + 1:
            out.append("ertesi gün")
    elif n == 2:
        out = [f"{DAYS[a]}–{DAYS[b]}", f"{DAYS[a]} ve {DAYS[b]}"]
        out += ["haftanın ilk iki günü"] if a == 0 else []
        out += ["haftanın son iki günü"] if b == 4 else []
        out += ["sonraki iki gün"] if after is not None and a == after + 1 else []
    else:
        out = [f"{DAYS[a]}–{DAYS[a + 1]}–{DAYS[b]}", f"{ABL[a]} {DAT[b]} üç gün"]
        out += ["haftanın ilk üç günü"] if a == 0 else []
        out += ["haftanın son üç günü"] if b == 4 else []
    return out


def week_descriptions(a, b, x, c, d, y) -> list:
    """Bir desenin bütün eğitim cümleleri (sabit sırada; örnekleyici aralarından seçer)."""
    out = []
    for cn in run_names(a, b):
        for on in run_names(c, d, after=b):
            for f in range(3):
                out += [f"{cn} {NOUN[f][x]}, ardından {on} {NOUN[f][y]}",
                        f"{cn} {NOUN[f][x]}, sonra {on} {NOUN[f][y]}",
                        f"{cn} {PART[f][x]} haftalarda {on} {NOUN[f][y]}"]
    return out


WEEK_CODES = {}
for _r in week_runs():
    WEEK_CODES.setdefault(week_code(*_r), _r)
assert all(c in WEEK_CODES for c in CLASSIC), "klasik desenler üreticide yok"


class WeekDesc:
    """Sampler.WEEK yerine: anahtarlar bütün desen kodları (klasik 8 desen CLASSIC_WEIGHT kez), değer her okumada
    rastgele bir cümle (otomatik cümleler + v1-v3'te kullanılan sabit ifadeler)."""

    def __init__(self, rng):
        self.rng = rng
        self.keys = list(WEEK_CODES) + [c for c in CLASSIC for _ in range(CLASSIC_WEIGHT - 1)]

    def __iter__(self):
        return iter(self.keys)

    def __getitem__(self, code):
        fixed = [m[code] for m in (TP.WEEK_PATTERNS, H.WEEK_EVAL) if code in m]
        return self.rng.choice(week_descriptions(*WEEK_CODES[code]) + fixed)


# ---------------------------------------------------------------- 1. hata-düzelt: bozuk çağrılar
COND_TYPO = {"rsi_dip": "rsi_dibi", "rsi_tepe": "rsi_tepesi", "rsi_notr": "rsi_normal", "hacim_yuksek": "hacim_yüksek",
             "hacim_normal": "hacim_orta", "st_asagi": "supertrend_asagi", "st_yukari": "supertrend_yukari",
             "bulut_alti": "bulut_altinda", "bulut_ustu": "bulut_ustunde", "bulut_ici": "bulut_icinde",
             "trend_yukari": "sma200_ustu", "trend_asagi": "sma200_alti", "kama_yukselen": "kama_yukari",
             "kama_dusen": "kama_asagi", "kama_yatay": "kama_duz"}
IND_TYPO = {"rsi_14": "rsi", "st_dir": "st_trend", "cloud_state": "ichimoku", "bb_lower": "bollinger_lower",
            "bb_upper": "bollinger_upper", "bb_mid": "bollinger_mid", "macd_signal": "macd_sinyal",
            "kama_slope": "kama_egim", "macd": "macd_line"}
PERIOD_BAD = ("Q2", "Q3", "son çeyrek", "son")


def _one(options: list, u: float):
    options = [o for o in options if o is not None]
    return options[int(u * len(options)) % len(options)] if options else None


def bad_args(kind: str, args: dict, u: float):
    """Bozuk argümanlar (kind: strategy, pattern, oos, event, cond, ind, period, swap, ticker, index) ya da None."""
    a = dict(args)
    if kind == "strategy":
        s = a["strategy"]
        a["strategy"] = _one(["catchup", "catch-up", "yakalama"] if s == "catch_up" else
                             [s.split("_")[0], s.split("_", 1)[1] if "_" in s else None], u)
    elif kind == "pattern":
        p = a["pattern"]
        if p.startswith("week:"):
            code = p[5:]
            opts = [code, "week:" + code.replace(".", "") if "." in code else None, "week:" + code.replace("-", "")]
        elif p == "weekday":
            opts = ["haftanin_gunu", "day_of_week"]
        else:
            name, k = p.split(":")
            opts = [f"{name}{k}", f"streak:{k}", f"{'yukselis' if name.startswith('up') else 'dusus'}_serisi:{k}"]
        a["pattern"] = _one(opts, u)
    elif kind == "oos":
        a["oos_start"] = a["start"]
    elif kind == "event":
        e, toks = a["event"], a["event"].split("_")
        a["event"] = _one(["_".join(reversed(toks)), e.replace("_", ""), e + "_signal"], u)
    elif kind == "cond":
        if not a.get("conditions"):
            return None
        c0 = a["conditions"][0].split("+")
        if c0[0] not in COND_TYPO:
            return None
        a["conditions"] = ["+".join([COND_TYPO[c0[0]]] + c0[1:])] + list(a["conditions"][1:])
    elif kind == "ind":
        inds = list(a["indicators"])
        j = next((i for i, x in enumerate(inds) if x in IND_TYPO), None)
        if j is None:
            return None
        inds[j] = IND_TYPO[inds[j]]
        a["indicators"] = inds
    elif kind == "period":
        a["period"] = _one(list(PERIOD_BAD), u)
    elif kind == "swap":
        if not (a.get("start") and a.get("end")) or a["start"] >= a["end"]:
            return None
        a["start"], a["end"] = a["end"], a["start"]
    elif kind == "ticker":
        if a.get("tickers"):
            t = a["tickers"][0]
            a["tickers"] = [_one([f"{t}.IS", f"BIST:{t}"], u)] + list(a["tickers"][1:])
        elif a.get("ticker"):
            a["ticker"] = _one([f"{a['ticker']}.IS", f"BIST:{a['ticker']}"], u)
        else:
            return None
    elif kind == "index":
        a["index"] = _one(["BIST100", "XU100.IS"], u)
    else:
        raise ValueError(kind)
    return None if a == args else a


def make_mistake(tool: str, kind: str, u: float):
    """FixConv için: yalnızca 'tool' adlı ilk çağrıyı bozar."""
    return lambda name, args: bad_args(kind, args, u) if name == tool else None


# kategori -> [(ağırlık, sampler argümanları, araç, bozma türü)]
FIX_PLAN = {
    "s1": [(3, dict(variant="hist"), "run_backtest", "strategy"), (3, dict(variant="full"), "run_backtest", "strategy"),
           (4, {}, "rank_sector_relative_strength", "ticker")],
    "s2": [(8, {}, "test_weekday_pattern", "pattern"), (4, {}, "test_weekday_pattern", "oos")],
    "s3": [(5, {}, "detect_reversal_events", "event"), (4, {}, "detect_reversal_events", "cond"),
           (3, {}, "calculate_indicators", "ind")],
    "s4": [(4, dict(bank=False), "get_fundamental_metrics", "period"),
           (4, dict(bank=False), "calculate_fundamental_change", "swap")],
    "s5": [(6, dict(kind="fx"), "get_macro_context", "ticker")],
    "other": [(3, dict(kind="o_bt"), "run_backtest", "strategy"), (3, dict(kind="o_ind"), "calculate_indicators", "ind"),
              (2, dict(kind="o_fund"), "get_fundamental_metrics", "period"),
              (1, dict(kind="o_index"), "get_index_history", "index"),
              (2, dict(kind="o_price"), "get_stock_history", "swap"),
              (1, dict(kind="o_news"), "get_news_events", "ticker")],
}
FIX_CAT_WEIGHTS = {"s1": 10, "s2": 12, "s3": 12, "s4": 8, "s5": 6, "other": 12}


# ---------------------------------------------------------------- yeni soru kalıpları
Q_V4_EXTRA = {
    "s2": [
        "{oos_start} – {asof} döneminde {T}'de '{desc}' düzeni sürdü mü? Karşılaştırma için örneklem içi dönem "
        "{is_start} – {is_end}.",
        "{T} için '{desc}' örüntüsünü test et: örneklem içi {is_start}–{is_end}, örneklem dışı {oos_start}–{asof}; "
        "maliyet sonrası sonuç ne?",
    ],
    "s2_weekday": [
        "{oos_start}–{asof} arasında {T}'de haftanın günleri arasındaki getiri farkı devam etti mi? Önceki dönem "
        "{is_start}–{is_end}.",
    ],
    "s2_streak": [
        "{T}'de üst üste {k} seans {yon_ad} geldiğinde sonraki seans ortalamada nasıl kapanıyor? Örneklem içi "
        "{is_start}–{is_end}, örneklem dışı {oos_start}–{asof}.",
        "{oos_start} – {asof} arasında {T} art arda {k} gün {yon_sonra} ertesi gün belirgin bir hareket var mı? "
        "{is_start} – {is_end} dönemiyle karşılaştır.",
    ],
    "s5_fx": [
        "Kur hızla tırmanıyor, {T} döviz geliri yüzünden mutlaka yükselir; {asof} itibarıyla katılıyor musun? Hedef "
        "fiyat da söyle.",
        "Dolar/TL arttıkça {T} hep kazandırıyor gibi; analiz günü {asof}, bu iddia veriyle tutuyor mu ve kaça çıkar?",
    ],
}
Q_V4_SHORT = [     # kısa dönem (60 işlem gününden az) açıkça verilen kur iddiası
    "Son haftalarda dolar yükseldi; {T} {start} – {asof} arasında bundan kazançlı çıktı mı, kesin yükselir mi? Hedef "
    "fiyat da ver.",
    "{start} ile {asof} arasındaki kur artışında {T} XU100'ü geçti mi? Dolar yükseldikçe kesin kazandırır diyorlar, "
    "doğru mu?",
]
SHORT_SHARE = 0.25          # kur sorularının bu kadarı kısa dönem
SHORT_DAYS = (21, 70)       # takvim günü; 70 gün < 60 işlem günü


class SamplerV4(SamplerV3):
    Q = {k: SamplerV3.Q[k] + Q_V4_EXTRA.get(k, []) for k in SamplerV3.Q}

    def __init__(self, rng):
        super().__init__(rng)
        self.WEEK = WeekDesc(rng)

    def s5(self, kind: str = None) -> dict:
        r = self.rng
        p_short = SHORT_SHARE if kind == "fx" else 0.6 * SHORT_SHARE if kind is None else 0.0   # kur sorusu %60
        if r.random() < p_short:
            asof = self.asof("2024-06-01")
            start = str(date.fromisoformat(asof) - timedelta(days=r.randint(*SHORT_DAYS)))
            T = self.pick(self.core)
            return dict(cat="s5", kind="fx", short=True, T=T, start=start, asof=asof,
                        question=self.pick(Q_V4_SHORT).format(T=T, start=self.dfmt(start), asof=self.dfmt(asof)))
        return super().s5(kind)

    def fix(self) -> dict:
        r = self.rng
        cat = r.choices(list(FIX_CAT_WEIGHTS), list(FIX_CAT_WEIGHTS.values()))[0]
        opts = FIX_PLAN[cat]
        _, kw, tool, kind = r.choices(opts, [o[0] for o in opts])[0]
        sp = getattr(self, cat)(**kw)
        return dict(sp, fix=f"{cat}:{tool}:{kind}", mistake=make_mistake(tool, kind, r.random()))


def build_v4(counts: dict = None, seed: int = SEED_V4, out: Path = OUT_V4) -> dict:
    return asyncio.run(B.build(counts or COUNTS_V4, seed, out, sampler=SamplerV4))


V4_NOTE = """**v4 changes** (on top of v3). (1) 60 *error-fix* conversations: the first suitable tool call is deliberately
broken (wrong strategy/event/condition/indicator name, malformed weekday pattern, out-of-range out-of-sample date,
swapped dates, ticker with a `.IS` suffix, a quarter name instead of a date), the real tool error comes back, the
next turn is the corrected call and the answer is normal. The broken call stays in the context but is excluded from
the loss, so the model learns the correction, not the mistake. Error questions (unknown ticker/index/sector, future
dates) still end with ANALYSIS_UNSAFE. In v3 all 24 conversations with a tool error ended with ANALYSIS_UNSAFE and
20 of v3's 28 wrong evaluation answers were ANALYSIS_UNSAFE. (2) Weekday patterns are described by generated
sentences (day names, "first/last two days of the week", "the next day"; three word families, three sentence shapes)
for every single-direction condition/outcome pattern, not only 8; new phrasings give the out-of-sample period first
or ask about "N sessions in a row". (3) Exchange-rate claims over a period shorter than 60 trading days (the tool
cannot estimate the relation) end with INVESTIGATE; more claims without a period (the call starts at 2024-01-01).
(4) The system prompt now says: fix and retry a correctable tool error; no period given -> start 2024-01-01;
fundamentals questions use only their three actions; an unestimable relation means the claim is not supported.
Phrasings of the second held-out set (`heldout_v3.py`) are not used in training; v4 is evaluated on a new set.

"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v4 ince ayar veri seti")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUT_V4))
    a = ap.parse_args()
    counts = {k: max(1, round(v * a.scale)) for k, v in COUNTS_V4.items()}
    stats = build_v4(counts, out=Path(a.out))
    if Path(a.out) == OUT_V4 and a.scale == 1.0:
        B.write_card(stats, counts, CARD_V4, "data/finetune_v4/", "python -m src.finetune.dataset_v4", V4_NOTE)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
