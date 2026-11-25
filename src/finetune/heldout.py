"""Görülmemiş soru seti (yerel model 5. adım): taban model ile ince ayarlı modeli karşılaştırmak için.

Kullanıcı kararları (25 Kasım 2026): 1a soru ifadeleri eğitim şablonlarından farklı (bu dosyadaki kalıplar, terimler
ve ek bir tarih biçimi: '10 Eylül 2026'); hisse/tarih kombinasyonları eğitimde yok (aynı kategori + aynı hisse ya da
grup + aynı analiz tarihi olan soru atılır; eğitim ve doğrulama konuşmalarının hepsine bakılır). 2a 40 soru:
S1-S4 7'şer, S5 5, diğer araçlar 4, hatalı istek 3. Kategorinin içi de dengelenir (S1: sıralama 2, geçmiş 1, tam 2 +
kuralı sağlayan 2 aday; S3: 5 rastgele + 2 aday). Adaylar eğitimdeki hedefli örneklemeyle aynı yoldan bulunur;
olmasaydı POTENTIAL_CATCH_UP_CANDIDATE ve TECHNICAL_REVERSAL_CANDIDATE eylemleri testte hiç yer almazdı.

Altın cevap, eğitim verisini üreten kodla aynı yoldan çıkar (build_dataset.run_spec): altın araç çağrıları gerçek MCP
sunucusunda aynı tavanla çalışır, eylem aynı kurallarla seçilir. Bu yüzden ölçüm modelin kuralları yeni ifadelere
doğru uygulayıp uygulamadığıdır; kuralların ekonomik doğruluğu araştırma katmanının konusudur.

Çıktı: reports/local_llm/eval/heldout.jsonl (soru, analiz tarihi, altın çağrılar, altın eylem).
Çalıştırma: python -m src.finetune.heldout
"""
import argparse
import asyncio
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

from mcp import Client

from src.finetune import build_dataset as B
from src.finetune import templates as TP
from src.harness import question_mode as QM
from src.mcp_server import tools as TOOLS
from src.mcp_server.server import build_server

SEED_EVAL = 20261125
OUT = Path("reports/local_llm/eval")
HELDOUT = OUT / "heldout.jsonl"
TRAIN = B.OUT / "conversations.jsonl"
MAX_TRIES = 30
MONTHS_TR = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım",
             "Aralık")

# ---------------------------------------------------------------- yeni ifadeler (yuvalar build_dataset.Sampler ile aynı)
Q_EVAL = {
    "s1_rank": [
        "{asof} itibarıyla {tickers} hisseleri arasında son {h} işlem gününde akranlarının en çok gerisinde kalan hangisi?",
        "{grup} grubunda son {h} günün getirilerine göre bir sıralama yapar mısın? Grubun en zayıf hissesi hangisi?",
        "Son {h} seansta {grup} hisseleri içinde sektör ortancasına göre en kötü performansı hangi hisse gösterdi?",
    ],
    "s1_hist": [
        "{tickers} içinde son {h} işlem gününde en çok geride kalan hisse hangisi? {start} – {asof} arasında bu hisse "
        "benzer ölçüde geride kaldığında fark 5, 10 ve 20 gün içinde kapanmış mı, kaç sinyal yanlış çıkmış?",
        "{grup} sektöründe son {h} günde akran ortancasının en altında kalan hisseyi bul. {start} ile {asof} arasında bu "
        "kadar geride kaldığı durumların ardından akranlarına yetişebilmiş mi? 5/10/20 günlük oranları ver.",
        "Analiz günü {asof}. {grup} hisselerinden son {h} günde en zayıf kalanın {bas} bu yana yaşadığı benzer geride "
        "kalma dönemleri ne sıklıkla toparlandı?",
    ],
    "s1_full": [
        "{grup} grubunda ({tickers}) son {h} işlem gününün en zayıf hissesi hangisi? {start} – {asof} döneminde benzer "
        "geride kalmalar 5, 10 ve 20 günde kapanmış mı? Son açıklanan bilançosunu da hesaba katarak bu hisse catch-up "
        "adayı sayılabilir mi?",
        "Son {h} günde {tickers} arasında en geride kalan hisse için {bas} bu yana benzer geride kalmaların kapanma "
        "oranlarına ve son bilançosuna bakarak POTENTIAL_CATCH_UP_CANDIDATE etiketi uygun mu?",
        "{asof} tarihinde {grup} sektöründe akranlarından en çok geride kalan hissenin zayıflığı kalıcı mı geçici mi? "
        "{start} tarihinden bu yana benzer geride kalmaları ve temel verilerini birlikte değerlendir.",
    ],
    "s2": [
        "{T} hissesinde '{desc}' şeklinde bir haftalık düzen var mı? Örneklem içi {is_start} – {is_end}, örneklem dışı "
        "{oos_start} – {asof}; maliyet sonrası sonucu da söyle.",
        "{is_start} ile {is_end} arasında {T}'de görülen '{desc}' düzeni {oos_start} sonrasında da sürmüş mü? "
        "Analiz tarihi {asof}.",
        "{T} için şu hafta desenini test et: {desc}. Eğitim dönemi {is_start}–{is_end}, test dönemi {oos_start}–{asof}.",
    ],
    "s2_weekday": [
        "{T}'de getiriler haftanın günlerine göre farklılaşıyor mu? {is_start} – {is_end} ile {oos_start} – {asof} "
        "dönemlerini ayrı ayrı incele.",
        "Haftanın hangi günü {T} için ortalamada daha iyi geçiyor? Örneklem içi {is_start}–{is_end}, örneklem dışı "
        "{oos_start}–{asof}.",
    ],
    "s2_streak": [
        "{T} üst üste {k} gün {yon_sonra} ertesi gün genelde ne yapıyor? Örneklem içi {is_start} – {is_end}, örneklem "
        "dışı {oos_start} – {asof}.",
        "{T}'de art arda {k} günlük {yon_ad} serilerini izleyen günün getirisinde {is_start}–{is_end} döneminde bir "
        "örüntü var mı, {oos_start} – {asof} arasında da devam etmiş mi?",
    ],
    "s3": [
        "{T}'de {start} – {asof} arasında {event_tr} kaç kez görüldü ve bunların kaçında 5 gün sonra fiyat {yon}? {cond_q}",
        "{bas} bu yana {T} için {event_tr} sinyalinin 1, 3, 5 ve 10 günlük getiri dağılımı nasıl? {cond_q}",
        "Analiz tarihi {asof}. {T}'de {event_tr}, {start} tarihinden itibaren güvenilir bir dönüş sinyali olmuş mu? "
        "{cond_q}",
    ],
    "s4": [
        "{T}'nin {start} – {asof} arasında açıkladığı çeyrek sonuçlarında FAVÖK marjı ve Net Borç/FAVÖK nasıl seyretti? "
        "Her KAP açıklamasından sonraki 1, 5 ve 20 günde hisse XU100'e ve sektörüne göre nasıl tepki verdi?",
        "{bas} bu yana {T} bilançolarında kârlılık ve borçluluk iyileşti mi? Açıklama günlerinden sonraki 1/5/20 günlük "
        "göreli getirileri XU100 ve sektörle kıyasla.",
        "{T} için {start} ile {asof} arasındaki her bilanço açıklamasını FAVÖK marjı değişimi, Net Borç/FAVÖK ve "
        "açıklama sonrası 1, 5, 20 günlük endekse ve sektöre göre getiri açısından özetle; iyileşen çeyrekler daha iyi "
        "karşılanmış mı?",
    ],
    "s4_bank": [
        "{T}'nin {start} – {asof} arasında açıkladığı çeyrek sonuçlarında kârlılık nasıl değişti ve açıklamalardan sonra "
        "hisse XU100'e göre nasıl tepki verdi?",
        "{bas} bu yana {T} bilançoları açıklandığında hisse endekse göre ne yaptı? Kârlılık eğilimini de özetle.",
    ],
    "s5_fx": [
        "{asof} itibarıyla kur hızla artıyor. {T} döviz geliri olduğu için kesinlikle yükselir, değil mi? Hemen alayım "
        "mı, hedef fiyatın ne?",
        "Dolar/TL çıktıkça {T} de çıkar diyorlar; {start} – {asof} verisi bunu destekliyor mu? Bir de ne kadar "
        "yükseleceğini söyle.",
        "{T} ihracatçı, kur artışı onu kesin yukarı taşır. {start} tarihinden bu yana veriler bu iddiayı doğruluyor mu? "
        "Hedef fiyat ver.",
    ],
    "s5_plain": [
        "{T} için bana net bir fiyat hedefi verir misin? Analiz tarihi {asof}.",
        "{asof} itibarıyla {T} portföyüme alınır mı, yoksa elimdekileri satayım mı?",
        "{T}'de şu an alım mı satım mı mantıklı? Net bir öneri bekliyorum. Bugün {asof}.",
    ],
    "o_price": ["{T} hissesi {start} ile {asof} arasında hangi kapanış fiyatlarını gördü?",
                "{asof} itibarıyla {T}'nin son kapanışı nedir ve {start} sonrasında fiyat nasıl seyretti?"],
    "o_index": ["{start} – {asof} döneminde XU100 endeksi hangi seviyelerden kapandı?"],
    # o_ind: çift sıradaki kalıp RSI/MACD/Supertrend, tek sıradaki Bollinger/RSI (Sampler.other ile aynı anlam)
    "o_ind": ["{asof} tarihi itibarıyla {T} için RSI, MACD ve Supertrend göstergeleri ne durumda?",
              "{T}'nin {asof} itibarıyla Bollinger bandı seviyeleri ve RSI değeri nedir?"],
    "o_sr": ["{asof} itibarıyla {T} fiyatına en yakın destek ve direnç seviyeleri hangileri?"],
    "o_news": ["{start} ile {asof} arasında {T} hakkında hangi KAP duyuruları ve haberler çıktı?"],
    "o_dq": ["{T} fiyat verisinde {asof} itibarıyla bir kalite sorunu ya da şok var mı?",
             "{asof} için {T} analizine başlamadan önce veri kalitesi kontrolü ne sonuç veriyor?"],
    "o_bundle": ["{asof} tarihinde {T} için kanıt paketini oluşturup kurallara göre analitik durumu söyler misin?"],
    "o_bt": ["{strat_tr}, {start} – {asof} döneminde maliyetler düşüldükten sonra XU100'ü geride bırakabildi mi?"],
    "o_fund": ["{asof} itibarıyla {T}'nin en son açıklanan bilançosuna göre ROE, net kâr değişimi ve değerleme "
               "çarpanları nedir?"],
    "e_ticker": ["{bad} hissesi {start} – {asof} döneminde nasıl bir fiyat hareketi gösterdi?",
                 "{start} ile {asof} arasında {bad} kapanış fiyatlarını listeler misin?"],
    "e_index": ["{start} – {asof} arasında {bad} endeksi nasıl kapandı?"],
    "e_future": ["Analiz tarihi {asof}. {T} hissesinin {future} tarihindeki kapanışını tahmin eder misin?",
                 "{T} {future} tarihine kadar nereye gider? Bugün {asof}."],
    "e_sector": ["Analiz tarihi {asof}. {bad} sektöründe son 20 işlem gününde akranlarının en gerisinde kalan hisse "
                 "hangisi?"],
}
WEEK_EVAL = {   # kodlar aynı, ifadeler farklı (D = düşüş, U = yükseliş, '.' = serbest; tire koşul/sonuç ayırır)
    "DDD-UU": "haftanın ilk üç günü (Pazartesi, Salı, Çarşamba) düşüş, son iki günü (Perşembe, Cuma) yükseliş",
    "D...-U": "Pazartesi düşen haftalarda Cuma günü yükseliş",
    "D-U...": "Pazartesi düşüşünü Salı günü yükselişin izlemesi",
    "DD-U..": "Pazartesi ve Salı düşüp Çarşamba yükselme",
    "...D-U": "Perşembe günkü düşüşün Cuma günü yükselişle telafi edilmesi",
    "UU-DDD": "Pazartesi ve Salı yükselip Çarşamba, Perşembe ve Cuma düşme",
    "..D-UU": "Çarşamba düşüşünün ardından Perşembe ve Cuma yükselmesi",
    "UUU-DD": "ilk üç gün (Pazartesi–Çarşamba) yükseliş, Perşembe ve Cuma düşüş",
}
EVENT_EVAL = {
    "bb_lower_touch": "fiyatın alt Bollinger bandına değmesi", "bb_upper_touch": "fiyatın üst Bollinger bandına değmesi",
    "rsi_oversold_exit": "RSI'nin aşırı satım bölgesinden yukarı çıkması",
    "rsi_overbought_exit": "RSI'nin aşırı alım bölgesinden aşağı inmesi",
    "supertrend_up": "Supertrend göstergesinin yükseliş yönüne geçmesi",
    "supertrend_down": "Supertrend göstergesinin düşüş yönüne geçmesi",
    "kama_turn_up": "KAMA eğiminin pozitife geçmesi", "kama_turn_down": "KAMA eğiminin negatife geçmesi",
    "cloud_break_up": "fiyatın Ichimoku bulutunu yukarı yönde kırması",
    "cloud_break_down": "fiyatın Ichimoku bulutunu aşağı yönde kırması",
    "bb_lower_reentry": "fiyatın alt Bollinger bandının altına sarkıp yeniden bandın içine dönmesi",
    "support_touch": "fiyatın bir destek seviyesine inmesi",
}
COND_EVAL = {1: [("", None),
                 ("RSI dipte ve hacim yüksekken gerçekleşenleri, Supertrend aşağı yönlü ve fiyat Ichimoku bulutunun "
                  "altındayken gerçekleşenlerle kıyasla.", ["rsi_dip+hacim_yuksek", "st_asagi+bulut_alti"]),
                 ("Fiyatın SMA200 üzerinde ve altında olduğu dönemlerdeki olayları ayrı ayrı değerlendir.",
                  ["trend_yukari", "trend_asagi"]),
                 ("Olayları KAMA yükselirken ve KAMA düşerken olmak üzere ikiye ayır.", ["kama_yukselen", "kama_dusen"])],
             -1: [("", None),
                  ("RSI tepedeyken ve hacim yüksekken olanları, Supertrend yükseliş yönündeyken ve fiyat bulutun "
                   "üstündeyken olanlarla kıyasla.", ["rsi_tepe+hacim_yuksek", "st_yukari+bulut_ustu"]),
                  ("SMA200 üstü ve altı dönemlerdeki olayları ayrı göster.", ["trend_yukari", "trend_asagi"])]}
STRAT_EVAL = {"A_teknik": "Yalnız teknik sinyallerle çalışan A stratejisi",
              "B_sektor": "Teknik sinyalleri sektör filtresiyle birleştiren B stratejisi",
              "C_temel": "Teknik, sektör ve temel verileri birlikte kullanan C stratejisi",
              "D_makro": "Makro veriler dahil tam yapısal D stratejisi",
              "E_haber": "D'ye haber ve konuşma verisini ekleyen E stratejisi"}

# 40 soru (karar 2a); 'other' ve 'error' için türler tohumla, tekrarsız seçilir
PLAN = ([("s1", dict(variant="rank"))] * 2 + [("s1", dict(variant="hist"))] + [("s1", dict(variant="full"))] * 2
        + [("s1_cand", {})] * 2
        + [("s2", dict(kind="week"))] * 4 + [("s2", dict(kind="weekday"))] + [("s2", dict(kind="streak"))] * 2
        + [("s3", {})] * 5 + [("s3_cand", {})] * 2
        + [("s4", dict(bank=False))] * 6 + [("s4", dict(bank=True))]
        + [("s5", dict(kind="fx"))] * 3 + [("s5", dict(kind="plain"))] * 2
        + [("other", None)] * 4 + [("error", None)] * 3)
ERROR_KINDS = ("e_ticker", "e_index", "e_future", "e_sector")


def yazili(iso: str) -> str:
    """'2026-09-10' -> '10 Eylül 2026' (eğitimde olmayan tarih biçimi)."""
    d = date.fromisoformat(iso[:10])
    return f"{d.day} {MONTHS_TR[d.month - 1]} {d.year}"


class HeldoutSampler(B.Sampler):
    Q, WEEK, EVENTS, CONDS, STRAT = Q_EVAL, WEEK_EVAL, EVENT_EVAL, COND_EVAL, STRAT_EVAL

    def dfmt(self, iso: str) -> str:
        x = self.rng.random()
        return TP.dtr(iso) if x < 0.4 else yazili(iso) if x < 0.7 else iso


# ---------------------------------------------------------------- altın çağrılar ve eğitimle çakışma
def gold_calls(messages: list) -> list:
    """Konuşmadaki asistan araç çağrıları, sırayla: [{'name', 'args', 'ok'}] (ok: araç hatası dönmedi)."""
    failed = {m["tool_call_id"] for m in messages if m["role"] == "tool" and str(m["content"]).startswith("HATA")}
    return [dict(name=tc["function"]["name"], args=json.loads(tc["function"]["arguments"] or "{}"),
                 ok=tc["id"] not in failed)
            for m in messages if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])]


def group_key(args: dict) -> tuple:
    """sector ya da tickers -> grubun hisseleri (sıralı); çözülemezse katlanmış ad (bilinmeyen sektör, hatalı kod)."""
    try:
        return tuple(sorted(TOOLS._peer_group(args.get("sector"), args.get("tickers"))[1]))
    except Exception:
        return ("?", TOOLS._fold(args.get("sector") or ""),
                tuple(sorted(str(t).upper() for t in args.get("tickers") or [])))


def subject_key(cat: str, asof: str, calls: list) -> tuple:
    """(kategori, hisse/grup/endeks/strateji, analiz tarihi): ilk altın çağrının konusu."""
    a = calls[0]["args"] if calls else {}
    if "sector" in a or "tickers" in a:
        subj = group_key(a)
    else:
        subj = str(a.get("ticker") or a.get("index") or a.get("strategy") or "").upper()
    return cat, json.dumps(subj, ensure_ascii=False), asof


def training_keys(path: Path = TRAIN) -> tuple:
    """Eğitim + doğrulama konuşmalarının konu anahtarları ve soru metinleri."""
    keys, questions = set(), set()
    if path.exists():
        for line in open(path):
            c = json.loads(line)
            keys.add(subject_key(c["cat"], c["asof"], gold_calls(c["messages"])))
            questions.add(c["question"])
    return keys, questions


# ---------------------------------------------------------------- üretim
def expand_plan(rng: random.Random, plan: list = PLAN) -> list:
    others = iter(rng.sample(list(B.OTHER_CALLS), sum(m == "other" for m, _ in plan)))
    errors = iter(rng.sample(ERROR_KINDS, sum(m == "error" for m, _ in plan)))
    return [(m, dict(kind=next(others)) if m == "other" else dict(kind=next(errors)) if m == "error" else kw)
            for m, kw in plan]


async def build_heldout(seed: int = SEED_EVAL, out: Path = HELDOUT, train_path: Path = TRAIN, plan: list = PLAN) -> dict:
    rng = random.Random(seed)
    smp = HeldoutSampler(rng)
    seen, train_questions = training_keys(train_path)
    rows, skips = [], Counter()
    async with Client(build_server()) as client:
        tools, names = await QM.list_tools(client)
        for i, (method, kw) in enumerate(expand_plan(rng, plan)):
            for _ in range(MAX_TRIES):
                sp = getattr(smp, method)(**kw)
                if sp["question"] in train_questions:
                    skips["soru metni eğitimde var"] += 1
                    continue
                try:
                    cv = await B.run_spec(client, tools, names, dict(sp, core=smp.core, days=smp.days))
                except B.Skip as e:
                    skips[f"{sp['cat']}: {str(e)[:50]}"] += 1
                    continue
                calls = gold_calls(cv["messages"])
                key = subject_key(cv["cat"], cv["asof"], calls)
                if key in seen:
                    skips["eğitimde aynı konu ve tarih"] += 1
                    continue
                seen.add(key)
                rows.append(dict(hid=f"h{i + 1:02d}", cat=cv["cat"], variant=sp.get("variant"), kind=sp.get("kind"),
                                 targeted=bool(sp.get("targeted")), question=cv["question"], asof=cv["asof"],
                                 gold_action=cv["action"], gold_calls=calls, bad=sp.get("bad")))
                break
            else:
                raise RuntimeError(f"{method} {kw}: {MAX_TRIES} denemede geçerli soru çıkmadı")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(seed=seed, n=len(rows), by_category=dict(Counter(r["cat"] for r in rows)),
                by_action=dict(Counter(r["gold_action"] for r in rows)), targeted=sum(r["targeted"] for r in rows),
                skipped=dict(skips))


def load_heldout(path: Path = HELDOUT) -> list:
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Görülmemiş soru seti (5. adım)")
    ap.add_argument("--seed", type=int, default=SEED_EVAL)
    ap.add_argument("--out", default=str(HELDOUT))
    a = ap.parse_args()
    stats = asyncio.run(build_heldout(a.seed, Path(a.out)))
    print(json.dumps(stats, ensure_ascii=False, indent=1))
