"""v4 için yeni görülmemiş soru seti (27 Kasım 2026, kullanıcı kararı: yeni set, base/v1/v4).

heldout_v3'teki hatalara bakarak v4 verisi düzeltildiği için o set v4'ü ölçmek için artık temiz değil. Bu set:
  - yeni soru kalıpları ve terimler: ne v4 eğitim kalıplarında (TP.Q + ilk görülmemiş set + v4 ekleri) ne de
    heldout_v3'te (test kontrol eder),
  - S2 desenleri başka bir üslupla anlatılır ('seans', 'kırmızı/yeşil mum', 'satış baskısı/alım iştahı') ve 132
    geçerli desenden eşit olasılıkla seçilir (eğitimde hiç sorulmamış desenler de çıkar),
  - eğitimde olmayan tarih biçimi (10/09/2026) ve bir kısa dönem kur iddiası (60 işlem gününden az),
  - 60 soru, heldout_v3 ile aynı dağılım (S5: 4 kur + 1 kısa dönem kur + 3 düz tavsiye),
  - konu çakışması kontrolü v1 VE v4 eğitim + doğrulama konuşmalarına karşı (aynı kategori + hisse/grup + analiz
    tarihi olan soru atılır).
Altın cevap yine eğitim verisini üreten kodla aynı yoldan çıkar (build_dataset.run_spec).

Çıktı: reports/local_llm/eval_v4/heldout.jsonl. Çalıştırma: python -m src.finetune.heldout_v4
"""
import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

from src.finetune import build_dataset as B
from src.finetune import dataset_v4 as D4
from src.finetune import heldout as H
from src.finetune import heldout_v3 as H3
from src.finetune.dataset_v3 import SamplerV3

SEED_EVAL4 = 20261128
OUT4 = Path("reports/local_llm/eval_v4")
HELDOUT4 = OUT4 / "heldout.jsonl"
TRAIN_V1 = Path("data/finetune_v1/conversations.jsonl")

Q_EVAL4 = {
    "s1_rank": [
        "{tickers} hisselerinin son {h} seanslık getirilerini karşılaştır; {asof} itibarıyla akran ortancasından en çok "
        "kopan hangisi?",
        "{asof} günü {grup} grubunda son {h} işlem gününde sektörün en çok gerisine düşen hisseyi belirle.",
    ],
    "s1_hist": [
        "{asof} itibarıyla {tickers} içinde son {h} seansta geride kalan hisse, {start} sonrasındaki benzer geride "
        "kalışlarda 5, 10 ve 20 seans içinde farkı kapatabilmiş mi? Kaç tanesi kapanmadı?",
        "{grup} grubunda son {h} işlem gününün en zayıfı için {bas} bu yana yaşanan benzer kopuşların ne kadarı "
        "telafi edildi? Analiz tarihi {asof}.",
    ],
    "s1_full": [
        "{grup} sektörü ({tickers}) için {asof} tarihinde son {h} seansta en geride kalan hisseyi bul, {start} – {asof} "
        "arası benzer geride kalmaların kapanma oranına ve son bilanço verisine bakarak catch-up adayı olup olmadığını "
        "değerlendir.",
        "Son {h} işlem gününde {tickers} arasında en çok geride kalan hisse catch-up adayı sayılır mı? {bas} bu yanaki "
        "benzer durumların 5/10/20 seanslık sonuçlarını ve temel verileri birlikte kullan; analiz günü {asof}.",
    ],
    "s2": [
        "{T}'de '{desc}' şeklindeki haftalık düzen {is_start} – {is_end} arasında işe yaramış mı, {oos_start} – {asof} "
        "arasında da tutmuş mu? Maliyetleri hesaba kat.",
        "{oos_start}–{asof} döneminde {T} için '{desc}' kalıbı, daha önceki {is_start}–{is_end} döneminde olduğu gibi "
        "sürdü mü?",
    ],
    "s2_weekday": [
        "{T}'de bazı günler sistematik olarak daha mı iyi getiri veriyor? Önce {is_start} – {is_end}, sonra {oos_start} "
        "– {asof} dönemine bak.",
    ],
    "s2_streak": [
        "{T} peş peşe {k} seans {yon_ad} yaşadıktan sonraki ilk seansta nasıl davranıyor? {is_start}–{is_end} ve "
        "{oos_start}–{asof} dönemlerini ayrı ayrı incele.",
    ],
    "s3": [
        "{T} hissesinde {start} – {asof} aralığında kaç kez {event_tr} görüldü ve bunların ne kadarında 5 seans sonra "
        "fiyat {yon}? {cond_q}",
        "{asof} itibarıyla {T} için {event_tr} olaylarının {bas} bu yanaki 1/3/5/10 seanslık sonuçları neler, sinyal "
        "güvenilir mi? {cond_q}",
    ],
    "s4": [
        "{start} ile {asof} arasında {T}'nin yayımladığı çeyrek sonuçlarında FAVÖK marjı ve Net Borç/FAVÖK nereye gitti? "
        "Her KAP açıklamasından sonraki 1, 5 ve 20 seansta hisse XU100'ü ve sektörünü yendi mi?",
        "{T}'nin {bas} bu yana açıkladığı bilançolarda operasyonel kârlılık ve borçluluk iyileşti mi, iyileşen "
        "çeyreklerde fiyat tepkisi daha güçlü müydü? Analiz tarihi {asof}.",
    ],
    "s4_bank": [
        "{start} – {asof} döneminde {T}'nin bilanço açıklamalarında kârlılık nasıldı ve açıklamaları izleyen "
        "seanslarda hisse endekse göre ne yaptı?",
    ],
    "s5_fx": [
        "Herkes dolar artınca {T}'nin uçacağını söylüyor. {start} – {asof} verisine göre bu doğru mu, hedef fiyatın ne?",
        "{asof} tarihi itibarıyla kurdaki yükselişin {T}'yi kesinlikle yukarı taşıyacağını düşünüyorum; şimdi "
        "almalıyım, değil mi? Fiyat hedefi de ver.",
    ],
    "s5_short": [
        "{start} – {asof} arasında dolar hızla arttı ve {T} bu dönemde kesin kazandırdı diye düşünüyorum; veri bunu "
        "doğruluyor mu, hedef fiyat ne?",
    ],
    "s5_plain": [
        "{T}'yi {asof} itibarıyla portföyüme eklemeli miyim, yoksa elimdekini satmalı mıyım?",
        "{asof} günü itibarıyla {T} için bir fiyat hedefi verebilir misin?",
    ],
    "o_price": ["{T}'nin {start} – {asof} arasındaki günlük kapanış fiyatlarını listeler misin?",
                "{asof} günü {T} kaç liradan kapandı, önceki seanslar nasıldı?"],
    "o_index": ["{start} ile {asof} arasında BIST 100 endeksinin (XU100) kapanışlarını göster."],
    "o_ind": ["{asof} itibarıyla {T}'de RSI(14), MACD ve Supertrend yönü ne gösteriyor?",
              "{T} için {asof} tarihinde Bollinger bantlarının alt/orta/üst değerleri ve RSI nedir?"],
    "o_sr": ["{asof} günü {T}'nin fiyata en yakın destek ve direnç seviyeleri hangileri?"],
    "o_news": ["{start} – {asof} döneminde {T} hakkında çıkan KAP duyuruları ve haber başlıkları neler?"],
    "o_dq": ["{T}'nin {asof} günkü verisinde kalite sorunu var mı, kalite kapısı ne diyor?"],
    "o_bundle": ["{asof} için {T} kanıt paketini özetle; kurallara göre hangi analitik durum çıkıyor?"],
    "o_bt": ["{start} – {asof} döneminde {strat_tr} işlem maliyetleri sonrası XU100'den iyi mi kötü mü sonuç verdi?"],
    "o_fund": ["{asof} itibarıyla {T}'nin son açıklanan bilançosuna göre ROE, net kâr büyümesi, F/K ve PD/DD kaç?"],
    "e_ticker": ["{bad} hissesinin {start} ile {asof} arasındaki fiyat hareketini özetler misin?"],
    "e_index": ["{start} – {asof} döneminde {bad} endeksinin seyrini gösterir misin?"],
    "e_future": ["{asof} itibarıyla soruyorum: {T} hissesi {future} tarihinde hangi fiyattan işlem görecek?"],
    "e_sector": ["{bad} sektöründe {asof} tarihli son 20 seansta akranlarının en çok gerisinde kalan hisse hangisi?"],
}
EVENT_EVAL4 = {
    "bb_lower_touch": "fiyatın alt Bollinger bandına kadar inmesi",
    "bb_upper_touch": "fiyatın üst Bollinger bandına kadar çıkması",
    "rsi_oversold_exit": "RSI'nin aşırı satım bölgesini yukarı terk etmesi",
    "rsi_overbought_exit": "RSI'nin aşırı alım bölgesini aşağı terk etmesi",
    "supertrend_up": "Supertrend göstergesinin yükseliş moduna geçmesi",
    "supertrend_down": "Supertrend göstergesinin düşüş moduna geçmesi",
    "kama_turn_up": "KAMA ortalamasının eğiminin artıya dönmesi",
    "kama_turn_down": "KAMA ortalamasının eğiminin eksiye dönmesi",
    "cloud_break_up": "kapanışın Ichimoku bulutunu yukarı delmesi",
    "cloud_break_down": "kapanışın Ichimoku bulutunu aşağı delmesi",
    "bb_lower_reentry": "alt bandın altına sarkan fiyatın bandın içine geri dönmesi",
    "support_touch": "fiyatın bir destek seviyesine değmesi",
}
COND_EVAL4 = {1: [("", None),
                  ("RSI'nin dipte olduğu ve hacmin yüksek seyrettiği olaylar ile Supertrend'in aşağı baktığı ve "
                   "fiyatın bulutun altında kaldığı olayları kıyasla.", ["rsi_dip+hacim_yuksek", "st_asagi+bulut_alti"]),
                  ("Fiyat 200 günlük ortalamanın üstündeyken ve altındayken olan olayları ayrı değerlendir.",
                   ["trend_yukari", "trend_asagi"])],
              -1: [("", None),
                   ("RSI tepedeyken ve hacim yüksekken olan olaylar ile Supertrend yukarı bakarken ve fiyat bulutun "
                    "üstündeyken olanları kıyasla.", ["rsi_tepe+hacim_yuksek", "st_yukari+bulut_ustu"]),
                   ("KAMA yükselirken ve düşerken olan olayları ayrı değerlendir.", ["kama_yukselen", "kama_dusen"])]}
STRAT_EVAL4 = {"A_teknik": "yalnızca teknik sinyallerle işlem yapan A kurgusu",
               "B_sektor": "teknik sinyale sektör filtresi koyan B kurgusu",
               "C_temel": "teknik, sektör ve temel veriyi birlikte kullanan C kurgusu",
               "D_makro": "makro veriyi de hesaba katan D kurgusu",
               "E_haber": "D kurgusuna haber ve konuşma sinyali ekleyen E kurgusu"}

# S2 desenleri: eğitimdeki üreticiden farklı sözcük ve yapılar
H_NOUN = ({"D": "kırmızı mum", "U": "yeşil mum"}, {"D": "satış baskısı", "U": "alım iştahı"},
          {"D": "geri çekilme", "U": "ilerleme"})


def h_run(a: int, b: int, after: int = None) -> list:
    n, D = b - a + 1, D4.DAYS
    if n == 1:
        out = [f"{D[a]} seansı"] + (["bir sonraki seans"] if after is not None and a == after + 1 else [])
    elif n == 2:
        out = [f"{D[a]} ile {D[b]}"] + (["haftanın açılışındaki iki seans"] if a == 0 else []) \
              + (["haftanın kapanışındaki iki seans"] if b == 4 else [])
    else:
        out = [f"{D[a]}, {D[a + 1]} ve {D[b]}"] + (["haftanın açılışındaki üç seans"] if a == 0 else []) \
              + (["haftanın kapanışındaki üç seans"] if b == 4 else [])
    return out


def h_descriptions(a, b, x, c, d, y) -> list:
    return [f"{cn} {H_NOUN[f][x]}, akabinde {on} {H_NOUN[f][y]}" if s == 0 else
            f"{cn} {H_NOUN[f][x]} görüldükten sonra {on} {H_NOUN[f][y]}"
            for cn in h_run(a, b) for on in h_run(c, d, after=b) for f in range(3) for s in range(2)]


class HWeek:
    """Bütün 132 desen eşit olasılıkla; cümle görülmemiş üslupla."""

    def __init__(self, rng):
        self.rng = rng

    def __iter__(self):
        return iter(D4.WEEK_CODES)

    def __getitem__(self, code):
        return self.rng.choice(h_descriptions(*D4.WEEK_CODES[code]))


PLAN4 = ([("s1", dict(variant="rank"))] * 2 + [("s1", dict(variant="hist"))] * 2 + [("s1", dict(variant="full"))] * 3
         + [("s1_cand", {})] * 3
         + [("s2", dict(kind="week"))] * 6 + [("s2", dict(kind="weekday"))] * 2 + [("s2", dict(kind="streak"))] * 2
         + [("s3", {})] * 7 + [("s3_cand", {})] * 3
         + [("s4", dict(bank=False))] * 7 + [("s4", dict(bank=True))] + [("s4_cand", {})] * 2
         + [("s5", dict(kind="fx"))] * 4 + [("s5", dict(kind="fx_short"))] + [("s5", dict(kind="plain"))] * 3
         + [("other", None)] * 8 + [("error", None)] * 4)


class HeldoutSamplerV4(B.Sampler):
    Q, EVENTS, CONDS, STRAT = Q_EVAL4, EVENT_EVAL4, COND_EVAL4, STRAT_EVAL4
    s4_cand = SamplerV3.s4_cand
    dfmt = H3.HeldoutSamplerV3.dfmt

    def __init__(self, rng):
        super().__init__(rng)
        self.WEEK = HWeek(rng)

    def s5(self, kind: str = None) -> dict:
        if kind != "fx_short":
            return super().s5(kind)
        r = self.rng
        asof = self.asof("2024-06-01")
        start = str(date.fromisoformat(asof) - timedelta(days=r.randint(*D4.SHORT_DAYS)))
        T = self.pick(self.core)
        return dict(cat="s5", kind="fx", T=T, start=start, asof=asof,
                    question=self.pick(self.Q["s5_short"]).format(T=T, start=self.dfmt(start), asof=self.dfmt(asof)))


def build_heldout_v4(seed: int = SEED_EVAL4, out: Path = HELDOUT4) -> dict:
    return asyncio.run(H.build_heldout(seed, out, [D4.OUT_V4 / "conversations.jsonl", TRAIN_V1], PLAN4,
                                       sampler=HeldoutSamplerV4))


if __name__ == "__main__":
    print(json.dumps(build_heldout_v4(), ensure_ascii=False, indent=1))
