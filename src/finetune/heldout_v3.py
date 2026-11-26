"""v3 için yeni görülmemiş soru seti (26 Kasım 2026).

İlk görülmemiş setin (heldout.py) kalıpları artık v3 eğitim verisinde olduğu için v3 bu setle ölçülemez. Bu set:
  - tamamen yeni soru kalıpları ve terimler (eğitim kalıplarında da ilk görülmemiş sette de yok; test kontrol eder),
  - eğitimde olmayan bir tarih biçimi (10/09/2026) ve iki yeni koşul birleşimi,
  - 60 soru (40 yerine; her soru 2,5 değil ~1,7 puan): S1-S4 10'ar, S5 8, diğer araçlar 8, hatalı istek 4; kategori
    içinde hedefli adaylar (catch-up 3, teknik dönüş 3, bilanço iyileşmesi 2),
  - konu çakışması kontrolü v3 eğitim + doğrulama konuşmalarına karşı (aynı kategori + hisse/grup + analiz tarihi
    olan soru atılır).
Altın cevap yine eğitim verisini üreten kodla aynı yoldan çıkar (build_dataset.run_spec, v3 kural satırıyla).

Çıktı: reports/local_llm/eval_v3/heldout.jsonl. Çalıştırma: python -m src.finetune.heldout_v3
"""
import asyncio
import json
from pathlib import Path

from src.finetune import build_dataset as B
from src.finetune import heldout as H
from src.finetune import templates as TP
from src.finetune.dataset_v3 import OUT_V3, SamplerV3

SEED_EVAL3 = 20261127
OUT3 = Path("reports/local_llm/eval_v3")
HELDOUT3 = OUT3 / "heldout.jsonl"

Q_EVAL3 = {
    "s1_rank": [
        "{asof} tarihinde {tickers} arasında son {h} seansın getirisine göre akranlarından en çok geri düşen hisse "
        "hangisi?",
        "{grup} hisselerini son {h} işlem günündeki performanslarına göre sırala; sektör ortancasının en altında hangisi "
        "kaldı?",
    ],
    "s1_hist": [
        "{grup} grubunda ({tickers}) son {h} seansta en çok geri kalan hisse için {start} ile {asof} arasındaki benzer "
        "durumlara bak: akran farkı 5, 10 ve 20 seans içinde kapandı mı, kaçı boşa çıktı?",
        "Son {h} işlem gününde {tickers} içinde en zayıf kalan hissenin, {bas} bu yana bu kadar geride kaldığı "
        "dönemlerden sonra akranlarını yakalama oranı nedir? Analiz günü {asof}.",
    ],
    "s1_full": [
        "{asof} itibarıyla {grup} sektöründe son {h} seansın en zayıf hissesi hangisi? {start} – {asof} arasındaki "
        "benzer geride kalmaların 5/10/20 seanslık kapanma oranlarını ve son bilançoyu birlikte değerlendirip "
        "potansiyel catch-up adayı olup olmadığını söyle.",
        "{tickers} arasında son {h} işlem gününde akran ortancasının en çok altında kalan hisse, {bas} bu yana benzer "
        "geride kalmaların ne kadarının kapandığına ve temel verilerine göre catch-up adayı mı?",
    ],
    "s2": [
        "{T} hisse senedinde haftalık şu düzeni sına: {desc}. {is_start} – {is_end} arası örneklem içi, {oos_start} – "
        "{asof} arası örneklem dışı olsun; işlem maliyeti sonrası ne kalıyor?",
        "{oos_start} ile {asof} arasında {T} için '{desc}' düzeni, {is_start}–{is_end} döneminde görüldüğü gibi "
        "geçerliliğini korudu mu?",
    ],
    "s2_weekday": [
        "{T}'nin günlük getirilerinde haftanın gününe bağlı bir fark var mı? {is_start}–{is_end} dönemini "
        "{oos_start}–{asof} dönemiyle sına.",
    ],
    "s2_streak": [
        "{T} hissesi {k} seans üst üste {yon_sonra} bir sonraki seansta ortalama ne oluyor? {is_start} – {is_end} ile "
        "{oos_start} – {asof} dönemlerini karşılaştır.",
    ],
    "s3": [
        "{start} ile {asof} arasında {T} hissesinde {event_tr} olaylarını say; 5 seans sonra fiyatın {yon} olaylar "
        "hangi oranda? {cond_q}",
        "{T} için {bas} bu yana {event_tr} sinyalinin ardından 1, 3, 5 ve 10 seanslık getiriler nasıl dağılıyor, sinyal "
        "işe yarıyor mu? Analiz günü {asof}. {cond_q}",
    ],
    "s4": [
        "{T} {start} – {asof} arasında açıkladığı her çeyrekte FAVÖK marjını ve Net Borç/FAVÖK oranını nasıl "
        "değiştirdi? KAP açıklamalarını izleyen 1, 5 ve 20 seansta hisse endeksi ve sektörünü geçti mi?",
        "{bas} bu yana {T}'nin çeyreklik sonuçlarında FAVÖK marjı ve borçluluk iyileşti mi; iyileşen çeyreklerin "
        "ardından 1/5/20 seanslık getiri XU100 ve sektörle kıyaslandığında daha mı iyi? Analiz günü {asof}.",
    ],
    "s4_bank": [
        "{T}'nin {start} ile {asof} arasında açıkladığı bilançolarda kârlılık nasıl seyretti, açıklamaların ardından "
        "hisse XU100'e karşı nasıl performans gösterdi?",
    ],
    "s5_fx": [
        "Kur yükseldikçe {T} kazanır, bu kesin. {start} – {asof} verisi bunu kanıtlıyor mu? Kaç TL'ye çıkar, "
        "söyler misin?",
        "{asof} itibarıyla dolar hızla yükseliyor; {T} ihracat yaptığı için şimdi alınmalı, yanılıyor muyum? Bir de "
        "hedef fiyat ver.",
    ],
    "s5_plain": [
        "{T} için alım mı satım mı önerirsin? Analiz günü {asof}.",
        "{asof} itibarıyla {T} hissesinin ulaşacağı fiyatı tahmin edip hedef verir misin?",
    ],
    "o_price": ["{start} ile {asof} arasında {T} hissesi hangi fiyatlardan kapandı?",
                "{T}'nin {asof} kapanışı ve ondan önceki birkaç seansın kapanışları neler?"],
    "o_index": ["XU100 {start} – {asof} aralığında hangi kapanış değerlerini gördü?"],
    "o_ind": ["{T} hissesinin {asof} günündeki RSI, MACD ve Supertrend okumaları neler?",
              "{asof} günü {T} için alt, orta ve üst Bollinger bandı ile RSI kaç?"],
    "o_sr": ["{T} için {asof} günü fiyatın altındaki ve üstündeki en yakın destek/direnç nerede?"],
    "o_news": ["{T} ile ilgili {start} – {asof} arasında yayımlanan KAP bildirimleri ve haberler nelerdi?"],
    "o_dq": ["{asof} günü {T} verisi analiz için yeterince temiz mi, kalite kapısının sonucu ne?"],
    "o_bundle": ["{T} için {asof} tarihli kanıt paketinde neler var ve kurallar hangi analitik durumu seçiyor?"],
    "o_bt": ["{strat_tr} {start} ile {asof} arasında maliyetler dahil XU100'ün önüne geçebildi mi?"],
    "o_fund": ["{T}'nin {asof} gününe kadar açıklanan en son bilançosunda ROE, net kâr değişimi ve F/K, PD/DD nedir?"],
    "e_ticker": ["{start} – {asof} arasında {bad} hissesinin kapanışlarını gösterir misin?"],
    "e_index": ["{bad} endeksi {start} ile {asof} arasında hangi seviyelerden kapandı?"],
    "e_future": ["Analiz günü {asof}. {T} {future} günü kaçtan kapanacak?"],
    "e_sector": ["{asof} itibarıyla {bad} sektörünün son 20 seansında en çok geri kalan hisse hangisi?"],
}
WEEK_EVAL3 = {
    "DDD-UU": "Pazartesi'den Çarşamba'ya üç gün düşüp Perşembe ve Cuma toparlanma",
    "D...-U": "haftaya düşüşle başlanınca (Pazartesi) haftanın Cuma günü yükselişle kapanması",
    "D-U...": "Pazartesi kaybının hemen ertesi gün (Salı) geri alınması",
    "DD-U..": "haftanın ilk iki günü kayıp, Çarşamba kazanç",
    "...D-U": "Perşembe kaybından sonra Cuma kazancı",
    "UU-DDD": "ilk iki gün (Pazartesi, Salı) kazanç, kalan üç gün kayıp",
    "..D-UU": "Çarşamba kaybının ardından haftanın son iki günü kazanç",
    "UUU-DD": "Pazartesi–Salı–Çarşamba kazanç, Perşembe–Cuma kayıp",
}
EVENT_EVAL3 = {
    "bb_lower_touch": "gün içinde alt Bollinger bandına inilmesi",
    "bb_upper_touch": "gün içinde üst Bollinger bandına çıkılması",
    "rsi_oversold_exit": "RSI'nin 30 altından yukarı dönmesi",
    "rsi_overbought_exit": "RSI'nin 70 üstünden aşağı dönmesi",
    "supertrend_up": "Supertrend'in al yönüne dönmesi", "supertrend_down": "Supertrend'in sat yönüne dönmesi",
    "kama_turn_up": "KAMA'nın yukarı kıvrılması", "kama_turn_down": "KAMA'nın aşağı kıvrılması",
    "cloud_break_up": "fiyatın Ichimoku bulutunun üstüne çıkması",
    "cloud_break_down": "fiyatın Ichimoku bulutunun altına inmesi",
    "bb_lower_reentry": "alt Bollinger bandının dışına çıkan fiyatın yeniden bant içine girmesi",
    "support_touch": "fiyatın destek seviyesine kadar gerilemesi",
}
COND_EVAL3 = {1: [("", None),
                  ("RSI dibinde ve hacmin yüksek olduğu olaylarla Supertrend düşüşteyken ve fiyat bulutun altındayken "
                   "olan olayları ayrı ayrı özetle.", ["rsi_dip+hacim_yuksek", "st_asagi+bulut_alti"]),
                  ("Yüksek hacimli olaylarla RSI dibindeki olayları ayrı ayrı göster.", ["hacim_yuksek", "rsi_dip"])],
              -1: [("", None),
                   ("RSI tepesinde ve hacim yüksekken olan olaylarla Supertrend yükselişteyken ve fiyat bulutun "
                    "üstündeyken olanları ayrı ayrı özetle.", ["rsi_tepe+hacim_yuksek", "st_yukari+bulut_ustu"]),
                   ("KAMA düşerken ve yatay seyrederken olan olayları ayır.", ["kama_dusen", "kama_yatay"])]}
STRAT_EVAL3 = {"A_teknik": "Sadece teknik göstergelere dayanan A stratejisi",
               "B_sektor": "Teknik göstergelere sektör filtresi ekleyen B stratejisi",
               "C_temel": "Teknik, sektör ve bilanço verisini birleştiren C stratejisi",
               "D_makro": "Makro göstergeleri de kullanan D stratejisi",
               "E_haber": "D stratejisine haber ve konuşma sinyallerini katan E stratejisi"}

PLAN3 = ([("s1", dict(variant="rank"))] * 2 + [("s1", dict(variant="hist"))] * 2 + [("s1", dict(variant="full"))] * 3
         + [("s1_cand", {})] * 3
         + [("s2", dict(kind="week"))] * 6 + [("s2", dict(kind="weekday"))] * 2 + [("s2", dict(kind="streak"))] * 2
         + [("s3", {})] * 7 + [("s3_cand", {})] * 3
         + [("s4", dict(bank=False))] * 7 + [("s4", dict(bank=True))] + [("s4_cand", {})] * 2
         + [("s5", dict(kind="fx"))] * 5 + [("s5", dict(kind="plain"))] * 3
         + [("other", None)] * 8 + [("error", None)] * 4)


def slash(iso: str) -> str:
    """'2026-09-10' -> '10/09/2026' (eğitimde olmayan tarih biçimi)."""
    return f"{iso[8:10]}/{iso[5:7]}/{iso[:4]}"


class HeldoutSamplerV3(B.Sampler):
    Q, WEEK, EVENTS, CONDS, STRAT = Q_EVAL3, WEEK_EVAL3, EVENT_EVAL3, COND_EVAL3, STRAT_EVAL3
    s4_cand = SamplerV3.s4_cand

    def dfmt(self, iso: str) -> str:
        x = self.rng.random()
        return slash(iso) if x < 0.4 else H.yazili(iso) if x < 0.6 else TP.dtr(iso) if x < 0.8 else iso


def build_heldout_v3(seed: int = SEED_EVAL3, out: Path = HELDOUT3) -> dict:
    return asyncio.run(H.build_heldout(seed, out, OUT_V3 / "conversations.jsonl", PLAN3, sampler=HeldoutSamplerV3))


if __name__ == "__main__":
    print(json.dumps(build_heldout_v3(), ensure_ascii=False, indent=1))
