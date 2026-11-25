"""İnce ayar veri seti (yerel model 3. adım): soru kalıpları, Türkçe sayı biçimi, cevap yazıcıları ve eylem kuralları.

Cevaplar yalnızca modelin gördüğü (kısaltılmış) araç çıktılarındaki sayıları kullanır; her cevap üretimden sonra soru
modunun kendi denetiminden (src/harness/question_mode.validate_answer) geçmek zorundadır.

Eylem kuralları (kullanıcı kararı, 24 Kasım 2026):
  S1 geride kalma : POTENTIAL_CATCH_UP_CANDIDATE <=> sektör güçlü (grup ortancası > 0) VE hissede >= 10 tamamlanmış
                    olay VE 20 günde isabet >= %60 VE yanlış pozitif <= %40; olay < 10 -> INVESTIGATE; aksi WATCH.
                    Yalnız sıralama sorulduysa (geçmiş test edilmedi) WATCH.
  S2 takvim       : araç 'kararsız' ya da maliyet sonrası <= 0 -> REJECT_SIGNAL; aksi WATCH.
  S3 teknik       : TECHNICAL_REVERSAL_CANDIDATE <=> >= 10 olay VE 5 günde başarı >= %60 VE son 3 işlem gününde olay;
                    olay < 10 -> INVESTIGATE; başarı <= %50 -> REJECT_SIGNAL; aksi WATCH.
  S4 bilanço      : FUNDAMENTAL_IMPROVEMENT_CANDIDATE <=> son çeyrek iyileşen VE <= 63 işlem günü önce açıklandı;
                    iyileşme bilinmiyorsa INVESTIGATE; aksi HOLD.
  S5 tavsiye      : tavsiye/hedef fiyat reddedilir; iddia desteklenmiyorsa (beta p >= 0,05 ya da beta <= 0)
                    INVESTIGATE, destekleniyorsa WATCH. Düz tavsiye sorusunda kanıt paketinin kurallı eylemi.
  Diğer araçlar   : betimleyici sorular HOLD; kalite kapısı / kanıt paketi kendi durumu; veri sorunu ANALYSIS_UNSAFE.
"""
from datetime import date

STABLE_COST = 0.30          # gidiş-dönüş maliyet (%), araç uyarısındaki "%0.30" ile aynı
FUND_RECENT = 63            # src/harness/evidence.FUND_RECENT (işlem günü)
SECTOR_TR = {"Automotive & Durables": "otomotiv ve dayanıklı tüketim", "Banking": "bankacılık",
             "Chemicals & Energy": "kimya ve enerji", "Construction RE & Utilities": "inşaat, GYO ve elektrik",
             "Healthcare": "sağlık", "Metals & Mining": "metal ve madencilik",
             "Other Financials": "holding, sigorta ve aracı kurum", "Retail & Consumer": "perakende",
             "Tech Telecom & Defense": "teknoloji, telekom ve savunma", "Transportation": "ulaştırma"}
DAY_TR = {"Pzt": "Pazartesi", "Sal": "Salı", "Çar": "Çarşamba", "Per": "Perşembe", "Cum": "Cuma"}
WEEK_PATTERNS = {   # kod -> sorudaki ifade
    "DDD-UU": "Pazartesi–Salı–Çarşamba düşüş, ardından Perşembe–Cuma yükseliş",
    "D...-U": "Pazartesi düşüşünün ardından Cuma yükselişi",
    "D-U...": "Pazartesi düşüşünün ardından Salı yükselişi",
    "DD-U..": "Pazartesi–Salı düşüş, ardından Çarşamba yükseliş",
    "...D-U": "Perşembe düşüşünün ardından Cuma yükselişi",
    "UU-DDD": "Pazartesi–Salı yükseliş, ardından Çarşamba–Perşembe–Cuma düşüş",
    "..D-UU": "Çarşamba düşüşünün ardından Perşembe–Cuma yükselişi",
    "UUU-DD": "Pazartesi–Salı–Çarşamba yükseliş, ardından Perşembe–Cuma düşüş",
}
EVENT_TR = {"bb_lower_touch": "alt Bollinger bandına dokunma", "bb_upper_touch": "üst Bollinger bandına dokunma",
            "rsi_oversold_exit": "RSI'nin aşırı satımdan çıkışı", "rsi_overbought_exit": "RSI'nin aşırı alımdan çıkışı",
            "supertrend_up": "Supertrend'in yukarı dönmesi", "supertrend_down": "Supertrend'in aşağı dönmesi",
            "kama_turn_up": "KAMA eğiminin yukarı dönmesi", "kama_turn_down": "KAMA eğiminin aşağı dönmesi",
            "cloud_break_up": "Ichimoku bulutunun yukarı kırılması", "cloud_break_down": "Ichimoku bulutunun aşağı kırılması",
            "bb_lower_reentry": "alt Bollinger bandının altından banda geri dönüş", "support_touch": "desteğe dokunma"}
COND_TR = {"rsi_dip": "RSI dibi (son 3 günde RSI ≤ 35)", "rsi_tepe": "RSI tepesi (son 3 günde RSI ≥ 65)",
           "hacim_yuksek": "yüksek hacim", "st_asagi": "Supertrend düşüş yönünde", "st_yukari": "Supertrend yükseliş yönünde",
           "bulut_alti": "fiyat Ichimoku bulutunun altında", "bulut_ustu": "fiyat Ichimoku bulutunun üstünde",
           "trend_yukari": "fiyat SMA200 üstünde", "trend_asagi": "fiyat SMA200 altında",
           "kama_yukselen": "KAMA yükselen", "kama_dusen": "KAMA düşen", "kama_yatay": "KAMA yatay"}
DISCLAIMER = "- Geçmiş sonuçlar gelecek getiri garantisi değildir; bu bir yatırım tavsiyesi değildir."
REFUSAL = "- Yatırım tavsiyesi, al/sat önerisi ya da hedef fiyat veremem; bunun yerine geçmiş veride ne olduğunu gösteriyorum."


# ---------------------------------------------------------------- biçim
def num(x, d: int = 2) -> str:
    """Türkçe sayı: 1.234,56 ; negatif '-' ile."""
    if x is None:
        return "bilinmiyor"
    x = round(float(x), d)
    if x == 0:
        x = 0.0
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("-" if x < 0 else "") + s


def pct(r, d: int = 2) -> str:
    """Oran -> yüzde: 0.1126 -> %11,26 ; -0.0506 -> -%5,06."""
    if r is None:
        return "bilinmiyor"
    v = round(float(r) * 100, d)
    return ("-" if v < 0 else "") + "%" + num(abs(v), d)


def pv(v, d: int = 2) -> str:
    """Zaten yüzde olan değer: 1.5023 -> %1,50."""
    if v is None:
        return "bilinmiyor"
    v = round(float(v), d)
    return ("-" if v < 0 else "") + "%" + num(abs(v), d)


def pts(v, d: int = 2) -> str:
    return "bilinmiyor" if v is None else f"{num(v, d)} puan"


def dtr(s) -> str:
    """'2026-09-10' -> '10.09.2026'."""
    s = str(s)[:10]
    return f"{s[8:10]}.{s[5:7]}.{s[:4]}"


def per(s) -> str:
    """Dönem sonu '2026-06-30' -> '2026/06'."""
    s = str(s)[:10]
    return f"{s[:4]}/{s[5:7]}"


def days(a, b) -> int:
    return (date.fromisoformat(str(b)[:10]) - date.fromisoformat(str(a)[:10])).days


def finish(lines: list, action: str, why: str) -> tuple:
    return "\n".join(lines + [f"- Eylem gerekçesi: {why}", f"ANALİTİK DURUM: {action}"]), action


# ---------------------------------------------------------------- soru kalıpları (slot: {..})
Q = {
    "s1_rank": [
        "{grup} sektöründe ({tickers}) {asof} itibarıyla son {h} işlem gününde sektör medyanının en çok gerisinde kalan hisse hangisi?",
        "{tickers} arasında son {h} günde akranlarına göre en zayıf kalan hisse hangisi? Analiz tarihi {asof}.",
        "{asof} tarihinde {grup} grubunda son {h} işlem günlük getiri sıralaması nasıl, en geride kalan hangisi?",
        "{grup} hisseleri ({tickers}) içinde son {h} günde sektör ortancasının en çok altında kalan hisseyi bul.",
    ],
    "s1_hist": [
        "{grup} sektöründe ({tickers}) {asof}'e kadar olan son {h} işlem gününde sektör medyanının en çok gerisinde kalan hisse hangisi? {start} – {asof} döneminde benzer geride kalmalar 5, 10 ve 20 gün içinde kapanmış mı?",
        "{tickers} içinde son {h} günde en geride kalan hissenin {start} sonrasındaki benzer geride kalmaları toparlanmış mı? Yanlış pozitifleri de söyle.",
        "Son {h} işlem gününde {grup} grubunun en zayıf hissesi hangisi ve geçmişte ({start} – {asof}) bu kadar geride kaldığında akranlarını yakalamış mı?",
        "{asof} itibarıyla {grup} sektöründe akran ortancasının en çok gerisinde kalan hisse için {start}'ten bu yana geride kalmaların 5/10/20 günlük kapanma oranlarını ver.",
    ],
    "s1_full": [
        "{grup} sektöründe ({tickers}) {asof}'e kadar olan son {h} işlem gününde sektör medyanının en çok gerisinde kalan hisse hangisi? {start} – {asof} döneminde benzer geride kalmalar 5, 10 ve 20 gün içinde kapanmış mı? Hisse POTENTIAL_CATCH_UP_CANDIDATE olarak etiketlenebilir mi?",
        "{tickers} arasında son {h} günde en çok geride kalan hisse catch-up adayı mı? {start}'ten beri benzer geride kalmaların kapanıp kapanmadığını ve geride kalmanın temel verilere göre kalıcı mı geçici mi olduğunu değerlendir.",
        "{grup} grubunda son {h} günün en zayıf hissesini bul; geçmiş geride kalmalarını ({start} – {asof}) ve son bilançosunu kontrol ederek potansiyel catch-up adayı olup olmadığını söyle.",
    ],
    "s2": [
        "{T}'de {is_start} – {is_end} döneminde '{desc}' örüntüsü var mı? Bu örüntü {oos_start} – {asof} döneminde de devam ediyor mu?",
        "{T} hissesinde {desc} şeklinde bir hafta içi düzen görülüyor mu? {is_start}–{is_end} ile {oos_start}–{asof} dönemlerini ayrı ayrı test et.",
        "{is_start} ile {is_end} arasında {T}'de '{desc}' deseni istatistiksel olarak anlamlı mı, örneklem dışında ({oos_start} sonrası, {asof}'e kadar) kalıcı mı?",
    ],
    "s2_weekday": [
        "{T}'de haftanın günü etkisi var mı? {is_start} – {is_end} dönemini {oos_start} – {asof} dönemiyle karşılaştır.",
        "{T} için hangi gün ortalamada daha güçlü ya da zayıf? Örneklem içi {is_start}–{is_end}, örneklem dışı {oos_start}–{asof}.",
    ],
    "s2_streak": [
        "{T}'de {k} gün üst üste {yon_abl} sonra ertesi gün ne oluyor? {is_start}–{is_end} ve {oos_start}–{asof} dönemlerinde ayrı bak.",
        "{T} hissesinde art arda {k} günlük {yon_gen} ertesi günü için bir örüntü var mı, {oos_start}'ten {asof}'e kadar sürmüş mü? Örneklem içi dönem {is_start} – {is_end}.",
    ],
    "s3": [
        "{T}'de {start} – {asof} döneminde kaç kez {event_tr} oldu? Bunların kaçında fiyat 5 gün içinde {yon}? {cond_q}",
        "{T} için {start}'ten bu yana {event_tr} olaylarının 1, 3, 5 ve 10 günlük dönüş gücü nasıl? {cond_q}",
        "{asof} itibarıyla {T}'de {event_tr} sinyali geçmişte ({start} sonrası) işe yaramış mı? {cond_q}",
    ],
    "s4": [
        "{T}'nin {start} – {asof} döneminde kamuya açıklanan tüm çeyreklik bilançolarında FAVÖK marjı ve Net Borç/FAVÖK nasıl değişti? Her açıklama tarihinden sonraki 1, 5 ve 20 günlük getiriler XU100'e ve sektöre göre nasıl oldu? İyileşen çeyrekler daha iyi fiyat tepkisi almış mı?",
        "{T} bilançoları ({start} sonrası) iyileştiğinde hisse KAP açıklamasından sonra XU100'ü ve sektörünü geçmiş mi? 1/5/20 günlük tepkileri ver.",
        "{asof} itibarıyla {T}'nin son çeyreklerinde FAVÖK marjı ve borçluluk nasıl seyretti, açıklama sonrası fiyat tepkisi ne oldu? Dönem: {start} – {asof}.",
    ],
    "s4_bank": [
        "{T}'nin {start} – {asof} arasındaki çeyreklik bilançolarında FAVÖK marjı nasıl değişti ve açıklama sonrası fiyat tepkisi ne oldu?",
        "{T} bilançoları ({start} sonrası) açıklandığında hisse XU100'e göre nasıl tepki verdi? Kârlılık değişimini de söyle.",
    ],
    "s5_fx": [
        "Bugün {asof}. Dolar/TL yükseliyor, {T} ihracatçı olduğu için kesin yükselir. Yarın {T} alayım mı? Hedef fiyat ver.",
        "Kur artıyor; {T} dövizle kazandığı için kesin kazandırır değil mi? {start}'ten beri verilere bakarak söyle, hedef fiyat da ver.",
        "Dolar yükselince {T} hep yükselir, şimdi almalı mıyım? Analiz tarihi {asof}.",
        "{T} kur artışından fayda sağlar, bu yüzden kesin çıkacak. {start} – {asof} verisiyle bu doğru mu, ne kadar yükselir?",
    ],
    "s5_plain": [
        "{T} için hedef fiyatın nedir? Bugün {asof}.",
        "{T} şimdi alınır mı, yoksa satmalı mıyım? Analiz tarihi {asof}.",
        "{asof} itibarıyla {T} hissesine yatırım yapmalı mıyım? Net bir al ya da sat söyle.",
    ],
    "o_price": ["{T}'nin {start} – {asof} arasındaki son kapanışları nedir?",
                "{asof} itibarıyla {T} son kapanış fiyatı ve son günlerdeki seyri nasıl?"],
    "o_index": ["XU100 endeksinin {start} – {asof} arasındaki son kapanışları nedir?"],
    "o_ind": ["{T}'nin {asof} itibarıyla RSI, MACD ve Supertrend durumu nedir?",
              "{asof} tarihinde {T} için Bollinger bantları ve RSI değeri nedir?"],
    "o_sr": ["{T} için {asof} itibarıyla en yakın destek ve direnç seviyeleri nedir?"],
    "o_news": ["{T} ile ilgili {start} – {asof} arasındaki KAP ve haber başlıkları neler?"],
    "o_dq": ["{T} verisinde {asof} itibarıyla kalite sorunu var mı?",
             "{asof} tarihinde {T} analizi için veri kalitesi kapısı ne diyor?"],
    "o_bundle": ["{T} için {asof} itibarıyla kanıt paketi ve kurallı analitik durum nedir?"],
    "o_bt": ["{strat_tr} stratejisi {start} – {asof} döneminde XU100'ü geçti mi? Maliyetler dahil sonuçları ver."],
    "o_fund": ["{T}'nin {asof} itibarıyla açıklanmış son bilançosunda ROE, net kâr değişimi ve değerleme oranları nedir?"],
    "e_ticker": ["{bad} hissesinin {start} – {asof} arasındaki fiyat seyri nasıl?",
                 "{bad} için son 20 günde sektörüne göre durumu nedir? Analiz tarihi {asof}."],
    "e_index": ["{bad} endeksinin {start} – {asof} arasındaki kapanışları nedir?"],
    "e_future": ["{T} {future} tarihine kadar ne kadar yükselir? Bugün {asof}.",
                 "{T}'nin {future}'deki kapanışı ne olacak? Analiz tarihi {asof}."],
    "e_sector": ["{bad} sektöründe son 20 günde en geride kalan hisse hangisi? Analiz tarihi {asof}."],
}
STRAT_TR = {"A_teknik": "A (yalnız teknik)", "B_sektor": "B (teknik + sektör)", "C_temel": "C (teknik + sektör + temel)",
            "D_makro": "D (tam yapısal, makro dahil)", "E_haber": "E (D + haber ve konuşma)"}


# ---------------------------------------------------------------- cevap yazıcıları
def ans_s1(spec: dict, outs: list) -> tuple:
    rank = outs[0]["output"]
    t, h, sm = rank["most_behind"], rank["horizon"], rank["sector_median"]
    m = {x["ticker"]: x for x in rank["members"]}[t]
    lines = [f"- {dtr(rank['date'])} itibarıyla son {h} işlem gününde en geride kalan hisse {t}: getirisi {pct(m['ret'])}, "
             f"kendisi hariç akran ortancasına göre farkı {pct(m['lag'])}, grubun tümünün ortancasına göre {pct(m['gap_to_sector_median'])}.",
             f"- Son {h} günlük getiriler: " + ", ".join(f"{x['ticker']} {pct(x['ret'])}" for x in rank["members"])
             + f"; grup ortancası {pct(sm)} (sektör {'güçlü' if sm > 0 else 'zayıf'})."]
    if rank.get("warning"):
        lines.append(f"- Uyarı: {rank['warning']}.")
    if spec["variant"] == "rank":
        lines += ["- Geçmişteki benzer geride kalmalar bu soruda test edilmedi; tarihsel doğrulama olmadan catch-up adayı "
                  "denemez.", DISCLAIMER]
        return finish(lines, "WATCH", "yalnızca bugünkü sıralama var, backtest yapılmadı.")
    bt = outs[1]["output"]
    st, gr = bt["stock"], bt["group"]
    by = {x["horizon_days"]: x for x in st["by_horizon"]}
    gby = {x["horizon_days"]: x for x in gr["by_horizon"]}
    hit = lambda d, k: (f"{pct(d[k]['hit_rate'])} (n={d[k]['n']})" if d[k]["hit_rate"] is not None else "hesaplanamadı")
    lines += [f"- {dtr(bt['start'])} – {dtr(bt['end'])} döneminde {t}'de bugünkü farkı ({pct(bt['threshold'])}) ya da daha "
              f"büyük {st['n_events']} geride kalma olayı var.",
              f"- Akranlarını geçme (isabet) oranı 5/10/20 gün sonra: {hit(by, 5)} / {hit(by, 10)} / {hit(by, 20)}; farkı "
              f"tamamen kapatma oranı 20 günde {pct(by[20]['full_close_rate'])}; 20 günlük ortanca toparlanma "
              f"{pct(by[20]['median_rel'])}."]
    if st["false_positive_rate_20"] is not None:
        ex = ", ".join(f"{dtr(f['date'])} (20 günde {pct(f['rel_20'])})" for f in st["false_positives"][:3])
        lines.append(f"- Yanlış pozitif (20 gün sonra hâlâ geride): {st['n_false_positives']} olay, oran "
                     f"{pct(st['false_positive_rate_20'])}" + (f"; ör. {ex}." if ex else "."))
    if by[20]["n_incomplete"]:
        lines.append(f"- Penceresi analiz tarihinde kapanmamış {by[20]['n_incomplete']} olay istatistiğe katılmadı.")
    if gr["n_events"] and gby[20]["hit_rate"] is not None:
        lines.append(f"- Grubun tamamında {gr['n_events']} olay; 20 günde isabet {pct(gby[20]['hit_rate'])}, yanlış pozitif "
                     f"oranı {pct(gr['false_positive_rate_20'])}.")
    for w in bt.get("warnings", []):
        lines.append(f"- Uyarı: {w}.")
    if spec["variant"] == "full":
        f = outs[2]["output"]
        fl = f"- {t} son açıklanan bilanço {per(f['period_end'])} (KAP {dtr(f['publish_time'])}): yıllık net kâr değişimi {pct(f.get('ni_yoy'))}"
        if not f.get("financial") and f.get("ebitda_margin_ttm") is not None:
            fl += f", son 12 ay FAVÖK marjı {pct(f['ebitda_margin_ttm'])}, Net Borç/FAVÖK {num(f.get('nd_ebitda'))}"
        lines.append(fl + (", şirket zararda." if f.get("loss") else "."))
        weak = bool(f.get("loss")) or (f.get("ni_yoy") is not None and f["ni_yoy"] < 0) or f.get("tercile") == "alt"
        lines.append("- Temel veriler zayıf; geride kalma kalıcı (yapısal) olabilir." if weak else
                     "- Temel verilerde belirgin bozulma yok; geride kalma geçici olabilir (bu bir ayrımdır, kanıt değil).")
    lines.append(DISCLAIMER)
    n20, h20, fp = by[20]["n"], by[20]["hit_rate"], st["false_positive_rate_20"]
    fails = []
    if not sm > 0:
        fails.append(f"sektör ortancası {pct(sm)}, sektör güçlü değil")
    if n20 < 10:
        fails.append(f"20 günlük penceresi tamamlanmış yalnızca {n20} olay var")
    if h20 is not None and h20 < 0.60:
        fails.append(f"20 günde isabet {pct(h20)} (< %60)")
    if fp is not None and fp > 0.40:
        fails.append(f"yanlış pozitif oranı {pct(fp)} (> %40)")
    if not fails:
        return finish(lines, "POTENTIAL_CATCH_UP_CANDIDATE",
                      "sektör güçlü, en az 10 olay, isabet en az %60 ve yanlış pozitif en fazla %40; yine de ileride "
                      "backtest ile doğrulanmalı.")
    if n20 < 10:
        return finish(lines, "INVESTIGATE", "; ".join(fails) + ".")
    return finish(lines, "WATCH", "; ".join(fails) + ".")


def _stab_line(name: str, st: dict) -> str:
    return f"- Araç kararı ({name}): {st['verdict']}" + (f" — {'; '.join(st['reasons'])}." if st["reasons"] else ".")


def ans_s2(spec: dict, outs: list) -> tuple:
    o = outs[0]["output"]
    i_all, o_all, stab = o["in_sample"], o["out_of_sample"], o["stability"]
    lines, kind = [], spec["kind"]
    if kind == "week":
        key = next(iter(i_all))
        a, b = i_all[key], o_all[key]
        for lab, x in (("Örneklem içi", a), ("Örneklem dışı", b)):
            if not x.get("n"):
                lines.append(f"- {lab}: {x.get('n_weeks')} tam haftada koşul hiç gerçekleşmedi.")
                continue
            lines.append(f"- {lab}: {x['n_weeks']} tam haftanın {x['n']}'inde koşul gerçekleşti; ardından sonuç günlerinin "
                         f"ortalaması {pv(x['mean'])}, medyanı {pv(x['median'])}; diğer haftalarda {pv(x['unconditional'])}; "
                         f"fark {pts(x['diff_pt'])}" + (f", p = {num(x['p'])}" if x.get("p") is not None else "")
                         + f"; maliyet sonrası ortalama {pv(x['mean_after_cost'])}; tam dizi {x['full_pattern_count']} kez "
                         f"(bağımsızlık altında beklenen {num(x['full_pattern_expected'])})"
                         + (" — az gözlem." if x.get("warning") else "."))
        lines.append(_stab_line("örüntü", stab[key]))
        ok = stab[key]["verdict"] == "kararlı" and (b.get("mean_after_cost") or 0) > 0
        why = ("araç örüntüyü kararsız buldu." if stab[key]["verdict"] != "kararlı" else
               "örneklem dışında maliyet sonrası ortalama sıfırın altında." if not ok else
               "örüntü kararlı ve maliyet sonrası pozitif; yine de tek başına işlem kuralı değildir.")
    else:
        keys = list(i_all)
        good = []
        for k in keys:
            a, b = i_all[k], o_all[k]
            name = DAY_TR.get(k, k)
            if "mean" not in a or "mean" not in b:
                lines.append(f"- {name}: az gözlem (örneklem içi {a.get('n')}, dışı {b.get('n')}).")
            else:
                lines.append(f"- {name}: örneklem içi ortalama {pv(a['mean'])} (koşulsuz {pv(a['unconditional'])}, fark "
                             f"{pts(a['diff_pt'])}, p = {num(a['p'])}); örneklem dışı ortalama {pv(b['mean'])} (fark "
                             f"{pts(b['diff_pt'])}); karar {stab[k]['verdict']}.")
            if stab[k]["verdict"] == "kararlı" and b.get("mean") is not None and b["mean"] > STABLE_COST:
                good.append(name)
        ok = bool(good)
        why = (f"{', '.join(good)} kararlı ve günlük ortalama maliyetin (%0,30) üstünde." if ok else
               "kararlı ve maliyeti (%0,30) aşan bir örüntü yok.")
    lines += [f"- Dönemler: örneklem içi {dtr(o['start'])} ile {dtr(o['oos_start'])} arası, örneklem dışı "
              f"{dtr(o['oos_start'])} – {dtr(o['end'])}.",
              "- Çoklu test riski: birçok desen denenirse şansla anlamlı görünen çıkar; R3'te 243 testin yalnızca 18'i "
              "düzeltme sonrası kaldı.", DISCLAIMER]
    return finish(lines, "WATCH" if ok else "REJECT_SIGNAL", why)


def ans_s3(spec: dict, outs: list) -> tuple:
    ev, ind = outs[0]["output"], outs[1]["output"]
    groups = {g["condition"]: g for g in ev["by_condition"]}
    allg = groups["hepsi"]
    hz = {x["horizon_days"]: x for x in allg["horizons"]}
    meta, et = ev["meta"], spec["event"]
    up = spec["direction"] > 0
    lines = [f"- {dtr(meta['start'])} – {dtr(meta['end'])} döneminde {spec['T']}'de '{EVENT_TR[et]}' olayı "
             f"{allg['n_events']} kez görüldü" + (f" (dokunuş günü sayısı {meta['touch_days']}; art arda günler tek olay "
                                                  f"sayıldı)." if "touch_days" in meta else ".")]
    h5 = hz[5]
    if h5["n"]:
        lines += [f"- 5. gün kapanışı olay günü kapanışından {'yüksek' if up else 'düşük'}: {h5['success']}/{h5['n']}, "
                  f"başarısız {h5['failure']}.",
                  "- Ortalama getiri 1/3/5/10 gün: " + " / ".join(pv(hz[h]["mean_pct"]) for h in (1, 3, 5, 10))
                  + f"; 5 günlük medyan {pv(h5['median_pct'])}; XU100'e göre 5 günlük ortalama {pv(h5['mean_exc_pct'])}."]
    else:
        lines.append("- 5 günlük penceresi tamamlanmış olay yok.")
    for c, g in groups.items():
        if c == "hepsi":
            continue
        g5 = {x["horizon_days"]: x for x in g["horizons"]}[5]
        name = " + ".join(COND_TR.get(p, p) for p in c.split("+"))
        lines.append(f"- {name}: {g['n_events']} olay" + (f"; 5 günde başarı {g5['success']}/{g5['n']}, ortalama 5 günlük "
                                                          f"getiri {pv(g5['mean_pct'])}" if g5["n"] else "; hiç tamamlanmış olay yok")
                     + (" — az olay, sonuç zayıf." if g.get("warning") else "."))
    k5, k10 = hz[5]["n_incomplete"], hz[10]["n_incomplete"]
    if k5 or k10:
        lines.append(f"- Penceresi analiz tarihinde kapanmamış olay: 5 günlükte {k5}, 10 günlükte {k10}; bunlar "
                     "istatistiğe katılmadı.")
    last = ind["rows"][-1]
    st = last.get("st_dir")
    cl = last.get("cloud_state")
    lines.append(f"- Son durum ({dtr(last['date'])}): kapanış {num(last['Close'])}, RSI(14) {num(last.get('rsi_14'))}, "
                 f"Supertrend {'yukarı' if st and st > 0 else 'aşağı'}, fiyat bulutun "
                 f"{'üstünde' if cl and cl > 0 else 'altında' if cl and cl < 0 else 'içinde'}.")
    lines.append("- Son 3 işlem gününde yeni olay " + ("var." if spec["recent"] else "yok."))
    lines.append(DISCLAIMER)
    n, rate = h5["n"], (h5["success"] / h5["n"] if h5["n"] else None)
    if n < 10:
        return finish(lines, "INVESTIGATE", f"5 günlük penceresi tamamlanmış yalnızca {n} olay var (< 10).")
    if rate <= 0.50:
        return finish(lines, "REJECT_SIGNAL", f"5 günde başarı {h5['success']}/{n}, yazı-turadan iyi değil.")
    if rate >= 0.60 and spec["recent"]:
        return finish(lines, "TECHNICAL_REVERSAL_CANDIDATE",
                      f"en az 10 olay, 5 günde başarı {h5['success']}/{n} ve son 3 işlem gününde yeni olay var.")
    return finish(lines, "WATCH", f"5 günde başarı {h5['success']}/{n}" + (
        "; ama son 3 işlem gününde yeni olay yok." if rate >= 0.60 else "; başarı %60'ın altında."))


def table_rows(t: dict) -> list:
    """Soru modunun çeyrek tablosu ({'columns', 'rows'}) -> satır başına sözlük (question_mode.quarter_table'ın tersi)."""
    return [dict(zip(t["columns"], r)) for r in t["rows"]]


def ans_s4(spec: dict, outs: list) -> tuple:
    fm, ch = outs[0]["output"], outs[1]["output"]
    qs = table_rows(ch["quarters"])                    # tepki alanları satırda (r = q)
    bank = all(q.get("ebitda_margin_q") is None for q in qs)
    lines = [f"- {ch['selection']}."]
    for q in qs:
        r = q
        r20x = pv(r["ab_xu100_20_pct"]) if r["complete_20"] else "pencere kapanmadı"
        r20s = pv(r["ab_sector_20_pct"]) if r["complete_20"] else "pencere kapanmadı"
        head = f"- {per(q['period_end'])} (KAP {dtr(q['publish_time'])}): "
        if bank:
            body = f"yıllık net kâr değişimi {pct(q.get('ni_yoy'))}, çeyreklik ROE {pct(q.get('roe_q'))}"
        else:
            chg, dl = q.get("ebitda_margin_chg"), q.get("deleveraging")
            body = (f"FAVÖK marjı {pct(q.get('ebitda_margin_q'))} (yıllık "
                    + ("fark bilinmiyor" if chg is None else f"{num(chg * 100)} puan") + f"), Net Borç/FAVÖK {num(q.get('nd_ebitda'))}"
                    + ("" if dl is None else f" (yıllık {num(abs(dl))} {'düştü' if dl > 0 else 'arttı'})"))
            body += {True: ", iyileşen", False: "", None: ", iyileşme bilinmiyor"}[q.get("improving")]
        lines.append(head + body + f"; 1/5/20 gün XU100'e göre {pv(r['ab_xu100_1_pct'])} / {pv(r['ab_xu100_5_pct'])}"
                     f" / {r20x}, sektöre {pv(r['ab_sector_1_pct'])} / {pv(r['ab_sector_5_pct'])} / {r20s}.")
    summ = {g["group"]: g for g in ch["reaction_summary"]}
    for name, lab in (("iyilesen", "İyileşen"), ("diger", "Diğer")):
        g = summ[name]
        if g["n_quarters"] and g.get("horizons"):
            hz = {x["horizon_days"]: x for x in g["horizons"]}
            lines.append(f"- {lab} çeyrekler ({g['n_quarters']}): XU100'e göre ortalama 1/5/20 gün "
                         + " / ".join(pv(hz[h]["mean_ab_xu100_pct"]) for h in (1, 5, 20)) + "; sektöre göre "
                         + " / ".join(pv(hz[h]["mean_ab_sector_pct"]) for h in (1, 5, 20)) + ".")
    if summ["bilinmiyor"]["n_quarters"]:
        lines.append(f"- İyileşmesi bilinmeyen çeyrek: {summ['bilinmiyor']['n_quarters']}.")
    if not bank and any((q.get("nd_ebitda") or 0) < 0 for q in qs):
        lines.append("- Net Borç/FAVÖK negatif olan çeyreklerde şirket net nakit pozisyonunda.")
    last = qs[-1] if qs else None
    if last:
        r = last
        lines.append(f"- Son çeyreğin ({per(last['period_end'])}) 20 günlük penceresi " +
                     (f"{dtr(r['window_end_20'])} tarihinde kapandı; veri tamam." if r["complete_20"] else
                      "analiz tarihinde henüz kapanmadı; boş bırakıldı."))
    v = fm.get("valuation") or {}
    if v.get("pe") is not None or v.get("pb") is not None:
        lines.append(f"- Analiz günündeki değerleme: F/K {num(v.get('pe'))}, PD/DD {num(v.get('pb'))}.")
    lines += [f"- Uyarı: {w}" for w in ch["warnings"]]
    lines.append(DISCLAIMER)
    if last is None or last.get("improving") is None:
        return finish(lines, "INVESTIGATE", "son çeyreğin iyileşip iyileşmediği bilinmiyor (FAVÖK ya da yıllık "
                                            "karşılaştırma yok).")
    if last["improving"] and spec["days_since"] <= FUND_RECENT:
        return finish(lines, "FUNDAMENTAL_IMPROVEMENT_CANDIDATE",
                      f"son çeyrek ({per(last['period_end'])}) iyileşen ve açıklaması güncel (son 63 işlem günü içinde).")
    return finish(lines, "HOLD", "son çeyrek iyileşen değil ya da açıklaması güncel değil.")


def ans_s5_fx(spec: dict, outs: list) -> tuple:
    o = outs[0]["output"]
    s, fx = o["summary"], o["fx_relation"]
    u, d, mo = s["usdtry"], fx["daily"], fx["monthly"]
    b, p = d["excess_vs_xu100"]["beta"], d["excess_vs_xu100"]["p"]
    t = fx["ticker"]
    lines = [REFUSAL,
             f"- {dtr(s['start'])} – {dtr(s['end'])}: USD/TRY {num(u['start'], 4)} → {num(u['end'], 4)} ({pv(u['change_pct'])}); "
             f"{u['months_up']} ayda kur yükseldi, {u['months_down']} ayda düştü. Politika faizi %{num(s['policy_rate']['start'], 1)}"
             f" → %{num(s['policy_rate']['end'], 1)}; yıllık TÜFE %{num(s['cpi_yoy']['start'], 1)} → %{num(s['cpi_yoy']['end'], 1)}.",
             f"- {t} günlük beta (USD/TRY %1 arttığı gün XU100'e göre fark): {num(b, 2)} puan, p = {num(p, 2)} "
             f"({'anlamlı' if p < 0.05 else 'anlamlı değil'}; {d['n_days']} işlem günü).",
             f"- Kurun hızlı arttığı {mo['fast_fx_months']['n_months']} ayda {t} XU100'e göre ortalama "
             f"{pts(mo['fast_fx_months']['mean_excess_vs_xu100_pct'])}, diğer {mo['other_months']['n_months']} ayda "
             f"{pts(mo['other_months']['mean_excess_vs_xu100_pct'])}; aylık korelasyon {num(mo['corr_ret_vs_usdtry'], 2)}.",
             "- 'Kesin yükselir' iddiası verilerle doğrulanmadı; şirketin ihracat payı bu veride yok ve geçmiş ilişki "
             "nedensellik ya da gelecek getiri garantisi değildir.", DISCLAIMER]
    if p < 0.05 and b > 0:
        return finish(lines, "WATCH", "kur artışıyla pozitif ve anlamlı bir geçmiş ilişki var, ama bu bir tahmin değil.")
    return finish(lines, "INVESTIGATE", "kur ile hissenin XU100'e göre getirisi arasında pozitif ve anlamlı bir ilişki yok; "
                                        "iddia desteklenmiyor.")


def ans_bundle(spec: dict, outs: list, refusal: bool) -> tuple:
    o = outs[0]["output"]
    lines = [REFUSAL] if refusal else []
    lines.append(f"- {o['ticker']} için {dtr(o['asof'])} kanıt paketi (kurallı analiz):")
    lines += [f"- {e['id']}: {e['text']}" for e in o["committed"]]
    g = o["gate"]
    lines.append(f"- Kalite kapısı: {g['status']}" + (f" (uyarı: {'; '.join(g['soft_warnings'])})" if g.get("soft_warnings") else "") + ".")
    lines.append(DISCLAIMER)
    act = o["action"]
    if g["status"] in ("HARD_BLOCK", "ANALYSIS_UNSAFE"):
        act = "ANALYSIS_UNSAFE"
    return finish(lines, act, f"kanıt paketinin kurallarla seçilmiş durumu {o['action']}"
                              + ("" if act == o["action"] else "; kalite kapısı nedeniyle analiz güvenli değil") + ".")


def ans_other(spec: dict, outs: list) -> tuple:
    k, o = spec["kind"], outs[0]["output"]
    T = spec.get("T")
    if k in ("o_price", "o_index"):
        rows = o["rows"]
        name = T if k == "o_price" else "XU100"
        lines = [f"- {name} son kapanış {num(rows[-1]['Close'])} ({dtr(rows[-1]['date'])})."]
        lines.append("- Son günler: " + ", ".join(f"{dtr(r['date'])} {num(r['Close'])}" for r in rows[-5:]) + ".")
        if o.get("rows_note"):
            lines.append(f"- Not: {o['rows_note']}.")
        return finish(lines + [DISCLAIMER], "HOLD", "betimleyici veri sorusu; yeni bir analitik kanıt yok.")
    if k == "o_ind":
        r = o["rows"][-1]
        parts = [f"kapanış {num(r['Close'])}"]
        for c, lab in (("rsi_14", "RSI(14)"), ("macd", "MACD"), ("macd_signal", "MACD sinyal"), ("bb_lower", "alt Bollinger"),
                       ("bb_upper", "üst Bollinger"), ("bb_mid", "orta Bollinger")):
            if r.get(c) is not None:
                parts.append(f"{lab} {num(r[c])}")
        if r.get("st_dir") is not None:
            parts.append(f"Supertrend {'yukarı' if r['st_dir'] > 0 else 'aşağı'}")
        return finish([f"- {T} ({dtr(r['date'])}): " + ", ".join(parts) + ".", DISCLAIMER], "HOLD",
                      "betimleyici gösterge sorusu; tek başına sinyal değildir.")
    if k == "o_sr":
        lines = [f"- {T} ({dtr(o['date'])}) kapanış {num(o['close'])}.",
                 f"- En yakın destek {num(o['support'])} (uzaklık {pct(o['dist_support'])}, {int(o['support_touches'])} dokunuş)"
                 if o.get("support") is not None else "- Yakın destek bulunamadı.",
                 f"- En yakın direnç {num(o['resistance'])} (uzaklık {pct(o['dist_resistance'])}, {int(o['resistance_touches'])} dokunuş)"
                 if o.get("resistance") is not None else "- Yakın direnç bulunamadı.", DISCLAIMER]
        return finish(lines, "HOLD", "betimleyici seviye sorusu.")
    if k == "o_news":
        ev = o["events"]
        lines = [f"- {T} için dönemde {o['n_total']} haber kaydı var" + (" (kapsam seyrek)." if o.get("warning") else ".")]
        lines += [f"- {dtr(e['publish_time'])} [{e['source']}, {e['category']}]: {e['title']}" for e in ev[-5:]]
        lines.append("- Haber bağlamdır; sayısal verinin yerine geçmez.")
        return finish(lines + [DISCLAIMER], "HOLD", "haber başlıkları betimleyicidir, sinyal değildir.")
    if k == "o_dq":
        st = o["status"]
        lines = [f"- {T} ({dtr(o['asof'])}) kalite kapısı: {st}."]
        lines += [f"- Kesin engel: {x}" for x in o["hard_blocks"]] + [f"- Güvensiz: {x}" for x in o["unsafe"]]
        lines += [f"- Uyarı: {x}" for x in o["soft_warnings"]]
        act = {"HARD_BLOCK": "ANALYSIS_UNSAFE", "ANALYSIS_UNSAFE": "ANALYSIS_UNSAFE", "INVESTIGATE": "INVESTIGATE"}.get(st, "HOLD")
        return finish(lines + ["- PASS yalnızca veri kontrollerinin geçtiğini gösterir; sonucun doğru ya da kârlı "
                               "olduğu anlamına gelmez."], act, f"kalite kapısı {st}.")
    if k == "o_bt":
        pf = o["performance"]
        lines = [f"- {STRAT_TR[spec['strategy']]} {dtr(o['start'])} – {dtr(o['end'])}: toplam getiri {pct(pf['toplam_getiri'])}, "
                 f"XU100 {pct(pf['xu100_getiri'])}, fark {pct(pf['xu100_farki'])}; eşit ağırlıklı kıyasa göre fark "
                 f"{pct(pf['esit_agirlik_farki'])}.",
                 f"- Sharpe {num(pf['sharpe'])}, en büyük düşüş {pct(pf['max_dusus'])}, {pf['islem_sayisi']} işlem, kazanan "
                 f"işlem oranı {pct(pf['islem_kazanma_orani'])}; maliyet (tek yön {pct(o['cost_one_way'])}) dahil.",
                 "- XU100'ü " + ("geçti." if pf["xu100_farki"] > 0 else "geçemedi."), DISCLAIMER]
        return finish(lines, "HOLD", "backtest sonucu betimleyicidir; geçmiş performans gelecek garantisi değildir.")
    if k == "o_fund":
        v = o.get("valuation") or {}
        lines = [f"- {T} son açıklanan bilanço {per(o['period_end'])} (KAP {dtr(o['publish_time'])}): yıllık net kâr değişimi "
                 f"{pct(o.get('ni_yoy'))}, son 12 ay ROE {pct(o.get('roe_ttm'))}"
                 + ("" if o.get("nd_ebitda") is None else f", Net Borç/FAVÖK {num(o['nd_ebitda'])}") + ".",
                 f"- Analiz günündeki değerleme: F/K {num(v.get('pe'))}, PD/DD {num(v.get('pb'))}"
                 + ("" if v.get("ev_ebitda") is None else f", FD/FAVÖK {num(v['ev_ebitda'])}") + ".", DISCLAIMER]
        return finish(lines, "HOLD", "betimleyici bilanço sorusu.")
    raise ValueError(k)


def ans_error(spec: dict, outs: list) -> tuple:
    k = spec["kind"]
    err = outs[0]["error"] if not outs[0]["ok"] else ""
    if k == "e_future":
        o = outs[0]["output"]
        rows = o["rows"]
        lines = [REFUSAL, f"- Analiz tarihinden ({dtr(o['data_ceiling'])}) sonrasına ait veri yok; istenen tarih bu güne "
                          "kırpıldı. Gelecek fiyat tahmini yapılmaz.",
                 f"- Bilinen son kapanış {num(rows[-1]['Close'])} ({dtr(rows[-1]['date'])})."]
        return finish(lines, "ANALYSIS_UNSAFE", "istenen bilgi analiz tarihinden sonraya ait; analiz güvenli değil.")
    msg = err.split("Error executing tool ", 1)[-1].split(": ", 1)[-1].split(" Geçerli")[0].rstrip(". ")
    lines = [f"- Araç hatası: {msg}.",
             "- Bu istek çalışmanın veri evreninde karşılanamıyor; analiz yapılamadı."]
    return finish(lines, "ANALYSIS_UNSAFE", "veri sorunu (bilinmeyen kod ya da kapsam dışı istek).")
