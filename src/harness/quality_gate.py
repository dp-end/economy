"""Risk ve veri kalitesi kapısı (PDF Bölüm 26). Kanıt paketini inceler, eylemi seçmez.

Kesin engel (HARD_BLOCK): gelecek bilgisi, bozuk/yanlış fiyat geçmişi, bilinmeyen hisse,
kritik dönem uyumsuzluğu, analiz anında bilinmeyen bilgiyle yapılmış backtest.
Güvensiz (ANALYSIS_UNSAFE): piyasanın kapalı olduğu deprem penceresi gibi analiz yapılamayan durumlar.
Yumuşak uyarılar sayısına göre PASS_WITH_WARNINGS ya da INVESTIGATE.
"""
import pandas as pd

GATE_OUTPUTS = ("PASS", "PASS_WITH_WARNINGS", "INVESTIGATE", "ANALYSIS_UNSAFE", "HARD_BLOCK")
MAX_PRICE_STALE_DAYS = 7          # son fiyat analiz gününden bu kadar eskiyse fiyat geçmişi bozuk sayılır
DAILY_LIMIT = 0.105               # BIST ±%10 limitinin üstünde günlük hareket = bozuk veri (bölünme vb.)
MAX_FUND_AGE_DAYS = 250           # son görünür bilanço dönemi bundan eskiyse kritik dönem uyumsuzluğu
STALE_FUND_DAYS = 150             # bundan eskiyse yumuşak uyarı
MIN_HIST_EVENTS = 20              # tarihsel istatistik bundan az olaya dayanıyorsa uyarı
MIN_NEWS_180D = 3                 # son 180 günde bundan az KAP haberi: haber kapsamı seyrek
MIN_PEERS = 3
MIN_HIT = 0.50                    # tarihsel isabet bunun altındaysa: kalıp geçmişte yazı-turadan iyi değil
INVESTIGATE_AT = 3                # bu kadar yumuşak uyarı birikirse INVESTIGATE
QUAKE = (pd.Timestamp("2023-02-06"), pd.Timestamp("2023-02-24"))


def _later(d, asof) -> bool:
    return d is not None and pd.notna(d) and pd.Timestamp(d).normalize() > pd.Timestamp(asof)


def check_data_quality(bundle: dict) -> dict:
    """{'status', 'hard_blocks', 'unsafe', 'soft_warnings'}. Paket alanları eksikse çökmez, engel üretir."""
    hard, unsafe, soft = [], [], []
    meta = bundle.get("meta", {})
    asof = meta.get("asof")
    obs = bundle.get("observed", {})

    if not meta.get("known_ticker", False):
        hard.append(f"bilinmeyen hisse kodu: {meta.get('ticker')}")
        return dict(status="HARD_BLOCK", hard_blocks=hard, unsafe=unsafe, soft_warnings=soft)

    # fiyat geçmişi
    price = obs.get("price", {})
    last = price.get("date")
    if last is None or pd.isna(price.get("close")):
        hard.append("fiyat yok ya da kapanış boş")
    elif (pd.Timestamp(asof) - pd.Timestamp(last)).days > MAX_PRICE_STALE_DAYS:
        hard.append(f"fiyat bayat: son fiyat {pd.Timestamp(last).date()}, analiz günü {pd.Timestamp(asof).date()}")
    if price.get("max_abs_ret_250", 0) > DAILY_LIMIT:
        hard.append(f"bozuk fiyat geçmişi: günlük hareket %{price['max_abs_ret_250'] * 100:.1f} (limit ±%10)")

    # gelecek bilgisi: kullanılan her kaynağın son tarihi analiz gününü geçemez
    for name, d in bundle.get("source_dates", {}).items():
        if _later(d, asof):
            hard.append(f"gelecek bilgisi: {name} = {pd.Timestamp(d).date()} > analiz günü {pd.Timestamp(asof).date()}")

    # bilanço dönemi
    fund = obs.get("fundamentals", {})
    if fund.get("period_end") is None:
        soft.append("görünür bilanço yok")
    else:
        pe = pd.Timestamp(fund["period_end"])
        age = (pd.Timestamp(asof) - pe).days
        if pe > pd.Timestamp(asof):
            hard.append(f"kritik dönem uyumsuzluğu: bilanço dönemi {pe.date()} analiz gününden sonra")
        elif age > MAX_FUND_AGE_DAYS:
            hard.append(f"kritik dönem uyumsuzluğu: son bilanço {age} gün eski ({pe.date()})")
        elif age > STALE_FUND_DAYS:
            soft.append(f"bilanço eski: {age} gün ({pe.date()})")

    # piyasa kapalı dönemi
    if QUAKE[0] <= pd.Timestamp(asof) <= QUAKE[1]:
        unsafe.append("deprem nedeniyle piyasa kapanış penceresi (Şubat 2023)")

    # yumuşak uyarılar
    macro = obs.get("macro", {})
    missing = [k for k in ("policy_rate", "cpi_yoy", "usdtry", "fed_upper") if pd.isna(macro.get(k))]
    if missing:
        soft.append(f"makro seri eksik: {', '.join(missing)}")
    if obs.get("news", {}).get("kap_news_180d", 0) < MIN_NEWS_180D:
        soft.append("haber kapsamı seyrek (son 180 günde < 3 KAP duyurusu)")
    for w in obs.get("speech", {}).get("warnings", []):
        if "örtüşme" in w or "güvenli" in w:
            soft.append(f"transkript kalitesi: {w}")
    for name, h in bundle.get("history", {}).items():
        if h and h.get("n", 0) < MIN_HIST_EVENTS:
            soft.append(f"az tarihsel olay: {name} n={h.get('n', 0)}")
        elif h and h.get("hit") is not None and h["hit"] < MIN_HIT:
            soft.append(f"tarihsel başarı yazı-turadan iyi değil: {name} %{h['hit'] * 100:.0f} (n={h['n']})")
    if obs.get("sector", {}).get("peer_size", MIN_PEERS) < MIN_PEERS:
        soft.append(f"sektör grubu küçük: {obs['sector'].get('peer_size')} hisse")
    bt = bundle.get("backtest", {})
    if bt.get("uyari"):
        soft.append(f"backtest uyarısı: {bt['uyari']}")

    if hard:
        status = "HARD_BLOCK"
    elif unsafe:
        status = "ANALYSIS_UNSAFE"
    elif len(soft) >= INVESTIGATE_AT:
        status = "INVESTIGATE"
    elif soft:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    return dict(status=status, hard_blocks=hard, unsafe=unsafe, soft_warnings=soft)
