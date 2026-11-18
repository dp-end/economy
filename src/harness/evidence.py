"""Kanıt paketi (PDF Bölüm 25) ve kurallı eylem seçimi (Bölüm 17).

Bir (hisse, analiz günü) için:
  observed   : o gün kapanışta bilinen tüm özellikler (gözlenen bağlam, 30+ alan)
  history    : kullanılan kalıbın yalnızca analiz gününden ÖNCE tamamlanmış örneklerinden istatistik
  backtest   : ilgili strateji varyantının analiz gününe kadarki sonucu
  action     : sınırlı eylem kümesinden kurallarla seçilen etiket (LLM seçmez, Adım 13'te açıklar)
  committed  : sonucu gerçekten destekleyen kanıtlar (E1, E2, ...)
Her şey yalnızca analiz günü kapanışında bilinen veriden üretilir; kapı (quality_gate) bunu denetler.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.backtest.engine import benchmarks, calculate_performance, load_returns, run_backtest
from src.backtest.strategies import WARMUP, generate_strategy_variants, macro_scale
from src.data.fundamentals import load_all
from src.data.macro import get_macro_context
from src.data.news_tags import load_news_events
from src.data.prices import get_index_history, get_stock_history
from src.data.universe import load_universe
from src.events.news_reaction import NEWS_WINDOW, news_signal_events
from src.harness.quality_gate import check_data_quality
from src.indicators.features import full_features

ACTIONS = ("HOLD", "WATCH", "INVESTIGATE", "POTENTIAL_CATCH_UP_CANDIDATE", "TECHNICAL_REVERSAL_CANDIDATE",
           "FUNDAMENTAL_IMPROVEMENT_CANDIDATE", "REJECT_SIGNAL", "ANALYSIS_UNSAFE")
CANDIDATES = ("POTENTIAL_CATCH_UP_CANDIDATE", "FUNDAMENTAL_IMPROVEMENT_CANDIDATE", "TECHNICAL_REVERSAL_CANDIDATE")
VARIANT_FOR = {"POTENTIAL_CATCH_UP_CANDIDATE": "B_sektor", "FUNDAMENTAL_IMPROVEMENT_CANDIDATE": "C_temel",
               "TECHNICAL_REVERSAL_CANDIDATE": "A_teknik"}
DEFAULT_VARIANT = "D_makro"
LAG_THR = -0.05              # Adım 5: 10 günde sektör ortancasının en az 5 puan gerisinde
SHOCK_LIMIT, SHOCK_DAYS = 0.095, 10
TECH_RECENT = 3              # son 3 işlem günündeki teknik olaylar "güncel"
FUND_RECENT = 63             # bilanço açıklamasından sonra ~3 ay "güncel"
TECH_H, SECTOR_H = 5, 10     # tarihsel başarı ufku (işlem günü)
SPEECH_DAYS = 30
TERCILE_TR = {"ust": "üst", "orta": "orta", "alt": "alt"}
WARNING_TEXT = "Yalnızca tarihsel kanıt; gelecek getiri garantisi ve yatırım tavsiyesi değildir."


# ---------- yardımcılar ----------
def trading_days() -> pd.DatetimeIndex:
    return get_index_history("XU100").index


def analysis_day(asof) -> pd.Timestamp:
    """Analiz günü işlem günü değilse önceki son işlem günü."""
    idx = trading_days()
    d = idx[idx <= pd.Timestamp(asof)]
    if len(d) == 0:
        raise ValueError(f"analiz günü veri başlangıcından önce: {asof}")
    return d[-1]


def shift_days(day: pd.Timestamp, n: int) -> pd.Timestamp:
    """İşlem günü takviminde n gün geri (n>0) git."""
    idx = trading_days()
    return idx[max(idx.get_loc(day) - n, 0)]


def _f(x):
    return None if x is None or pd.isna(x) else float(x)


@lru_cache(maxsize=1)
def _tables() -> dict:
    return dict(tech=pd.read_parquet("reports/technical_events.parquet"),
                sector=pd.read_parquet("reports/sector_events.parquet"),
                earnings=pd.read_parquet("reports/earnings_events.parquet"),
                news=load_news_events(), signals=news_signal_events(),
                speech=pd.read_parquet("reports/speech_features.parquet"))


@lru_cache(maxsize=1)
def _variants() -> dict:
    return generate_strategy_variants()


# ---------- gözlenen bağlam ----------
def observe(ticker: str, day: pd.Timestamp) -> tuple:
    """(observed, source_dates). source_dates: kullanılan her kaynağın en geç tarihi (kapı kontrol eder)."""
    t = _tables()
    uni = load_universe().set_index("ticker")
    f = full_features(ticker).loc[:day]
    r = f.iloc[-1]
    close = f["Close"]
    ret = close.pct_change()
    src = {"fiyat": f.index[-1]}

    price = dict(date=f.index[-1], close=_f(r["Close"]), ret_5d=_f(close.iloc[-1] / close.iloc[-6] - 1),
                 ret_20d=_f(close.iloc[-1] / close.iloc[-21] - 1), vol_20d=_f(r["vol_20d"]),
                 rel_xu_10=_f(r["rel_xu_10"]), rel_xu_20=_f(r["rel_xu_20"]),
                 max_abs_ret_250=_f(ret.iloc[-250:].abs().max()),
                 shock=bool((ret.iloc[-SHOCK_DAYS:].abs() >= SHOCK_LIMIT).any()))
    technical = dict(rsi_14=_f(r["rsi_14"]), macd_hist=_f(r["macd_hist"]), bb_pctb=_f(r["bb_pctb"]),
                     bb_bandwidth=_f(r["bb_bandwidth"]), st_dir=_f(r["st_dir"]), kama_slope=_f(r["kama_slope"]),
                     above_kama=bool(r["Close"] > r["kama"]), cloud_state=_f(r["cloud_state"]),
                     above_sma_50=bool(r["Close"] > r["sma_50"]), above_sma_200=bool(r["Close"] > r["sma_200"]),
                     vol_z_20=_f(r["vol_z_20"]), dist_support=_f(r["dist_support"]),
                     dist_resistance=_f(r["dist_resistance"]), support_touches=_f(r["support_touches"]),
                     resistance_touches=_f(r["resistance_touches"]))
    te = t["tech"]
    recent = te[(te["ticker"] == ticker) & (te["date"] <= day) & (te["date"] >= shift_days(day, TECH_RECENT - 1))]
    technical["recent_events"] = recent[["date", "event", "direction"]].to_dict("records")
    src["teknik_olay"] = te.loc[te["date"] <= day, "date"].max() if len(recent) else None

    group = uni.loc[ticker, "peer_group"]
    peers = [p for p in uni.index[uni["peer_group"] == group] if p != ticker]
    pr = pd.Series({p: get_stock_history(p)["Close"].loc[:day].pipe(lambda s: s.iloc[-1] / s.iloc[-11] - 1)
                    for p in peers})
    own10 = close.iloc[-1] / close.iloc[-11] - 1
    sector = dict(peer_group=group, peer_size=len(peers) + 1, sector_lag_5=_f(r["sector_lag_5"]),
                  sector_lag_10=_f(r["sector_lag_10"]), sector_lag_20=_f(r["sector_lag_20"]),
                  sector_median_10=_f(pr.median()) if len(pr) else None,
                  rank_in_group=int((pr > own10).sum() + 1), own_ret_10=_f(own10))

    ev = t["earnings"]
    vis = ev[(ev["ticker"] == ticker) & (ev["event_day"] <= day)].sort_values("event_day")
    fundamentals = dict(period_end=None)
    if len(vis):
        e = vis.iloc[-1]
        fundamentals = dict(period_end=e["period_end"], event_day=e["event_day"], ni_yoy=_f(e["ni_yoy"]),
                            tercile=e["tercile"] if isinstance(e["tercile"], str) else None, loss=bool(e["loss"]),
                            days_since=int(trading_days().get_loc(day) - trading_days().get_loc(e["event_day"])),
                            financial=bool(e["financial"]))
        st = load_all()[ticker]
        st = st[st.index == e["period_end"]]                   # Fintables tablosu dönem sonuna göre indeksli
        if len(st):
            s = st.iloc[-1]
            for k in ("roe_q", "leverage", "net_margin"):
                if k in s.index:
                    fundamentals[k] = _f(s[k])
        src["bilanco_event_day"] = e["event_day"]
        src["bilanco_donemi"] = e["period_end"]

    m = get_macro_context(day).iloc[-1]
    macro = {k: _f(m[k]) for k in ("policy_rate", "cpi_yoy", "real_rate", "usdtry", "usdtry_chg_20d",
                                   "fed_lower", "fed_upper")}
    macro["cpi_month"] = m["cpi_month"]
    macro["risk_half_weight"] = bool(macro_scale(trading_days()).loc[day] < 1)
    src["makro"] = day

    sig = t["signals"]
    win = sig[(sig["ticker"] == ticker) & (sig["event_day"] <= day) &
              (sig["event_day"] >= shift_days(day, NEWS_WINDOW - 1))]
    nw = t["news"]
    kap = nw[(nw["source"] == "KAP") & (nw["ticker"] == ticker) & (nw["event_day"] <= day)]
    news = dict(negative=win[win["direction"] == "olumsuz"][["event_day", "kind", "summary", "url"]].to_dict("records"),
                positive=win[win["direction"] == "olumlu"][["event_day", "kind", "summary", "url"]].to_dict("records"),
                kap_news_180d=int((kap["event_day"] >= day - pd.Timedelta(days=180)).sum()))
    src["haber"] = win["event_day"].max() if len(win) else None

    sp = t["speech"]
    sp = sp[(sp["publish_time"].dt.normalize() <= day) &
            (sp["publish_time"] >= day - pd.Timedelta(days=SPEECH_DAYS))]
    speech = dict(sources=[], warnings=[])
    if len(sp):
        from src.data.speech_features import get_speech_features
        for sid in sp["source_id"].unique():
            s = get_speech_features(sp, sid)
            speech["sources"].append(dict(source_id=sid, speaker=s["speaker"], publish_time=s["publish_time"],
                                          stance=s["stance"], verified_claims=int(s["claim_status"].get("VERIFIED", 0))))
            speech["warnings"] += s["warnings"]
        src["konusma"] = sp["publish_time"].max()
    return dict(price=price, technical=technical, sector=sector, fundamentals=fundamentals, macro=macro,
                news=news, speech=speech), src


# ---------- tarihsel istatistik (yalnızca analiz gününden önce TAMAMLANMIŞ olaylar) ----------
def tech_history(event: str, day: pd.Timestamp) -> dict:
    te = _tables()["tech"]
    cut = shift_days(day, TECH_H)                          # 5 günlük ileri pencere analiz gününe kadar kapanmış olmalı
    h = te[(te["event"] == event) & (te["date"] <= cut) & ~te["quake_window"]][f"exc_{TECH_H}"].dropna()
    return dict(n=int(len(h)), hit=_f((h > 0).mean()), mean_exc=_f(h.mean()), last_date=cut, horizon=TECH_H)


def catchup_history(day: pd.Timestamp) -> dict:
    se = _tables()["sector"]
    cut = shift_days(day, SECTOR_H)
    h = se[(se["config"] == "lag10_5pct") & (se["side"] == "laggard") & ~se["shock"] & ~se["quake_window"] &
           (se["date"] <= cut)][f"gap_{SECTOR_H}"].dropna()
    return dict(n=int(len(h)), hit=_f((h > 0).mean()), mean_gap=_f(h.mean()), last_date=cut, horizon=SECTOR_H)


def fundamental_history(day: pd.Timestamp) -> dict:
    ev = _tables()["earnings"]
    cut = shift_days(day, 5)
    g = ev[(ev["event_day"] <= cut) & ~ev["quake_window"]]
    top, bot = g[g["tercile"] == "ust"]["d0_4"].dropna(), g[g["tercile"] == "alt"]["d0_4"].dropna()
    return dict(n=int(len(top)), mean_top=_f(top.mean()), mean_bottom=_f(bot.mean()),
                spread=_f(top.mean() - bot.mean()), last_date=cut, horizon=5)


def backtest_until(variant: str, day: pd.Timestamp) -> dict:
    """Varyantın yalnızca analiz gününe kadarki sonucu (sonraki günler hesaba girmez)."""
    ret = load_returns().iloc[WARMUP:].loc[:day]
    if len(ret) < 60:
        return dict(variant=variant, end=None, uyari="backtest için yeterli gün yok")
    bt = run_backtest(_variants()[variant].reindex(ret.index), ret)
    bm = benchmarks(ret)
    p = calculate_performance(bt, bm["esit_agirlik"])
    xu = float((1 + bm["xu100"].fillna(0)).prod() - 1)
    return dict(variant=variant, start=ret.index[0], end=ret.index[-1], toplam_getiri=p["toplam_getiri"],
                esit_agirlik=p["kiyas_getiri"], fark=p["fark"], xu100=xu, xu100_fark=p["toplam_getiri"] - xu,
                sharpe=p["sharpe"], max_dusus=p["max_dusus"], islem_sayisi=p["islem_sayisi"], uyari=p["uyari"])


# ---------- eylem seçimi ----------
def conditions(obs: dict) -> dict:
    s, tch, fu, pr, nw = obs["sector"], obs["technical"], obs["fundamentals"], obs["price"], obs["news"]
    lag = s.get("sector_lag_10")
    laggard = lag is not None and lag <= LAG_THR
    sector_strong = (s.get("sector_median_10") or 0) > 0
    recovering = tch["above_kama"] or (tch["st_dir"] or 0) > 0
    weak_earn = fu.get("tercile") == "alt"
    return dict(
        laggard=laggard, sector_strong=sector_strong, recovering=recovering, weak_earnings=weak_earn,
        catch_up=laggard and sector_strong and recovering and not weak_earn,
        fundamental=fu.get("tercile") == "ust" and not fu.get("loss") and (fu.get("days_since") or 999) <= FUND_RECENT,
        technical=any(e["direction"] == 1 for e in tch["recent_events"])
        and not any(e["direction"] == -1 for e in tch["recent_events"]),          # çelişen ayı olayı varsa aday değil
        shock=pr["shock"], negative_news=len(nw["negative"]) > 0,
    )


def decide(cond: dict, gate_status: str) -> tuple:
    """(action, secondary). Sıra: kapı -> engelleyici -> aday -> izle/bekle."""
    cands = [a for a, k in zip(CANDIDATES, ("catch_up", "fundamental", "technical")) if cond[k]]
    blockers = cond["shock"] or cond["negative_news"]
    if gate_status in ("HARD_BLOCK", "ANALYSIS_UNSAFE"):
        return "ANALYSIS_UNSAFE", cands
    if cands and blockers:
        return "REJECT_SIGNAL", cands
    if blockers or gate_status == "INVESTIGATE":
        return "INVESTIGATE", cands
    if cands:
        return cands[0], cands[1:]
    if cond["laggard"]:                                     # geride ama toparlanma/sektör gücü yok: izle
        return "WATCH", []
    return "HOLD", []


# ---------- kanıt cümleleri ----------
def _pct(x, d=1):
    return "yok" if x is None else f"%{x * 100:.{d}f}"


def committed_evidence(action: str, secondary: list, cond: dict, b: dict) -> list:
    """Yalnızca sonucu destekleyen (ya da reddettiren) kanıtlar. Metinler sayılardan üretilir, model yazmaz."""
    o, h, bt = b["observed"], b["history"], b["backtest"]
    s, tch, fu, mac, nw = o["sector"], o["technical"], o["fundamentals"], o["macro"], o["news"]
    E = []

    def add(text, value, tool):
        E.append(dict(id=f"E{len(E) + 1}", text=text, value=value, tool=tool))

    focus = [action] + list(secondary)
    if "POTENTIAL_CATCH_UP_CANDIDATE" in focus or action == "WATCH":
        add(f"Hisse 10 günde sektör ortancasının {abs(s['sector_lag_10']) * 100:.1f} puan "
            f"{'gerisinde' if s['sector_lag_10'] < 0 else 'önünde'} ({s['peer_group']}, {s['peer_size']} hisse)",
            s["sector_lag_10"], "rank_sector_relative_strength")
        add(f"Sektör ortancası 10 günlük getiri {_pct(s['sector_median_10'])}"
            f" ({'güçlü' if cond['sector_strong'] else 'zayıf'})", s["sector_median_10"], "rank_sector_relative_strength")
        add(f"Fiyat {'KAMA üstünde' if tch['above_kama'] else 'KAMA altında'}, Supertrend "
            f"{'yukarı' if (tch['st_dir'] or 0) > 0 else 'aşağı'}", tch["st_dir"], "calculate_indicators")
        c = h.get("catch_up") or {}
        if c.get("n"):
            add(f"Şoksuz geride kalanlar geçmişte {c['horizon']} günde akranlarını %{c['hit'] * 100:.0f} oranında geçti "
                f"(n={c['n']}, ortalama fark {_pct(c['mean_gap'], 2)})", c["hit"], "rank_sector_relative_strength")
    if "FUNDAMENTAL_IMPROVEMENT_CANDIDATE" in focus or action in ("INVESTIGATE", "REJECT_SIGNAL"):
        if fu.get("period_end") is not None:
            add(f"Son açıklanan bilanço {pd.Timestamp(fu['period_end']).date()} ({pd.Timestamp(fu['event_day']).date()} "
                f"görünür): yıllık net kâr değişimi {_pct(fu['ni_yoy'])}; aynı çeyrekte 30 hisse içinde (sektör içi değil) "
                f"{TERCILE_TR.get(fu['tercile'], 'sırasız')} üçte birde"
                f"{', zarar' if fu['loss'] else ''}", fu["ni_yoy"], "calculate_fundamental_change")
        f_h = h.get("fundamental") or {}
        if "FUNDAMENTAL_IMPROVEMENT_CANDIDATE" in focus and f_h.get("n"):
            add(f"Geçmişte üst üçte bir bilançolar açıklama sonrası 5 günde alt üçte birden {_pct(f_h['spread'], 2)} "
                f"daha iyi (n={f_h['n']})", f_h["spread"], "get_fundamental_metrics")
    if "TECHNICAL_REVERSAL_CANDIDATE" in focus:
        for e in tch["recent_events"]:
            th = h.get(f"tech:{e['event']}") or {}
            hist = (f"; geçmişte XU100'ü {th['horizon']} günde %{th['hit'] * 100:.0f} oranında geçti (n={th['n']})"
                    if th.get("n") else "")
            add(f"{pd.Timestamp(e['date']).date()} teknik olay: {e['event']} "
                f"({'boğa' if e['direction'] == 1 else 'ayı'}){hist}", e["event"], "detect_reversal_events")
    if cond["shock"]:
        add("Son 10 işlem gününde ±%9,5 üzeri günlük hareket var (şok bayrağı)", True, "check_data_quality")
    for n in nw["negative"]:
        add(f"{pd.Timestamp(n['event_day']).date()} olumsuz KAP duyurusu ({n['kind']}): {n['summary']}", n["url"],
            "get_news_events")
    for n in nw["positive"]:
        add(f"{pd.Timestamp(n['event_day']).date()} olumlu KAP duyurusu ({n['kind']}): {n['summary']}", n["url"],
            "get_news_events")
    if not cond["negative_news"] and not cond["shock"] and action in CANDIDATES:
        add("Son 10 işlem gününde olumsuz KAP duyurusu ve şok yok", False, "get_news_events")
    add(f"Makro: politika faizi %{mac['policy_rate']:.1f}, yıllık TÜFE %{mac['cpi_yoy']:.1f} ({mac['cpi_month']}), "
        f"makro risk {'var (yarı ağırlık)' if mac['risk_half_weight'] else 'yok'}", mac["risk_half_weight"],
        "get_macro_context")
    if bt.get("end") is not None:
        add(f"Backtest {bt['variant']} ({pd.Timestamp(bt['start']).date()}-{pd.Timestamp(bt['end']).date()}, maliyet dahil): "
            f"{_pct(bt['toplam_getiri'], 0)}; eşit ağırlıklı al-tut {_pct(bt['esit_agirlik'], 0)}, "
            f"XU100 {_pct(bt['xu100'], 0)}, en büyük düşüş {_pct(bt['max_dusus'], 0)}", bt["fark"], "run_backtest")
    return E


# ---------- ana akış ----------
def build_evidence_bundle(ticker: str, asof) -> dict:
    """Gözlem + tarihsel istatistik + backtest. Eylem ve kapı sonucu run_analysis'te eklenir."""
    ticker = str(ticker).strip().upper()
    core = load_universe().query("is_core == 1")["ticker"].tolist()
    known = ticker in core
    day = analysis_day(asof)
    meta = dict(ticker=ticker, asof=day, requested_asof=pd.Timestamp(asof), known_ticker=known)
    if not known:
        return dict(meta=meta, observed={}, history={}, backtest={}, source_dates={})
    obs, src = observe(ticker, day)
    meta["sector"] = load_universe().set_index("ticker").loc[ticker, "sector_pdf"]
    cond = conditions(obs)
    hist = dict(catch_up=catchup_history(day) if cond["laggard"] else None,
                fundamental=fundamental_history(day) if obs["fundamentals"].get("tercile") else None)
    for e in obs["technical"]["recent_events"]:
        hist[f"tech:{e['event']}"] = tech_history(e["event"], day)
    for k, v in hist.items():
        if v:
            src[f"tarihsel:{k}"] = v["last_date"]
    return dict(meta=meta, observed=obs, history=hist, backtest={}, source_dates=src, conditions=cond)


def run_analysis(ticker: str, asof) -> dict:
    """Kanıt paketi -> ön kapı -> eylem -> backtest -> son kapı -> committed evidence."""
    b = build_evidence_bundle(ticker, asof)
    if not b["meta"]["known_ticker"]:
        gate = check_data_quality(b)
        b.update(gate=gate, action="ANALYSIS_UNSAFE", secondary=[], committed=[], warning=WARNING_TEXT)
        return b
    pre = check_data_quality(b)
    action, secondary = decide(b["conditions"], pre["status"])
    variant = VARIANT_FOR.get(action) or next((VARIANT_FOR[a] for a in secondary if a in VARIANT_FOR), DEFAULT_VARIANT)
    b["backtest"] = backtest_until(variant, b["meta"]["asof"])
    if b["backtest"].get("end") is not None:
        b["source_dates"]["backtest_bitis"] = b["backtest"]["end"]
    gate = check_data_quality(b)
    action, secondary = decide(b["conditions"], gate["status"])
    assert action in ACTIONS
    b.update(gate=gate, action=action, secondary=secondary, warning=WARNING_TEXT,
             committed=committed_evidence(action, secondary, b["conditions"], b))
    return b
