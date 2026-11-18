"""Durum makinesi (PDF Bölüm 23) + ajanlar (Bölüm 20). Araçlar MCP üzerinden çağrılır (Adım 12 sunucusu).

İlkeler:
  - Durum sırası sabittir; LLM durum seçemez, kalite/backtest durumlarını atlayamaz.
  - Her durum yalnızca STATE_TOOLS'taki araçları açar. Ajan izinsiz araç isterse çağrı engellenir ve sayılır.
  - Eylemi kurallar seçer (evidence.run_analysis); Orchestrator yalnızca committed evidence'a dayanarak açıklar.
  - Açıklamadaki her sayı kanıtla karşılaştırılır; desteksiz sayı varsa bir kez düzeltme istenir.
  - İnsan kararı olmadan kapanmaz; her şey karar günlüğüne ve iz tablosuna yazılır.

Çalıştırma:
  python -m src.harness.state_machine ASELS 2026-11-06 --mock                 # anahtarsız, sahte model
  python -m src.harness.state_machine ASELS 2026-11-06                        # Gemini (kullanıcı çalıştırır)
  ... --review ACCEPT --reason "gerekçe"   |  --review MODIFY --final WATCH --reason "..."
"""
import argparse
import asyncio
import json
import time

import pandas as pd
from mcp import Client

from src.agents.llm import MockLLM, tool_schemas
from src.agents.prompts import FIX_PROMPT, system_prompt
from src.agents.validate import check_numbers
from src.data.universe import load_universe
from src.harness.decision_log import (REVIEW_DECISIONS, log_decision, log_trace, past_cases, record_review)
from src.harness.evidence import ACTIONS, VARIANT_FOR, DEFAULT_VARIANT, run_analysis
from src.mcp_server.server import build_server

STATES = ["START", "SELECT_UNIVERSE", "LOAD_MARKET_DATA", "CHECK_DATA_QUALITY", "CALCULATE_TECHNICALS",
          "RUN_SECTOR_RELATIVE_ANALYSIS", "RUN_PATTERN_AND_REVERSAL_ANALYSIS", "LOAD_FUNDAMENTALS",
          "LOAD_MACRO_CONTEXT", "LOAD_NEWS_AND_SPEECH_CONTEXT", "BUILD_EVIDENCE", "PREVIEW_STRATEGIES",
          "RUN_BACKTEST", "RISK_GATE", "GENERATE_EXPLANATION", "HUMAN_REVIEW", "LOG_AND_CLOSE"]
STATE_TOOLS = {
    "START": [], "SELECT_UNIVERSE": [],
    "LOAD_MARKET_DATA": ["get_stock_history", "get_index_history"],
    "CHECK_DATA_QUALITY": ["check_data_quality"],
    "CALCULATE_TECHNICALS": ["calculate_indicators", "find_support_resistance"],
    "RUN_SECTOR_RELATIVE_ANALYSIS": ["rank_sector_relative_strength"],
    "RUN_PATTERN_AND_REVERSAL_ANALYSIS": ["detect_reversal_events", "test_weekday_pattern", "calculate_indicators",
                                          "find_support_resistance", "rank_sector_relative_strength"],
    "LOAD_FUNDAMENTALS": ["get_fundamental_metrics", "calculate_fundamental_change"],
    "LOAD_MACRO_CONTEXT": ["get_macro_context", "get_fundamental_metrics", "calculate_fundamental_change"],
    "LOAD_NEWS_AND_SPEECH_CONTEXT": ["get_news_events", "get_speech_features"],
    "BUILD_EVIDENCE": ["build_evidence_bundle"],
    "PREVIEW_STRATEGIES": ["run_backtest"],
    "RUN_BACKTEST": ["run_backtest"],
    "RISK_GATE": ["check_data_quality"],
    "GENERATE_EXPLANATION": [], "HUMAN_REVIEW": [], "LOG_AND_CLOSE": [],
}
AGENT_AT = {"RUN_PATTERN_AND_REVERSAL_ANALYSIS": "market", "LOAD_MACRO_CONTEXT": "fundamental_macro",
            "LOAD_NEWS_AND_SPEECH_CONTEXT": "news_speech", "PREVIEW_STRATEGIES": "strategy", "RISK_GATE": "risk",
            "GENERATE_EXPLANATION": "orchestrator"}
# Güvenli başarısızlık yolu: veri kesin engellenirse analiz durumları atlanır ama kapı, açıklama, insan ve kayıt atlanmaz.
SAFE_PATH = ["RISK_GATE", "GENERATE_EXPLANATION", "HUMAN_REVIEW", "LOG_AND_CLOSE"]
MANDATORY = {"CHECK_DATA_QUALITY", "RISK_GATE", "HUMAN_REVIEW", "LOG_AND_CLOSE"}
MAX_TOOL_ROUNDS = 2
CLAIM_TR = {"VERIFIED": "veriyle_dogrulanan", "CONSISTENT_WITH_HISTORY": "gecmis_degerle_tutarli",
            "CONTRADICTED": "veriyle_celisen", "NO_TOOL": "veri_araci_olmayan",
            "FORECAST_OR_TARGET": "tahmin_veya_politika_hedefi (fiyat hedefi degil)"}
PREVIEW_VARIANTS = ["A_teknik", "B_sektor", "C_temel", "D_makro", "E_haber"]


# Adım 14 deney sürümleri: C = araçlar + durum makinesi (kanıt/kapı/insan yok, eylemi LLM seçer),
# D = C + kanıt paketi + kalite kapısı (eylemi kurallar seçer), E = D + hafıza + insan incelemesi + kayıt (tam sistem).
VERSION_STATES = {
    "C": [s for s in STATES if s not in ("BUILD_EVIDENCE", "RISK_GATE", "HUMAN_REVIEW", "LOG_AND_CLOSE")],
    "D": [s for s in STATES if s not in ("HUMAN_REVIEW", "LOG_AND_CLOSE")],
    "E": STATES,
}
VERSION_MANDATORY = {"C": set(), "D": {"CHECK_DATA_QUALITY", "RISK_GATE"}, "E": MANDATORY}


def allowed_next(state: str, hard_blocked: bool, version: str = "E") -> str:
    """Bir sonraki durum. Kesin engelde güvenli yola geçilir (C sürümünde kapı olmadığı için geçilmez)."""
    seq = VERSION_STATES[version]
    if state == seq[-1]:
        return None
    if hard_blocked and version != "C":
        if STATES.index(state) < STATES.index("CHECK_DATA_QUALITY"):
            return "CHECK_DATA_QUALITY"                  # kalite kontrolü engelde bile atlanmaz
        if state == "CHECK_DATA_QUALITY" or state not in SAFE_PATH:
            return "RISK_GATE"
    return seq[seq.index(state) + 1]


def reason_hint(rec: dict) -> str:
    """İnsan incelemesi için gerekçe rehberi + bu vakanın kanıtlarından üretilmiş örnek (kararı insan verir)."""
    ids = [e["id"] for e in rec.get("committed", [])]
    g = rec.get("gate", {})
    weak = g.get("hard_blocks") or g.get("unsafe") or g.get("soft_warnings") or []
    bt = rec.get("backtest") or {}
    if not weak and bt.get("fark") is not None and bt["fark"] < 0:
        weak = ["strateji backtest'te eşit ağırlıklı al-tut'un gerisinde"]
    support = ", ".join(ids[:2]) or "kanıt yok"
    example = (f"'{support} sonucu destekliyor; ancak {weak[0] if weak else 'belirgin bir uyarı yok'}; "
               f"bu yüzden <kararın ve kısa nedeni>.'")
    return ("Gerekçe yazarken üç şeyi söyle: (1) kararını hangi kanıtlara dayandırdığını (E numarası), "
            "(2) sonucu zayıflatan uyarı, çelişki veya risk, (3) neden kabul ettiğin/değiştirdiğin/reddettiğin.\n"
            f"Örnek kalıp (kendi sözlerinle yaz): {example}")


def parse_answer(text: str) -> tuple:
    """LLM'in {"action": ..., "explanation": ...} yanıtı. Geçersizse eylem 'GECERSIZ' olur (sayılır, düzeltilmez)."""
    import re as _re
    m = _re.search(r"\{.*\}", text or "", _re.S)
    try:
        obj = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        obj = {}
    action = str(obj.get("action", "")).strip().upper()
    return (action if action in ACTIONS else "GECERSIZ"), str(obj.get("explanation") or text or "")


def describe_history(history) -> dict:
    """Tarihsel istatistikleri açık Türkçe alan adlarıyla verir (modelin alanları yanlış adlandırmaması için)."""
    out = {}
    for name, h in (history or {}).items():
        if not h:
            continue
        if name == "fundamental":
            out["bilanco_ucte_bir_tepkisi"] = dict(
                olay_sayisi=h["n"], ufuk_gun=h["horizon"], son_tamamlanan_olay=h["last_date"],
                ust_ucte_bir_ortalama_fazla_getiri=h["mean_top"], alt_ucte_bir_ortalama_fazla_getiri=h["mean_bottom"],
                ust_eksi_alt_fark=h["spread"])
        elif name == "catch_up":
            out["sektor_geride_kalan_toparlanmasi"] = dict(
                olay_sayisi=h["n"], ufuk_gun=h["horizon"], son_tamamlanan_olay=h["last_date"],
                akranlari_gecme_orani=h["hit"], ortalama_akran_farki=h["mean_gap"])
        else:
            out[f"teknik_olay_{name.split(':', 1)[1]}"] = dict(
                olay_sayisi=h["n"], ufuk_gun=h["horizon"], son_tamamlanan_olay=h["last_date"],
                xu100_gecme_orani=h["hit"], ortalama_xu100_fazla_getiri=h["mean_exc"])
    return out


def _short(x, n=4000) -> str:
    s = json.dumps(x, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + " …(kesildi)"


class Harness:
    def __init__(self, llm, review: dict = None, log: bool = True, db=None, version: str = "E"):
        self.llm, self.review, self.db, self.version = llm, review, db, version
        self.log = log and version == "E"                   # yalnızca tam sistem kayıt tutar
        self.memory = version == "E"

    # ---------- araç ----------
    async def tool(self, state: str, name: str, args: dict, by: str = "harness"):
        rec = dict(state=state, tool=name, args=args, by=by)
        if name not in STATE_TOOLS[state]:
            rec.update(allowed=False)
            self.trace["invalid_tool_calls"].append(rec)
            return dict(error=f"İZİNSİZ ARAÇ: '{name}' {state} durumunda açık değil. "
                              f"Açık araçlar: {STATE_TOOLS[state] or 'yok'}")
        t0 = time.time()
        r = await self.client.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in r.content)
        rec.update(allowed=True, is_error=bool(r.is_error), bytes=len(text), latency=round(time.time() - t0, 2))
        self.trace["tool_calls"].append(rec)
        if r.is_error:
            self.trace["tool_errors"].append(rec | dict(message=text[:300]))
            return dict(error=text[:500])
        return json.loads(text)

    # ---------- ajan ----------
    async def agent(self, state: str, agent: str, context: dict, task: str) -> str:
        allowed = [self.tools[n] for n in STATE_TOOLS[state]]
        schemas = tool_schemas(allowed) if allowed else None
        msgs = [dict(role="system", content=system_prompt(agent)),
                dict(role="user", content=f"{task}\n\nVERİ (araç çıktıları):\n{_short(context, 9000)}")]
        extra = []
        t0 = time.time()
        r = self.llm.chat(msgs, schemas)
        rounds = 0
        while r["tool_calls"] and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            msgs.append(dict(role="assistant", content=r["content"] or None,
                             tool_calls=[dict(id=c["id"], type="function",
                                              function=dict(name=c["name"], arguments=c["arguments"]))
                                         for c in r["tool_calls"]]))
            for c in r["tool_calls"]:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = await self.tool(state, c["name"], args, by=agent)
                extra.append(out)
                msgs.append(dict(role="tool", tool_call_id=c["id"], content=_short(out)))
            r = self.llm.chat(msgs, schemas)
        note = r["content"]
        chk = check_numbers(note, context, *extra)
        self.trace["llm"].append(dict(state=state, agent=agent, tool_rounds=rounds, cached=r.get("cached", False),
                                      latency=round(time.time() - t0, 2), numbers=chk["n_numbers"],
                                      unsupported=chk["unsupported"], rounding=chk["rounding"]))
        self.trace["notes"][agent] = note
        return note

    # ---------- ana akış ----------
    async def run(self, ticker: str, asof: str) -> dict:
        ticker = str(ticker).strip().upper()
        self.trace = dict(ticker=ticker, asof=str(asof), states=[], tool_calls=[], invalid_tool_calls=[],
                          tool_errors=[], llm=[], notes={})
        ctx, t_start = {}, time.time()
        async with Client(build_server()) as self.client:
            self.tools = {t.name: t for t in (await self.client.list_tools()).tools}
            state, hard = "START", False
            while state:
                self.trace["states"].append(state)
                hard = await self.step(state, ticker, asof, ctx) or hard
                state = allowed_next(state, hard, self.version)
        missing = VERSION_MANDATORY[self.version] - set(self.trace["states"])
        assert not missing, f"zorunlu durum atlandı: {missing}"
        self.trace["latency_total"] = round(time.time() - t_start, 2)
        ctx["summary"] = self.summary()
        return ctx

    async def step(self, state: str, t: str, asof: str, ctx: dict) -> bool:
        """Durumu çalıştırır. True dönerse veri kesin engellidir (güvenli yola geçilir)."""
        d = pd.Timestamp(asof)
        m30, y1 = str((d - pd.Timedelta(days=30)).date()), str((d - pd.Timedelta(days=365)).date())
        T = lambda name, **a: self.tool(state, name, a)
        if state == "SELECT_UNIVERSE":
            u = load_universe()
            ctx["universe"] = u[u["is_core"] == 1]["ticker"].tolist()
            ctx["peer_group"] = u.set_index("ticker")["peer_group"].get(t)
            return t not in ctx["universe"]
        if state == "LOAD_MARKET_DATA":
            ctx["prices"] = await T("get_stock_history", ticker=t, start=y1, end=asof)
            ctx["index"] = await T("get_index_history", start=y1, end=asof)
            return "error" in ctx["prices"]
        if state == "CHECK_DATA_QUALITY":
            ctx["gate_pre"] = await T("check_data_quality", ticker=t, asof=asof)
            return ctx["gate_pre"].get("status") == "HARD_BLOCK"
        if state == "CALCULATE_TECHNICALS":
            ctx["indicators"] = await T("calculate_indicators", ticker=t, start=m30, end=asof,
                                        indicators=["rsi_14", "macd_hist", "bb_pctb", "st_dir", "kama_slope",
                                                    "cloud_state", "vol_z_20", "ret_5d"])
            ctx["sr"] = await T("find_support_resistance", ticker=t, date=asof)
        elif state == "RUN_SECTOR_RELATIVE_ANALYSIS":
            ctx["sector"] = await T("rank_sector_relative_strength", sector=ctx["peer_group"], horizon=10, date=asof)
        elif state == "RUN_PATTERN_AND_REVERSAL_ANALYSIS":
            ctx["reversal"] = await T("detect_reversal_events", ticker=t, start=m30, end=asof)
            ctx["weekday"] = await T("test_weekday_pattern", ticker=t, pattern="weekday", end=asof)
            data = dict(indicators_last=ctx["indicators"].get("rows", [])[-3:], support_resistance=ctx["sr"],
                        sector=ctx["sector"], reversal_summary=ctx["reversal"].get("summary"),
                        recent_events=ctx["reversal"].get("events", [])[-8:], weekday=ctx["weekday"].get("results"))
            await self.agent(state, "market", data, f"{t} için {asof} itibarıyla piyasa/teknik/sektör durumunu yorumla.")
        elif state == "LOAD_FUNDAMENTALS":
            ctx["fund"] = await T("get_fundamental_metrics", ticker=t, asof=asof)
            ctx["fund_change"] = await T("calculate_fundamental_change", ticker=t, periods=4, asof=asof)
        elif state == "LOAD_MACRO_CONTEXT":
            ctx["macro"] = await T("get_macro_context", start=m30, end=asof)
            rows = ctx["macro"].get("rows", [])
            data = dict(fundamentals=ctx["fund"], quarterly_changes=ctx["fund_change"].get("quarters"),
                        macro_first=rows[:1], macro_last=rows[-1:], macro_meta=ctx["macro"].get("meta"))
            await self.agent(state, "fundamental_macro", data, f"{t} için bilanço ve makro bağlamı yorumla.")
        elif state == "LOAD_NEWS_AND_SPEECH_CONTEXT":
            ctx["news"] = await T("get_news_events", ticker=t, start=m30, asof=asof)
            srcs = (await T("get_speech_features")).get("sources", [])
            recent = [s["source_id"] for s in srcs if m30 <= s["publish_time"][:10] <= str(d.date())]
            ctx["speech"] = [await T("get_speech_features", source_id=s) for s in recent]
            speech = [dict(source_id=s["source_id"], speaker=s["speaker"], publish_time=s["publish_time"],
                           stance=s["stance"], warnings=s["warnings"],
                           iddia_turleri={CLAIM_TR.get(k, k): v for k, v in s["claim_status"].items()},
                           verified=[dict(metin=c["text"][:160], konusmadaki_deger=c["values"], veri_degeri=c["tool_value"],
                                          degisken=c["variable"]) for c in s["claims"] if c["status"] == "VERIFIED"])
                      for s in ctx["speech"]]
            news = [dict(source=n["source"], event_day=n["event_day"], category=n["category"],
                         title=n["title"], summary=n["summary"]) for n in ctx["news"].get("events", [])[-15:]]
            await self.agent(state, "news_speech", dict(news=news, speech=speech),
                             f"{t} için son 30 günün haber ve konuşma bağlamını özetle.")
        elif state == "BUILD_EVIDENCE":
            ctx["bundle"] = await T("build_evidence_bundle", ticker=t, asof=asof)
        elif state == "PREVIEW_STRATEGIES":
            full = {v: await T("run_backtest", strategy=v, end=asof) for v in PREVIEW_VARIANTS}
            mid = full[PREVIEW_VARIANTS[0]].get("start"), full[PREVIEW_VARIANTS[0]].get("end")
            half = str((pd.Timestamp(mid[0]) + (pd.Timestamp(mid[1]) - pd.Timestamp(mid[0])) / 2).date()) if all(mid) else None
            second = {v: await T("run_backtest", strategy=v, start=half, end=asof) for v in PREVIEW_VARIANTS} if half else {}
            ctx["previews"] = {v: r.get("performance") for v, r in full.items()}
            keep = ("toplam_getiri", "esit_agirlik_getiri", "esit_agirlik_farki", "xu100_getiri", "xu100_farki",
                    "sharpe", "max_dusus", "islem_sayisi", "uyari")
            pick = lambda r: {k: (r.get("performance") or {}).get(k) for k in keep}
            data = dict(aciklama="Getiriler oran: 2.41 = %241. esit_agirlik = 30 hissenin eşit ağırlıklı al-tut kıyası; "
                                 "xu100 = BIST 100 endeksi. *_farki = strateji eksi kıyas.",
                        tum_donem=dict(baslangic=mid[0], bitis=mid[1], **{v: pick(r) for v, r in full.items()}),
                        ikinci_yari=dict(baslangic=half, bitis=mid[1], **{v: pick(r) for v, r in second.items()}))
            await self.agent(state, "strategy", data, "Strateji varyantlarını aynı metriklerle karşılaştır.")
        elif state == "RUN_BACKTEST":
            action = ctx.get("bundle", {}).get("action")
            sec = ctx.get("bundle", {}).get("secondary") or []
            v = VARIANT_FOR.get(action) or next((VARIANT_FOR[a] for a in sec if a in VARIANT_FOR), DEFAULT_VARIANT)
            ctx["backtest"] = await T("run_backtest", strategy=v, end=asof)
        elif state == "RISK_GATE":
            ctx["gate"] = await T("check_data_quality", ticker=t, asof=asof)
            await self.agent(state, "risk", dict(gate=ctx["gate"],
                                                 tarihsel_istatistik=describe_history(ctx.get("bundle", {}).get("history"))),
                             "Kalite kapısı sonucunu ve uyarıları açıkla.")
        elif state == "GENERATE_EXPLANATION":
            await self.explain(t, asof, ctx)
        elif state == "HUMAN_REVIEW":
            ctx["review"] = self.ask_review(ctx)
        elif state == "LOG_AND_CLOSE":
            if self.log:
                self.close(t, asof, ctx)
        return False

    # ---------- açıklama + sayı kontrolü ----------
    async def explain(self, t: str, asof: str, ctx: dict):
        rec = run_analysis(t, asof)
        ctx["record"] = rec
        if self.version == "C":
            return await self.explain_free(t, asof, ctx)
        ctx["action"] = rec["action"]
        b = ctx.get("bundle") or {}
        if b and b.get("action") != rec["action"]:
            raise RuntimeError("araç çıktısı ile doğrudan analiz farklı eylem verdi")
        memory = []
        if self.memory:
            memory = (past_cases(t, db=self.db) if self.db else past_cases(t)).tail(3).to_dict("records")
        data = dict(ticker=t, asof=asof, action=rec["action"], secondary=rec["secondary"],
                    gate=dict(status=rec["gate"]["status"], warnings=rec["gate"]["soft_warnings"],
                              hard_blocks=rec["gate"]["hard_blocks"], unsafe=rec["gate"]["unsafe"]),
                    committed=[dict(id=e["id"], text=e["text"]) for e in rec["committed"]],
                    past_human_decisions=[dict(asof=m["asof"], action=m["action"], decision=m["decision"],
                                               reason=m["reason"]) for m in memory])
        task = (f"Kural tabanlı analiz durumu: {rec['action']}. Bu sonucu yalnızca committed kanıtlarla açıkla. "
                f"Geçmiş insan kararları varsa kısaca an.")
        msgs = [dict(role="system", content=system_prompt("orchestrator")),
                dict(role="user", content=f"{task}\n\nVERİ:\n{_short(data, 9000)}")]
        t0 = time.time()
        r = self.llm.chat(msgs)
        text, chk, retried = r["content"], check_numbers(r["content"], data), False
        if chk["unsupported"] or chk["rounding"]:
            retried = True
            msgs += [dict(role="assistant", content=text),
                     dict(role="user", content=FIX_PROMPT.format(bad=", ".join(chk["unsupported"] + chk["rounding"])))]
            r = self.llm.chat(msgs)
            text, chk2 = r["content"], check_numbers(r["content"], data)
        else:
            chk2 = chk
        verified = not chk2["unsupported"]
        self.trace["llm"].append(dict(state="GENERATE_EXPLANATION", agent="orchestrator", tool_rounds=0,
                                      cached=r.get("cached", False), latency=round(time.time() - t0, 2),
                                      numbers=chk["n_numbers"], unsupported=chk["unsupported"],
                                      rounding=chk["rounding"], retried=retried,
                                      unsupported_after_retry=chk2["unsupported"],
                                      rounding_after_retry=chk2["rounding"]))
        if not verified:
            text = "[DOĞRULANMAMIŞ AÇIKLAMA: kanıtta olmayan sayı içeriyor: " + ", ".join(chk2["unsupported"]) + "]\n" + text
        ctx["explanation"], ctx["explanation_verified"] = text, verified
        self.trace["notes"]["orchestrator"] = text

    async def explain_free(self, t: str, asof: str, ctx: dict):
        """C sürümü: kanıt paketi ve kapı yok; Orchestrator ajan notları ve araç özetlerinden eylemi kendisi seçer."""
        data = dict(ticker=t, asof=asof, allowed_actions=list(ACTIONS), agent_notes=self.trace["notes"],
                    quality_check=ctx.get("gate_pre"), fundamentals=ctx.get("fund"),
                    sector=ctx.get("sector"), reversal_summary=(ctx.get("reversal") or {}).get("summary"))
        msgs = [dict(role="system", content=system_prompt("orchestrator_free")),
                dict(role="user", content=f"{t} için {asof} itibarıyla analiz durumunu seç ve açıkla.\n\n"
                                          f"VERİ:\n{_short(data, 9000)}")]
        t0 = time.time()
        r = self.llm.chat(msgs)
        action, text = parse_answer(r["content"])
        chk, retried = check_numbers(text, data), False
        if chk["unsupported"] or chk["rounding"]:
            retried = True
            msgs += [dict(role="assistant", content=r["content"]),
                     dict(role="user", content=FIX_PROMPT.format(bad=", ".join(chk["unsupported"] + chk["rounding"]))
                          + " Yanıtı yine aynı JSON biçiminde ver.")]
            r = self.llm.chat(msgs)
            action, text = parse_answer(r["content"])
        chk2 = check_numbers(text, data)
        self.trace["llm"].append(dict(state="GENERATE_EXPLANATION", agent="orchestrator", tool_rounds=0,
                                      cached=r.get("cached", False), latency=round(time.time() - t0, 2),
                                      numbers=chk["n_numbers"], unsupported=chk["unsupported"],
                                      rounding=chk["rounding"], retried=retried,
                                      unsupported_after_retry=chk2["unsupported"],
                                      rounding_after_retry=chk2["rounding"]))
        ctx["action"], ctx["explanation"] = action, text
        ctx["explanation_verified"] = not chk2["unsupported"]
        self.trace["notes"]["orchestrator"] = text

    # ---------- insan ----------
    def ask_review(self, ctx: dict) -> dict:
        if self.review:
            return self.review
        rec = ctx["record"]
        print("\n" + "=" * 70)
        print(f"{rec['meta']['ticker']} {pd.Timestamp(rec['meta']['asof']).date()} -> {rec['action']} "
              f"(kapı {rec['gate']['status']})")
        for e in rec["committed"]:
            print(f"  {e['id']}: {e['text']}")
        print("\nAçıklama:\n" + ctx["explanation"])
        print("\n" + reason_hint(rec))
        while True:
            dec = input(f"\nKarar {REVIEW_DECISIONS}: ").strip().upper()
            if dec in REVIEW_DECISIONS:
                break
        final = None
        if dec == "MODIFY":
            while final not in ACTIONS:
                final = input(f"Yeni eylem {ACTIONS}: ").strip().upper()
        reason = ""
        while not reason.strip():
            reason = input("Gerekçe (zorunlu): ")
        return dict(decision=dec, reason=reason, final_action=final)

    # ---------- kayıt ----------
    def close(self, t: str, asof: str, ctx: dict):
        kw = dict(db=self.db) if self.db else {}
        tools_called = [c["tool"] for c in self.trace["tool_calls"]]
        aid = log_decision(ctx["record"], tools_called=tools_called, explanation=ctx["explanation"], **kw)
        rv = ctx["review"]
        record_review(aid, rv["decision"], rv["reason"], rv.get("final_action"), **kw)
        log_trace(aid, self.llm.name, self.summary(), self.trace, **kw)
        ctx["analysis_id"] = aid

    def summary(self) -> dict:
        tr = self.trace
        nums = sum(x["numbers"] for x in tr["llm"])
        bad = sum(len(x["unsupported"]) for x in tr["llm"])
        rnd = sum(len(x.get("rounding", [])) for x in tr["llm"])
        return dict(states=len(tr["states"]), tool_calls=len(tr["tool_calls"]),
                    invalid_tool_calls=len(tr["invalid_tool_calls"]), tool_errors=len(tr["tool_errors"]),
                    llm_calls=len(tr["llm"]), numbers=nums, unsupported_numbers=bad, rounding_errors=rnd,
                    hallucination_rate=round(bad / nums, 3) if nums else 0.0,
                    rounding_rate=round(rnd / nums, 3) if nums else 0.0,
                    latency_total=tr.get("latency_total"))


def run_harness(ticker: str, asof: str, llm=None, review: dict = None, log: bool = True, db=None) -> tuple:
    h = Harness(llm or MockLLM(), review=review, log=log, db=db)
    ctx = asyncio.run(h.run(ticker, asof))
    return ctx, h.trace


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BIST harness: durum makinesi + ajanlar")
    ap.add_argument("ticker")
    ap.add_argument("asof")
    ap.add_argument("--mock", action="store_true", help="Gemini yerine sahte model (anahtar kullanılmaz)")
    ap.add_argument("--review", choices=REVIEW_DECISIONS)
    ap.add_argument("--reason", default="")
    ap.add_argument("--final", choices=ACTIONS)
    a = ap.parse_args()
    if a.mock:
        llm = MockLLM()
    else:
        from src.agents.llm import GeminiLLM
        llm = GeminiLLM()
    review = dict(decision=a.review, reason=a.reason, final_action=a.final) if a.review else None
    ctx, trace = run_harness(a.ticker, a.asof, llm, review)
    print("\nDurumlar:", " -> ".join(trace["states"]))
    print("Özet:", json.dumps(ctx["summary"], ensure_ascii=False))
    for agent, note in trace["notes"].items():
        print(f"\n--- {agent} ---\n{note}")
    print(f"\nanalysis_id: {ctx.get('analysis_id')}")
