"""Adım 14: Harness deneyleri A-E (PDF Bölüm 30). Aynı vakalar beş sistem sürümünde, her biri iki kez.

  A  Ham LLM            : veriler metin olarak verilir, araç yok, eylemi LLM seçer
  B  LLM + araçlar      : 15 MCP aracının hepsi açık, sıra yok, eylemi LLM seçer
  C  + durum makinesi   : sıralı durumlar ve durum başına izinli araçlar; kanıt/kapı yok, eylemi LLM seçer
  D  + kanıt + kapı     : committed evidence ve engelleyici kurallar; eylemi kurallar seçer
  E  + hafıza + insan   : geçmiş kararlar, insan incelemesi, karar günlüğü ve tekrar oynatma

Referans: deterministik kural motorunun (evidence.run_analysis) sonucu. Metrikler otomatik hesaplanır (elle değil).
Her (vaka, sürüm, tekrar) sonucu reports/experiments/raw.jsonl'e yazılır; komut yarıda kalırsa kaldığı yerden sürer.
2. tekrar ayrı önbellekle çalışır (önbellekten okumaz) -> tekrarlanabilirlik ölçülür.

Çalıştırma:
  python -m src.harness.experiments --mock          # anahtarsız deneme (sahte model)
  python -m src.harness.experiments                 # Gemini; E sürümünün 1. tekrarında karar sorulur
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from src.agents.llm import CACHE, MockLLM, assistant_message, tool_schemas
from src.agents.prompts import FIX_PROMPT, system_prompt
from src.agents.validate import check_numbers, classify, numbers_in
from src.harness.evidence import ACTIONS, CANDIDATES, run_analysis
from src.harness.state_machine import Harness, _short, parse_answer
from src.mcp_server import tools as T
from src.mcp_server.server import build_server

CASES = [("ASELS", "2026-11-06", "bilanço adayı"), ("TURSG", "2026-11-06", "şok + olumsuz haber"),
         ("PETKM", "2026-11-06", "çelişen teknik sinyaller"), ("KRDMD", "2026-11-06", "şok nedeniyle ret"),
         ("ASELS", "2023-02-15", "deprem penceresi, analiz güvensiz"), ("YOKBOYLE", "2026-11-06", "bilinmeyen hisse")]
VERSIONS = ["A", "B", "C", "D", "E"]
OUT = Path("reports/experiments")
RAW = OUT / "raw.jsonl"
EXP_DB = OUT / "experiment_log.sqlite"
MAX_B_ROUNDS = 8
DATE_ARGS = {"get_stock_history": "end", "get_index_history": "end", "calculate_indicators": "end",
             "detect_reversal_events": "end", "find_support_resistance": "date", "rank_sector_relative_strength": "date",
             "test_weekday_pattern": "end", "get_fundamental_metrics": "asof", "calculate_fundamental_change": "asof",
             "get_macro_context": "end", "get_news_events": "asof", "run_backtest": "end",
             "check_data_quality": "asof", "build_evidence_bundle": "asof"}
KEYWORDS = {"get_news_events": ["KAP", "duyuru", "sözleşme", "haber"], "check_data_quality": ["şok"],
            "get_macro_context": ["faiz", "TÜFE", "makro"], "run_backtest": ["backtest", "strateji"]}


def question(t: str, d: str) -> str:
    return f"{t} hissesi için {d} kapanışı itibarıyla eğitim amaçlı analiz durumu nedir? İzinli eylemlerden birini seç."


# ---------- A: ham veri metni ----------
def raw_context(t: str, d: str) -> dict:
    """A sürümüne metin olarak verilen ham veri (biz point-in-time hazırlarız; gösterge/kanıt hesaplanmaz)."""
    day = pd.Timestamp(d)
    start90, start30 = str((day - pd.Timedelta(days=90)).date()), str((day - pd.Timedelta(days=30)).date())

    def safe(fn, **kw):
        try:
            return fn(**kw)
        except ToolError as e:
            return dict(hata=str(e))
    px = safe(T.get_stock_history, ticker=t, start=start90, end=d)
    return dict(fiyat=[{k: r[k] for k in ("date", "Close", "Volume")} for r in px.get("rows", [])] or px,
                xu100=[{k: r[k] for k in ("date", "Close")} for r in safe(T.get_index_history, start=start90,
                                                                         end=d).get("rows", [])],
                bilanco=safe(T.get_fundamental_metrics, ticker=t, asof=d),
                makro=(safe(T.get_macro_context, start=start30, end=d).get("rows") or [None])[-1],
                haberler=[{k: n[k] for k in ("event_day", "title", "summary")}
                          for n in safe(T.get_news_events, ticker=t, start=start30, asof=d).get("events", [])][-15:]
                if t != "YOKBOYLE" else dict(hata="bilinmeyen hisse"))


def run_A(llm, t: str, d: str) -> dict:
    ctx = raw_context(t, d)
    msgs = [dict(role="system", content=system_prompt("raw")),
            dict(role="user", content=f"{question(t, d)}\n\nHAM VERİ:\n{_short(ctx, 12000)}")]
    t0 = time.time()
    r = llm.chat(msgs)
    action, text = parse_answer(r["content"])
    return dict(action=action, explanation=text, check=check_numbers(text, ctx), tool_calls=0, invalid_calls=0,
                tool_errors=0, future_calls=0, llm_calls=1, latency=round(time.time() - t0, 2), backtest_run=False)


# ---------- B: serbest araç kullanımı ----------
def future_call(name: str, args: dict, d: str) -> bool:
    """Araç, analiz gününden sonraki veriyi döndürecek şekilde mi çağrıldı? (tarih yok = son gün varsayılır)"""
    key = DATE_ARGS.get(name)
    if key is None:
        return False
    last = T._index("XU100").index[-1]
    v = args.get(key)
    try:
        when = pd.Timestamp(v) if v else last
    except (ValueError, TypeError):
        return False
    return when.normalize() > pd.Timestamp(d)


async def run_B(llm, t: str, d: str) -> dict:
    calls, invalid, errors, future, outputs = [], 0, 0, 0, []
    t0, n_llm = time.time(), 0
    async with Client(build_server()) as c:
        tools = (await c.list_tools()).tools
        names = {x.name for x in tools}
        schemas = tool_schemas(tools)
        msgs = [dict(role="system", content=system_prompt("tools_free")), dict(role="user", content=question(t, d))]
        r = llm.chat(msgs, schemas)
        n_llm += 1
        for _ in range(MAX_B_ROUNDS):
            if not r["tool_calls"]:
                break
            msgs.append(assistant_message(r))
            for x in r["tool_calls"]:
                try:
                    args = json.loads(x["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = None
                if x["name"] not in names or args is None:
                    invalid += 1
                    out = dict(error=f"geçersiz araç çağrısı: {x['name']}")
                else:
                    future += future_call(x["name"], args, d)
                    res = await c.call_tool(x["name"], args)
                    text = "".join(getattr(k, "text", "") for k in res.content)
                    errors += bool(res.is_error)
                    out = dict(error=text[:400]) if res.is_error else json.loads(text)
                calls.append(dict(tool=x["name"], args=args))
                outputs.append(out)
                msgs.append(dict(role="tool", tool_call_id=x["id"], content=_short(out)))
            r = llm.chat(msgs, schemas)
            n_llm += 1
        if r["tool_calls"]:                                          # tur sınırı doldu: son yanıtı zorla iste
            msgs.append(dict(role="user", content="Araç turu bitti. Şimdi yalnızca JSON yanıtını ver."))
            r = llm.chat(msgs)
            n_llm += 1
    action, text = parse_answer(r["content"])
    return dict(action=action, explanation=text, check=check_numbers(text, *outputs), tool_calls=len(calls),
                invalid_calls=invalid, tool_errors=errors, future_calls=future, llm_calls=n_llm,
                latency=round(time.time() - t0, 2), backtest_run=any(x["tool"] == "run_backtest" for x in calls),
                tools_used=[x["tool"] for x in calls])


# ---------- C, D, E: durum makinesi ----------
def run_harness_version(llm, t: str, d: str, version: str, review: dict = None) -> dict:
    h = Harness(llm, review=review, db=EXP_DB, version=version)
    ctx = asyncio.run(h.run(t, d))
    tr = h.trace
    o = [x for x in tr["llm"] if x["agent"] == "orchestrator"][-1]
    final = dict(n_numbers=o["numbers"], unsupported=o.get("unsupported_after_retry", o["unsupported"]),
                 rounding=o.get("rounding_after_retry", o.get("rounding", [])))
    n = final["n_numbers"]
    final["rate"] = len(final["unsupported"]) / n if n else 0.0
    final["rounding_rate"] = len(final["rounding"]) / n if n else 0.0
    s = ctx["summary"]
    return dict(action=ctx.get("action"), explanation=ctx.get("explanation", ""), check=final,
                tool_calls=s["tool_calls"], invalid_calls=s["invalid_tool_calls"], tool_errors=s["tool_errors"],
                future_calls=0, llm_calls=s["llm_calls"], latency=s["latency_total"],
                backtest_run=any(c["tool"] == "run_backtest" for c in tr["tool_calls"]),
                all_agents_hallucination=s["hallucination_rate"], all_agents_rounding=s["rounding_rate"],
                states=tr["states"], review=ctx.get("review"), analysis_id=ctx.get("analysis_id"))


# ---------- puanlama ----------
def reference(t: str, d: str) -> dict:
    r = run_analysis(t, d)
    return dict(action=r["action"], gate=r["gate"]["status"], conditions=r.get("conditions") or {},
                committed=[dict(id=e["id"], text=e["text"], tool=e["tool"]) for e in r["committed"]])


def unsupported_conclusion(action: str, ref: dict) -> bool:
    """Eylem deterministik kanıtla desteklenmiyor mu? (geçersiz eylem, engelde aday, şok/olumsuz haberde aday,
    koşulu olmayan aday etiketi)"""
    if action not in ACTIONS:
        return True
    if ref["gate"] in ("HARD_BLOCK", "ANALYSIS_UNSAFE"):
        return action != "ANALYSIS_UNSAFE"
    c = ref["conditions"]
    if action in CANDIDATES and (c.get("shock") or c.get("negative_news")):
        return True
    need = {"POTENTIAL_CATCH_UP_CANDIDATE": "catch_up", "FUNDAMENTAL_IMPROVEMENT_CANDIDATE": "fundamental",
            "TECHNICAL_REVERSAL_CANDIDATE": "technical"}
    return action in need and not c.get(need[action])


def evidence_completeness(text: str, ref: dict):
    """Referans kanıtların açıklamada karşılanan oranı: kimlik atfı, kanıttaki bir sayı ya da anahtar kelime."""
    items = ref["committed"]
    if not items:
        return None
    low = (text or "").lower()
    hit = 0
    for e in items:
        if f"({e['id']})" in text or f"{e['id']}," in text or f"{e['id']})" in text:
            hit += 1
            continue
        nums = [(v, dd) for v, dd, _ in numbers_in(e["text"]) if not (dd == 0 and abs(v) < 10)]
        said = [(v, dd) for v, dd, _ in numbers_in(text)]
        if any(classify(v, dd, [n for n, _ in nums]) != "uydurma" for v, dd in said) and nums:
            hit += 1
            continue
        if any(k.lower() in low for k in KEYWORDS.get(e["tool"], [])):
            hit += 1
    return hit / len(items)


def score(res: dict, ref: dict) -> dict:
    ch = res["check"]
    return dict(action=res["action"], ref_action=ref["action"], valid_action=res["action"] in ACTIONS,
                matches_reference=res["action"] == ref["action"],
                unsupported_conclusion=unsupported_conclusion(res["action"], ref),
                numbers=ch["n_numbers"], fabricated=len(ch["unsupported"]), rounding=len(ch["rounding"]),
                evidence_completeness=evidence_completeness(res["explanation"], ref),
                tool_calls=res["tool_calls"], invalid_calls=res["invalid_calls"], tool_errors=res["tool_errors"],
                future_calls=res["future_calls"], llm_calls=res["llm_calls"], latency=res["latency"],
                backtest_run=res["backtest_run"])


# ---------- akış ----------
def done_keys() -> set:
    if not RAW.exists():
        return set()
    return {(r["ticker"], r["asof"], r["version"], r["run"]) for r in map(json.loads, RAW.read_text().splitlines())}


def stored_review(t: str, d: str):
    if not RAW.exists():
        return None
    for r in map(json.loads, RAW.read_text().splitlines()):
        if (r["ticker"], r["asof"], r["version"], r["run"]) == (t, d, "E", 1) and r.get("review"):
            return r["review"]
    return None


def make_llm(mock: bool, run: int):
    if mock:
        return MockLLM()
    from src.agents.llm import GeminiLLM
    return GeminiLLM(cache=CACHE / f"exp_run{run}")         # 2. tekrar 1.'nin önbelleğini görmez


def run_all(mock: bool = False, versions=VERSIONS, runs=(1, 2), cases=CASES) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    for run in runs:
        llm = make_llm(mock, run)
        for t, d, label in cases:
            ref = reference(t, d)
            for v in versions:
                if (t, d, v, run) in done:
                    continue
                print(f"[tekrar {run}] {t} {d} ({label}) sürüm {v} ...", flush=True)
                if v == "A":
                    res = run_A(llm, t, d)
                elif v == "B":
                    res = asyncio.run(run_B(llm, t, d))
                else:
                    review = None
                    if v == "E":
                        review = dict(decision="ACCEPT", reason="sahte model denemesi") if mock else (
                            stored_review(t, d) if run > 1 else None)  # 1. tekrarda insan karar verir
                    res = run_harness_version(llm, t, d, v, review)
                row = dict(ticker=t, asof=d, label=label, version=v, run=run, model=llm.name, **score(res, ref),
                           explanation=res["explanation"], unsupported=res["check"]["unsupported"],
                           rounding_list=res["check"]["rounding"], review=res.get("review"),
                           all_agents_hallucination=res.get("all_agents_hallucination"),
                           tools_used=res.get("tools_used"))
                with RAW.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                print(f"   -> {row['action']} (referans {ref['action']}), uydurma {row['fabricated']}/{row['numbers']}, "
                      f"araç {row['tool_calls']}, geçersiz {row['invalid_calls']}, geleceğe bakan {row['future_calls']}",
                      flush=True)
    return summarize()


def summarize() -> pd.DataFrame:
    rows = pd.DataFrame(map(json.loads, RAW.read_text().splitlines()))
    rep = (rows.pivot_table(index=["ticker", "asof", "version"], columns="run", values="action", aggfunc="first")
           .dropna())
    repro = (rep.nunique(axis=1) == 1).groupby(level="version").mean() if len(rep.columns) > 1 else None
    g = rows.groupby("version")
    out = pd.DataFrame(dict(
        vaka_tekrar=g.size(),
        gecerli_eylem=g["valid_action"].mean(),
        referansla_ayni=g["matches_reference"].mean(),
        desteksiz_sonuc=g["unsupported_conclusion"].mean(),
        sayisal_halusinasyon=g.apply(lambda x: x["fabricated"].sum() / max(x["numbers"].sum(), 1)),
        yuvarlama_hatasi=g.apply(lambda x: x["rounding"].sum() / max(x["numbers"].sum(), 1)),
        kanit_tamligi=g["evidence_completeness"].mean(),
        gecersiz_arac_orani=g.apply(lambda x: (x["invalid_calls"].sum() + x["tool_errors"].sum())
                                    / max(x["tool_calls"].sum() + x["invalid_calls"].sum(), 1)),
        gelecege_bakan_cagri=g["future_calls"].sum(),
        backtest_yapildi=g["backtest_run"].mean(),
        arac_cagrisi=g["tool_calls"].mean(),
        llm_cagrisi=g["llm_calls"].mean(),
        sure_sn=g["latency"].mean(),
    ))
    if repro is not None:
        out.insert(4, "tekrarlanabilirlik", repro)
    out.to_csv("reports/harness_experiments.csv")
    rows.drop(columns=["explanation"]).to_csv(OUT / "per_case.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Harness deneyleri A-E")
    ap.add_argument("--mock", action="store_true", help="sahte model (anahtar kullanılmaz)")
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--runs", default="1,2")
    ap.add_argument("--fresh", action="store_true", help="önceki deney sonuçlarını sil ve baştan başla")
    a = ap.parse_args()
    if a.fresh:
        for f in (RAW, EXP_DB):
            f.unlink(missing_ok=True)
    pd.set_option("display.width", 250)
    res = run_all(a.mock, a.versions.split(","), tuple(int(x) for x in a.runs.split(",")))
    print("\nSONUÇ (reports/harness_experiments.csv):")
    print(res.round(3).to_string())
