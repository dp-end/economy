"""Soru modu (yerel model, 2. adım): serbest Türkçe soru -> model araç çağırır -> harness denetler -> cevap.

Hocanın duyurusu: model soruyu anlar, doğru MCP aracını doğru parametrelerle çağırır, yalnızca araç çıktılarına
dayanarak açıklar ve sonucu izinli eylem setinden verir. Hesabı araçlar yapar, kontrolü harness yapar.

Durumlar ve izinler (PDF §23: her durum yalnızca tanımlı araçları açar):
  START -> SET_CEILING      analiz tarihi tavanı konur (src/mcp_server/tools.data_ceiling); model değiştiremez
        -> CALL_TOOLS       15 okuma aracı açık, en fazla MAX_CALLS çağrı; geçersiz çağrının hatası modele döner
        -> ANSWER           araç yok; cevap 'ANALİTİK DURUM: <eylem>' satırıyla biter
        -> VALIDATE         sayı denetimi (src/agents/validate), tavsiye/hedef fiyat, eylem listesi, kural geçersiz
                            kılmaları; sorun varsa bir kez düzeltme isteği, yine varsa cevap gösterilmez
        -> LOG_AND_CLOSE    reports/question_log.sqlite (soru, araç çağrıları, modelin gördüğü çıktılar, cevap)
Kullanıcı kararları (24 Kasım 2026): 1a eylemi model seçer, harness denetler; 2a 15 aracın hepsi, en fazla 6 çağrı;
3a çıktılar modele kısaltılmış verilir (uzun satır listeleri atılır, 4 ondalık); 4a uydurma sayı -> bir düzeltme
denemesi, yine olursa ANALYSIS_UNSAFE.

Çalıştırma (önce yerel sunucu: venv-llm/bin/mlx_lm.server --model models/Qwen3-4B-Instruct-2507 --port 8080):
  python -m src.harness.question_mode --asof 2026-09-10 "Perakende sektöründe (BIMAS, MGROS, MAVI) ..."
"""
import argparse
import asyncio
import copy
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from mcp import Client

from src.agents.llm import LocalLLM, tool_schemas
from src.agents.validate import check_numbers
from src.harness.evidence import ACTIONS
from src.mcp_server.server import build_server
from src.mcp_server.tools import data_ceiling

QDB = Path("reports/question_log.sqlite")
MAX_CALLS = 6
KEEP_ROWS = 20
ROUND = 4
ACTION_HELP = {
    "HOLD": "durum değişmedi, yeni bir kanıt yok",
    "WATCH": "izlemeye değer ama güçlü/doğrulanmış kanıt yok",
    "INVESTIGATE": "kanıt karışık ya da yetersiz, daha fazla inceleme gerekli",
    "POTENTIAL_CATCH_UP_CANDIDATE": "sektörünün gerisinde ve geçmişte benzer geride kalmalar çoğunlukla kapanmış",
    "TECHNICAL_REVERSAL_CANDIDATE": "geçmişte tekrarlanmış ve yeterli sayıda olayla desteklenen teknik dönüş",
    "FUNDAMENTAL_IMPROVEMENT_CANDIDATE": "son açıklanan bilançolarda belirgin iyileşme",
    "REJECT_SIGNAL": "iddia ya da sinyal geçmiş verilerle desteklenmiyor",
    "ANALYSIS_UNSAFE": "veri ya da kalite sorunu, analiz güvenli değil",
}
SYSTEM = """Sen BIST 100 hisseleri için eğitim/araştırma amaçlı bir analiz asistanısın. Analiz tarihi: {asof}.

Kurallar:
1. Önce uygun aracı çağır; sayıları yalnızca araç çıktılarından al. Hesap yapma, sayı uydurma.
2. Sorudaki tarih aralığını ve hisseleri araç parametrelerine aynen yaz. Analiz tarihinden sonrası yoktur.
3. Yatırım tavsiyesi, al/sat önerisi ya da hedef fiyat verme. İstenirse bunu açıkça reddet; bunun yerine geçmiş
   verideki ilişkiyi araçla göster. "Kesin yükselir" gibi iddiaları doğrulanmamış iddia olarak belirt.
4. Penceresi analiz tarihinde kapanmamış olayları (complete=false, n_incomplete) istatistiğe katma ve bunu söyle.
   Olay sayısı azsa uyar. Nedensellik iddia etme. Birçok desen denendiyse çoklu test riskini belirt.
5. Oranları yüzde olarak yaz (0.1126 = %11,26). Sayıları araç çıktısındaki gibi doğru yuvarla.
6. Cevabı Türkçe, kısa ve maddeli yaz: önce bulgular, sonra sınırlamalar. Son satır tam olarak şöyle olsun:
   ANALİTİK DURUM: <eylem>
   İzinli eylemler: {actions}.
   Kanıt yetersizse INVESTIGATE, veri sorunu varsa ANALYSIS_UNSAFE seç.""".strip()
ACTION_RE = re.compile(r"ANAL[İIı]T[İIı]K\s+DURUM\s*:\s*\**\s*([A-Za-z_]+)", re.I)
ADVICE_RE = [re.compile(p, re.I) for p in (
    r"hedef\s+fiyat\w*\s*(?:[:=]|olarak|yaklaşık|civarı)?\s*[-~]?\s*\d",     # hedef fiyat + sayı
    r"\b(?:almanızı|satmanızı|tutmanızı|alımı|satışı)\s+(?:öneririm|tavsiye ederim|öneriyorum)",
    r"\b(?:almalısınız|satmalısınız|alım fırsatı|satın alın)\b",
)]
RETRY_TEXT = ("DÜZELT: Cevabın denetimden geçmedi: {problems}. Yalnızca araç çıktılarındaki sayıları kullan, "
              "tavsiye ya da hedef fiyat yazma ve cevabı yeniden yaz. Son satır 'ANALİTİK DURUM: <eylem>' olmalı.")
FORCE_TEXT = "Araç çağırma hakkın bitti. Elindeki araç çıktılarıyla cevabı şimdi yaz."
SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    qid TEXT PRIMARY KEY, created_at TEXT, question TEXT, asof TEXT, model TEXT, action TEXT, action_source TEXT,
    status TEXT, answer TEXT, answer_raw TEXT, tool_calls TEXT, outputs TEXT, validation TEXT,
    retries INTEGER, llm_calls INTEGER, latency REAL)
"""


# ---------------------------------------------------------------- modelin gördüğü kısaltılmış çıktı (karar 3a)
MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00$")


def _round(x):
    """Ondalıkları yuvarlar, saatsiz tarihleri ('2024-03-05 00:00:00') güne kısaltır."""
    if isinstance(x, float):
        return round(x, ROUND)
    if isinstance(x, str):
        m = MIDNIGHT.match(x)
        return m.group(1) if m else x
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _tail(o: dict, key: str, n: int = KEEP_ROWS, cols=None):
    rows = o.get(key) or []
    if cols:
        rows = [{c: r.get(c) for c in cols if c in r} for r in rows]
    if len(rows) > n:
        o[f"{key}_note"] = f"{len(rows)} satırın son {n} tanesi gösterildi"
        rows = rows[-n:]
    o[key] = rows


EVENT_COLS = ("date", "event", "direction", "rsi_state", "vol_recent", "st_state", "cloud_pos", "kama_state", "regime",
              "fwd_1", "fwd_3", "fwd_5", "fwd_10", "exc_5")
EVENT_COLS_SHORT = ("date", "rsi_state", "vol_recent", "st_state", "cloud_pos", "fwd_5")
QUARTER_COLS = ("period_end", "publish_time", "event_day", "ni_yoy", "roe_q", "ebitda_margin_q", "ebitda_margin_chg",
                "nd_ebitda", "deleveraging", "improving", "reaction")
REACTION_COLS = ("complete_20", "window_end_20", "ab_xu100_1_pct", "ab_xu100_5_pct", "ab_xu100_20_pct",
                 "ab_sector_1_pct", "ab_sector_5_pct", "ab_sector_20_pct")
NEWS_COLS = ("publish_time", "event_day", "source", "ticker", "category", "title")


def compact(name: str, out: dict) -> dict:
    """Araç çıktısının modele verilen hâli. Hiçbir yeni sayı üretmez: yalnızca uzun listeleri kısaltır, sütun seçer
    ve ondalıkları yuvarlar. Özetler, notlar ve uyarılar olduğu gibi kalır."""
    o = copy.deepcopy(out)
    if name in ("get_stock_history", "get_index_history"):
        _tail(o, "rows")
    elif name == "calculate_indicators":
        _tail(o, "rows", n=5)                    # son durum; geçmiş istatistik olay aracında
    elif name == "detect_reversal_events":
        if "by_condition" in o:                  # koşul özeti varken olay listesi örnek olarak kalır
            o.pop("summary", None)
            _tail(o, "events", n=5, cols=EVENT_COLS_SHORT)
        else:
            _tail(o, "events", cols=EVENT_COLS)
    elif name == "calculate_fundamental_change":
        _tail(o, "quarters", cols=QUARTER_COLS)
        for q in o["quarters"]:
            q["reaction"] = {k: q["reaction"].get(k) for k in REACTION_COLS}
        o.pop("note", None)                      # FUND_NOTE get_fundamental_metrics'te de var; reaction_note kalır
    elif name == "get_macro_context":
        if o.get("meta", {}).get("frequency") != "günlük":
            _tail(o, "rows", n=3)               # uzun dönemde özet yeterli; son 3 ay sonu satırı kalır
    elif name == "get_news_events":
        _tail(o, "events", n=10, cols=NEWS_COLS)
    elif name == "build_evidence_bundle":
        for k in ("observed", "history", "backtest"):
            o.pop(k, None)
    elif name == "run_backtest":
        for scope in ("stock", "group"):
            if isinstance(o.get(scope), dict):
                _tail(o[scope], "false_positives", n=10)
        _tail(o, "recent_trades", n=5)
    return _round(o)


# ---------------------------------------------------------------- denetim (karar 1a ve 4a)
def parse_action(text: str):
    m = ACTION_RE.findall(text or "")
    return m[-1].upper() if m else None


def advice_hits(text: str) -> list:
    return [m.group(0) for p in ADVICE_RE for m in p.finditer(text or "")]


def hard_block(outputs: list) -> bool:
    """Araç çıktılarından herhangi birinde kalite kapısı HARD_BLOCK verdiyse True (kural modelin eylemini ezer)."""
    def walk(x):
        if isinstance(x, dict):
            if x.get("status") == "HARD_BLOCK":
                return True
            return any(walk(v) for v in x.values())
        if isinstance(x, list):
            return any(walk(v) for v in x)
        return False
    return any(walk(o["output"]) for o in outputs if o["ok"])


def validate_answer(text: str, question: str, outputs: list) -> dict:
    nums = check_numbers(text, question, *[o["output"] for o in outputs if o["ok"]])
    action = parse_action(text)
    adv = advice_hits(text)
    problems = []
    if nums["unsupported"]:
        problems.append("araç çıktısında olmayan sayılar: " + ", ".join(nums["unsupported"]))
    if adv:
        problems.append("tavsiye/hedef fiyat ifadesi: " + "; ".join(adv))
    if action not in ACTIONS:
        problems.append(f"son satırda izinli eylem yok ('{action}')" if action else "son satırda ANALİTİK DURUM yok")
    return dict(numbers=nums, advice=adv, action=action, action_valid=action in ACTIONS, problems=problems)


# ---------------------------------------------------------------- akış
def _assistant(r: dict) -> dict:
    return dict(role="assistant", content=r["content"] or None,
                tool_calls=[dict(id=c["id"], type="function", function=dict(name=c["name"], arguments=c["arguments"]))
                            for c in r["tool_calls"]])


async def _call_tool(client, names: set, call: dict) -> dict:
    """Tek araç çağrısı: ad ve JSON denetimi harness'ta, parametre denetimi MCP şemasında ve araçta."""
    rec = dict(name=call["name"], arguments=call["arguments"], ok=False, error=None, output=None)
    if call["name"] not in names:
        rec["error"] = f"bilinmeyen araç: '{call['name']}'. Geçerli: {', '.join(sorted(names))}"
        return rec
    try:
        args = json.loads(call["arguments"] or "{}")
        assert isinstance(args, dict)
    except (json.JSONDecodeError, AssertionError):
        rec["error"] = f"argümanlar JSON nesnesi değil: {call['arguments'][:200]}"
        return rec
    rec["args"] = args
    r = await client.call_tool(call["name"], args)
    text = "".join(getattr(x, "text", "") for x in r.content)
    if r.is_error:
        rec["error"] = text[:500]
        return rec
    rec["ok"], rec["output"] = True, compact(call["name"], json.loads(text))
    return rec


async def _converse(question: str, asof: str, llm, outputs: list) -> tuple:
    """CALL_TOOLS -> ANSWER -> VALIDATE (+ bir düzeltme). (cevap, ham cevap, düzeltme sayısı, denetim, son durum)."""
    messages = [dict(role="system", content=SYSTEM.format(asof=asof, actions=", ".join(
                    f"{a} ({h})" for a, h in ACTION_HELP.items()))),
                dict(role="user", content=f"Analiz tarihi: {asof}\n\n{question}")]
    answer = None
    with data_ceiling(asof):
        async with Client(build_server()) as client:
            mcp_tools = (await client.list_tools()).tools
            tools, names = tool_schemas(mcp_tools, max_desc=None), {t.name for t in mcp_tools}
            for _ in range(MAX_CALLS + 2):                             # CALL_TOOLS: model araç çağırabilir
                r = llm.chat(messages, tools=tools)
                if not r["tool_calls"]:
                    answer = r["content"]
                    break
                messages.append(_assistant(r))
                for c in r["tool_calls"]:
                    if len(outputs) >= MAX_CALLS:
                        content = f"HATA: en fazla {MAX_CALLS} araç çağrısı yapılabilir; bu çağrı yapılmadı."
                        outputs.append(dict(name=c["name"], arguments=c["arguments"], ok=False, output=None,
                                            error="çağrı hakkı bitti", over_budget=True))
                    else:
                        rec = await _call_tool(client, names, c)
                        outputs.append(rec)
                        content = json.dumps(rec["output"], ensure_ascii=False) if rec["ok"] else f"HATA: {rec['error']}"
                    messages.append(dict(role="tool", tool_call_id=c["id"], content=content))
                if len(outputs) >= MAX_CALLS:
                    break
            if answer is None:                                         # ANSWER: hak bitti, araçsız cevap iste
                messages.append(dict(role="user", content=FORCE_TEXT))
                answer = llm.chat(messages, tools=None)["content"]
    raw, retries = answer, 0
    v = validate_answer(answer, question, outputs)                   # VALIDATE
    if v["problems"]:
        retries = 1
        messages += [dict(role="assistant", content=answer),
                     dict(role="user", content=RETRY_TEXT.format(problems="; ".join(v["problems"])))]
        answer = llm.chat(messages, tools=None)["content"]
        v = validate_answer(answer, question, outputs)
    return answer, raw, retries, v


async def _run(question: str, asof: str, llm) -> dict:
    t0 = time.time()
    outputs = []
    try:
        answer, raw, retries, v = await _converse(question, asof, llm, outputs)
        error = None
    except Exception as e:                       # model sunucusu düştü, bellek yetmedi vb.: çökme yok, güvenli durum
        while isinstance(e, BaseExceptionGroup) and e.exceptions:     # MCP istemcisi asıl hatayı grup içine sarar
            e = e.exceptions[0]
        error = f"{type(e).__name__}: {str(e)[:300]}"
        answer = raw = None
        retries, v = 0, dict(numbers=dict(n_numbers=0, unsupported=[], rounding=[]), advice=[], action=None,
                            action_valid=False, problems=[f"model hatası: {error}"])
    ok_calls = [o for o in outputs if o["ok"]]
    action, source, status = v["action"], "model", "ok"
    if error:
        action, source, status = "ANALYSIS_UNSAFE", "kural: model yanıt veremedi", "llm_error"
        answer = f"Model yanıt veremedi ({error}).\nANALİTİK DURUM: ANALYSIS_UNSAFE"
    elif v["problems"]:
        action, source, status = "ANALYSIS_UNSAFE", "kural: denetim iki kez geçmedi", "withheld"
        answer = ("Cevap denetimden geçmediği için gösterilmiyor (" + "; ".join(v["problems"]) + ").\n"
                  "ANALİTİK DURUM: ANALYSIS_UNSAFE")
    elif not ok_calls:
        action, source = "ANALYSIS_UNSAFE", "kural: başarılı araç çağrısı yok, cevap kanıta dayanmıyor"
    elif hard_block(outputs):
        action, source = "ANALYSIS_UNSAFE", "kural: kalite kapısı HARD_BLOCK"
    return dict(question=question, asof=asof, model=getattr(llm, "name", "?"), action=action, action_source=source,
                status=status, answer=answer, answer_raw=raw, tool_calls=outputs, validation=v, retries=retries,
                invalid_tool_calls=sum(not o["ok"] for o in outputs), llm_calls=llm.calls, error=error,
                latency=round(time.time() - t0, 1),
                states=["START", "SET_CEILING", "CALL_TOOLS", "ANSWER", "VALIDATE", "LOG_AND_CLOSE"])


def run_question(question: str, asof: str, llm=None, db: Path = QDB, log: bool = True) -> dict:
    res = asyncio.run(_run(question, str(asof), llm or LocalLLM()))
    if log:
        res["qid"] = log_question(res, db)
    return res


# ---------------------------------------------------------------- günlük ve tekrar oynatma
def log_question(res: dict, db: Path = QDB) -> str:
    created = datetime.now().isoformat(timespec="seconds")
    qid = hashlib.sha256(f"{res['question']}|{res['asof']}|{res['model']}|{created}".encode()).hexdigest()[:12]
    calls = [{k: o.get(k) for k in ("name", "arguments", "args", "ok", "error")} for o in res["tool_calls"]]
    with sqlite3.connect(db) as con:
        con.execute(SCHEMA)
        con.execute("INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (qid, created, res["question"], res["asof"], res["model"], res["action"], res["action_source"],
                     res["status"], res["answer"], res["answer_raw"], json.dumps(calls, ensure_ascii=False),
                     json.dumps([o["output"] for o in res["tool_calls"]], ensure_ascii=False),
                     json.dumps(res["validation"], ensure_ascii=False), res["retries"], res["llm_calls"], res["latency"]))
    return qid


def load_question(qid: str, db: Path = QDB) -> dict:
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM questions WHERE qid = ?", (qid,)).fetchone()
    if row is None:
        raise ValueError(f"soru kaydı yok: {qid}")
    out = dict(row)
    for k in ("tool_calls", "outputs", "validation"):
        out[k] = json.loads(out[k])
    return out


def replay_question(qid: str, db: Path = QDB) -> dict:
    """Kayıttaki araç çağrılarını aynı tavanla yeniden çalıştırır; modelin gördüğü çıktılar aynı mı?"""
    q = load_question(qid, db)

    async def go():
        with data_ceiling(q["asof"]):
            async with Client(build_server()) as client:
                names = {t.name for t in (await client.list_tools()).tools}
                return [await _call_tool(client, names, dict(name=c["name"], arguments=c["arguments"]))
                        for c in q["tool_calls"]]
    again = asyncio.run(go())
    same = [a["output"] == b for a, b in zip(again, q["outputs"])]
    return dict(qid=qid, asof=q["asof"], action=q["action"], n_calls=len(again), same_outputs=all(same), per_call=same)


def _print(res: dict):
    print(f"\nAraç çağrıları ({len(res['tool_calls'])}, geçersiz {res['invalid_tool_calls']}):")
    for o in res["tool_calls"]:
        print(f"  {'OK ' if o['ok'] else 'HATA'} {o['name']}({o['arguments']})" + ("" if o["ok"] else f" -> {o['error']}"))
    print(f"\n{res['answer']}\n")
    v = res["validation"]
    print(f"Eylem: {res['action']} ({res['action_source']}) | durum {res['status']} | düzeltme {res['retries']} | "
          f"sayı {v['numbers']['n_numbers']}, uydurma {len(v['numbers']['unsupported'])} | {res['latency']} sn | "
          f"kayıt {res.get('qid', '-')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Soru modu: yerel model + MCP araçları + harness denetimi")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--asof", required=False, help="analiz tarihi (YYYY-AA-GG); sonrası hiçbir araçta görünmez")
    ap.add_argument("--url", default=None, help="yerel OpenAI uyumlu sunucu (varsayılan http://127.0.0.1:8080/v1)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--replay", default=None, help="kayıt kimliği: araç çağrılarını yeniden çalıştır ve karşılaştır")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.replay:
        print(replay_question(a.replay))
    else:
        if not (a.question and a.asof):
            ap.error("soru ve --asof gerekli")
        _print(run_question(a.question, a.asof, LocalLLM(model=a.model, base_url=a.url), log=not a.no_log))
