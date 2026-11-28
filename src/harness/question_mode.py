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
                            + tam iz kaydı reports/traces/<kimlik>.json (write_trace)
İz kaydı (sunum duyurusu, 28 Kasım 2026): model + sürüm + adaptör SHA-256, sıcaklık, seed, sistem ve kullanıcı
istemi, modele sunulan araç tanımları, her model ve araç adımı zaman damgasıyla (araç parametreleri, tarih aralıkları,
modelin gördüğü ve tam araç çıktısı), harness'ın eklediği mesajlar (düzeltme isteği), nihai cevap ve denetim.
Kullanıcı kararları (24 Kasım 2026): 1a eylemi model seçer, harness denetler; 2a 15 aracın hepsi, en fazla 6 çağrı;
3a çıktılar modele kısaltılmış verilir (uzun satır listeleri atılır, 4 ondalık); 4a uydurma sayı -> bir düzeltme
denemesi, yine olursa ANALYSIS_UNSAFE.

Çalıştırma (önce yerel sunucu, adaptörsüz: venv-llm/bin/mlx_lm.server --model models/Qwen3-4B-Instruct-2507-4bit
--port 8080; adaptörü istemci her istekle gönderir, src/agents/llm.LocalLLM):
  python -m src.harness.question_mode --asof 2026-09-10 "Perakende sektöründe (BIMAS, MGROS, MAVI) ..."
  python -m src.harness.question_mode --adapter none --asof 2026-09-10 "..."      # taban model (karşılaştırma)
"""
import argparse
import asyncio
import copy
import hashlib
import json
import re
import sqlite3
import subprocess
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
TRACE_VERSION = 1
DATE_KEYS = ("start", "end", "date", "asof", "oos_start")      # araç parametrelerindeki tarih alanları
MAX_CALLS = 6
BUDGET_ERROR = "çağrı hakkı bitti"     # hak bitince çalıştırılmayan çağrının kayıttaki hata metni
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
2. Sorudaki tarih aralığını ve hisseleri araç parametrelerine aynen yaz. Soruda başlangıç yoksa veri döneminin
   başını (2024-01-01) kullan. Analiz tarihinden sonrası yoktur.
3. Yatırım tavsiyesi, al/sat önerisi ya da hedef fiyat verme. İstenirse bunu açıkça reddet; bunun yerine geçmiş
   verideki ilişkiyi araçla göster. "Kesin yükselir" gibi iddiaları doğrulanmamış iddia olarak belirt.
4. Penceresi analiz tarihinde kapanmamış olayları (complete=false, n_incomplete) istatistiğe katma ve bunu söyle.
   Olay sayısı azsa uyar. Nedensellik iddia etme. Birçok desen denendiyse çoklu test riskini belirt.
5. Oranları yüzde olarak yaz (0.1126 = %11,26). Sayıları araç çıktısındaki gibi doğru yuvarla.
   Araç hata verirse mesajı oku: parametre yanlışsa (geçersiz ad, desen, tarih sırası) mesajdaki geçerli değerlerle
   çağrıyı düzeltip yeniden çağır. İstenen hisse, endeks ya da sektör yoksa ya da istek analiz tarihinden sonrasına
   aitse düzeltme yapma.
6. Cevabı Türkçe, kısa ve maddeli yaz: önce bulgular, sonra sınırlamalar, sonra 'Kural kontrolü' (kuralın her
   koşulu, araç çıktısındaki değeriyle, evet/hayır), sonra eylem gerekçesi. Son satır tam olarak:
   ANALİTİK DURUM: <eylem>
   İzinli eylemler: {actions}.
7. Eylem kuralları:
   - Geride kalma: sektör ortancası > 0, hissede en az 10 tamamlanmış olay, 20 günde isabet en az %60 ve yanlış
     pozitif en fazla %40 ise POTENTIAL_CATCH_UP_CANDIDATE; olay 10'dan azsa INVESTIGATE; aksi WATCH. Geçmiş test
     edilmediyse WATCH.
   - Takvim deseni: araç 'kararsız' dediyse ya da maliyet sonrası sonuç sıfır ya da altındaysa REJECT_SIGNAL; aksi WATCH.
   - Teknik olay: en az 10 olay, 5 günde başarı en az %60 (success_rate) ve son 3 işlem gününde yeni olay varsa
     (event_in_last_3_trading_days) TECHNICAL_REVERSAL_CANDIDATE; olay 10'dan azsa INVESTIGATE; başarı en fazla %50
     ise REJECT_SIGNAL; aksi WATCH.
   - Bilanço: son çeyrek iyileşen (last_quarter.improving) ve son 63 işlem gününde açıklandıysa
     (last_quarter.trading_days_since_disclosure ≤ 63) FUNDAMENTAL_IMPROVEMENT_CANDIDATE; iyileşme bilinmiyorsa
     INVESTIGATE; aksi HOLD. Bilanço sorusunda yalnızca bu üç eylemden biri seçilir; tepki penceresi kapanmamış
     çeyrek yalnızca belirtilir, eylemi değiştirmez.
   - Tavsiye ya da iddia: iddia verilerle desteklenmiyorsa INVESTIGATE, destekleniyorsa WATCH (kur iddiasında
     destek = beta > 0 ve p < 0,05; dönem 60 işlem gününden kısa olduğu için ilişki hesaplanamadıysa iddia
     desteklenmiyor sayılır).
   - Betimleyici sorular HOLD; düzeltilemeyen araç hatası ya da analiz tarihinden sonraya ait istek
     ANALYSIS_UNSAFE.""".strip()
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
FIN_ONLY = ("ni_yoy", "roe_q")                     # banka/sigorta: FAVÖK yok, cevap bunları kullanır
NONFIN_ONLY = ("ebitda_margin_q", "ebitda_margin_chg", "nd_ebitda", "deleveraging")


def quarter_table(quarters: list) -> dict:
    """Çeyrek listesi -> {'columns': [...], 'rows': [[...], ...]} (S4 kısaltması, 25 Kasım 2026).

    Alan adları her çeyrekte tekrar yazılmaz; tepki alanları satıra düzleşir. Yeni sayı üretmez, yalnızca:
    FAVÖK'lü şirkette ni_yoy/roe_q, bankada (FAVÖK hep boş) FAVÖK sütunları çıkar; *_pct tepkileri 2 ondalığa
    yuvarlanır (cevapta da 2 ondalık yazılır); window_end_20 yalnızca son çeyrekte kalır (cevap yalnız onu kullanır)."""
    bank = all(q.get("ebitda_margin_q") is None for q in quarters)
    drop = set(NONFIN_ONLY if bank else FIN_ONLY)
    flat = []
    for i, q in enumerate(quarters):
        d = {k: v for k, v in q.items() if k != "reaction" and k not in drop}
        for k in REACTION_COLS:
            v = (q.get("reaction") or {}).get(k)
            if k == "window_end_20" and i < len(quarters) - 1:
                continue
            d[k] = round(v, 2) if isinstance(v, float) and k.endswith("_pct") else v
        flat.append(d)
    cols = list(dict.fromkeys(c for d in flat for c in d))
    return dict(columns=cols, rows=[[d.get(c) for c in cols] for d in flat])


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
        o["quarters"] = quarter_table(o["quarters"])
        o.pop("note", None)                      # FUND_NOTE get_fundamental_metrics'te de var; reaction_note kalır
    elif name == "get_fundamental_metrics":
        o.pop("note", None)                      # uzun tanım notu (~580 token) günlükte kalır, modele gitmez
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


def trim_after_action(text: str) -> str:
    """Son 'ANALİTİK DURUM' satırından sonrasını atar (temel model istemdeki eylem listesini kopyalıyordu)."""
    ms = list(ACTION_RE.finditer(text or ""))
    if not ms:
        return text
    end = (text.find("\n", ms[-1].end()) + 1) or len(text)
    return text[:end].rstrip()


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


def system_prompt(asof: str) -> str:
    return SYSTEM.format(asof=asof, actions=", ".join(f"{a} ({h})" for a, h in ACTION_HELP.items()))


def first_messages(question: str, asof: str) -> list:
    """Soru modunun ve ince ayar veri setinin ortak başlangıcı (sistem + kullanıcı)."""
    return [dict(role="system", content=system_prompt(asof)),
            dict(role="user", content=f"Analiz tarihi: {asof}\n\n{question}")]


def validate_answer(text: str, question: str, outputs: list) -> dict:
    """Bağlam = soru + modelin gördüğü araç çıktıları + araç hata mesajları + kural eşikleri (sistem metni)."""
    ctx = [o["output"] for o in outputs if o["ok"]] + [o["error"] for o in outputs if not o["ok"] and o.get("error")]
    nums = check_numbers(text, question, system_prompt(""), *ctx)
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
    full = json.loads(text)
    rec["ok"], rec["output"], rec["output_full"] = True, compact(call["name"], full), full
    return rec


async def list_tools(client) -> tuple:
    mcp_tools = (await client.list_tools()).tools
    return tool_schemas(mcp_tools, max_desc=None), {t.name for t in mcp_tools}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _dates(args) -> dict:
    return {k: args[k] for k in DATE_KEYS if isinstance(args, dict) and k in args}


def _chat(llm, messages: list, tools, steps: list, phase: str) -> dict:
    """llm.chat + iz adımı (zaman damgaları, modelin ürettiği metin ve araç istekleri). Hata da adım olarak kalır."""
    t0, step = time.time(), dict(step=len(steps) + 1, type="llm", phase=phase, started_at=_now(),
                                 n_input_messages=len(messages), tools_offered=bool(tools))
    steps.append(step)
    try:
        r = llm.chat(messages, tools=tools)
    except Exception as e:
        step.update(ended_at=_now(), seconds=round(time.time() - t0, 2), error=f"{type(e).__name__}: {str(e)[:300]}")
        raise
    step.update(ended_at=_now(), seconds=round(time.time() - t0, 2), content=r["content"],
                tool_calls=[dict(id=c["id"], name=c["name"], arguments=c["arguments"]) for c in r["tool_calls"]],
                prompt_tokens=r.get("prompt_tokens"), completion_tokens=r.get("completion_tokens"))
    return r


def _harness_message(messages: list, steps: list, content: str, reason: str):
    """Harness'ın modele eklediği kullanıcı mesajı (hak bitti / düzeltme isteği): istemin parçası, izde görünür."""
    messages.append(dict(role="user", content=content))
    steps.append(dict(step=len(steps) + 1, type="harness_message", reason=reason, at=_now(), content=content))


async def _converse(question: str, asof: str, llm, outputs: list, trace: dict) -> tuple:
    """CALL_TOOLS -> ANSWER -> VALIDATE (+ bir düzeltme). (cevap, ham cevap, düzeltme sayısı, denetim).
    trace['steps'] her model/araç adımıyla, trace['tools'] modele sunulan araç tanımlarıyla dolar."""
    messages = first_messages(question, asof)
    steps = trace["steps"]
    answer = None
    with data_ceiling(asof):
        async with Client(build_server()) as client:
            tools, names = await list_tools(client)
            trace["tools"] = tools
            for _ in range(MAX_CALLS + 2):                             # CALL_TOOLS: model araç çağırabilir
                r = _chat(llm, messages, tools, steps, "call_tools")
                if not r["tool_calls"]:
                    answer = r["content"]
                    break
                messages.append(_assistant(r))
                for c in r["tool_calls"]:
                    started = _now()
                    if len(outputs) >= MAX_CALLS:
                        content = f"HATA: en fazla {MAX_CALLS} araç çağrısı yapılabilir; bu çağrı yapılmadı."
                        rec = dict(name=c["name"], arguments=c["arguments"], ok=False, output=None,
                                   error=BUDGET_ERROR, over_budget=True)
                    else:
                        rec = await _call_tool(client, names, c)
                        content = json.dumps(rec["output"], ensure_ascii=False) if rec["ok"] else f"HATA: {rec['error']}"
                    outputs.append(rec)
                    steps.append(dict(step=len(steps) + 1, type="tool", call_index=len(outputs) - 1,
                                      tool_call_id=c["id"], started_at=started, ended_at=_now(), name=rec["name"],
                                      arguments=rec["arguments"], args=rec.get("args"), dates=_dates(rec.get("args")),
                                      data_ceiling=asof, ok=rec["ok"], error=rec.get("error"),
                                      over_budget=bool(rec.get("over_budget")), output_model=rec.get("output"),
                                      output_full=rec.get("output_full"), message_to_model=content))
                    messages.append(dict(role="tool", tool_call_id=c["id"], content=content))
                if len(outputs) >= MAX_CALLS:
                    break
            if answer is None:                                         # ANSWER: hak bitti, araçsız cevap iste
                _harness_message(messages, steps, FORCE_TEXT, "araç çağrı hakkı bitti")
                answer = _chat(llm, messages, None, steps, "answer")["content"]
    raw, retries = answer, 0
    v = validate_answer(answer, question, outputs)                   # VALIDATE
    steps.append(dict(step=len(steps) + 1, type="validate", at=_now(), problems=v["problems"],
                      unsupported_numbers=v["numbers"]["unsupported"], advice=v["advice"], action=v["action"]))
    if v["problems"]:
        retries = 1
        messages.append(dict(role="assistant", content=answer))
        _harness_message(messages, steps, RETRY_TEXT.format(problems="; ".join(v["problems"])), "denetimden geçmedi")
        answer = _chat(llm, messages, None, steps, "retry")["content"]
        v = validate_answer(answer, question, outputs)
        steps.append(dict(step=len(steps) + 1, type="validate", at=_now(), problems=v["problems"],
                          unsupported_numbers=v["numbers"]["unsupported"], advice=v["advice"], action=v["action"]))
    return answer, raw, retries, v


def llm_info(llm) -> dict:
    """Modelin kimliği ve üretim ayarları (LocalLLM.info; sahte modellerde yalnızca ad)."""
    if hasattr(llm, "info"):
        return llm.info()
    return dict(name=getattr(llm, "name", "?"), temperature=getattr(llm, "temperature", None),
                seed=getattr(llm, "seed", None), max_tokens=getattr(llm, "max_tokens", None))


async def _run(question: str, asof: str, llm) -> dict:
    t0, started = time.time(), _now()
    calls0 = llm.calls          # aynı LLM nesnesi birçok soruda kullanılabilir (değerlendirme): bu sorunun çağrıları
    outputs, trace = [], dict(steps=[], tools=None)
    try:
        answer, raw, retries, v = await _converse(question, asof, llm, outputs, trace)
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
    elif not ok_calls and outputs and v["action"] == "ANALYSIS_UNSAFE":
        source = "kural: başarılı araç çağrısı yok; model araç hatasını açıkladı"   # ör. bilinmeyen hisse: metin kalır
    elif not ok_calls:                           # kanıt yok: metin de gösterilmez (temel model "0 örüntü" uydurdu)
        action, source, status = "ANALYSIS_UNSAFE", "kural: başarılı araç çağrısı yok, cevap kanıta dayanmıyor", \
            "no_evidence"
        answer = ("Hiçbir araç çağrısı başarılı olmadığı için kanıta dayalı bir cevap verilemiyor.\n"
                  "ANALİTİK DURUM: ANALYSIS_UNSAFE")
    elif hard_block(outputs):
        action, source = "ANALYSIS_UNSAFE", "kural: kalite kapısı HARD_BLOCK"
    if status == "ok":
        answer = trim_after_action(answer)
    return dict(question=question, asof=asof, model=getattr(llm, "name", "?"), action=action, action_source=source,
                status=status, answer=answer, answer_raw=raw, tool_calls=outputs, validation=v, retries=retries,
                invalid_tool_calls=sum(not o["ok"] for o in outputs), llm_calls=llm.calls - calls0, error=error,
                latency=round(time.time() - t0, 1), started_at=started, ended_at=_now(), llm_info=llm_info(llm),
                steps=trace["steps"], tools_offered=trace["tools"],
                states=["START", "SET_CEILING", "CALL_TOOLS", "ANSWER", "VALIDATE", "LOG_AND_CLOSE"])


def run_question(question: str, asof: str, llm=None, db: Path = QDB, log: bool = True, trace_dir: Path = None) -> dict:
    """log=True: soru günlüğüne (db) yazar ve tam iz kaydını trace_dir'e (varsayılan <db klasörü>/traces) koyar."""
    res = asyncio.run(_run(question, str(asof), llm or LocalLLM()))
    if log:
        res["qid"] = log_question(res, db)
        res["trace"] = str(write_trace(res, Path(trace_dir) if trace_dir else Path(db).parent / "traces"))
    return res


def _git() -> dict:
    """Cevabı üreten kodun sürümü (commit + kaydedilmemiş değişiklik var mı)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return dict(commit=head or None, uncommitted_changes=bool(dirty))
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, uncommitted_changes=None)


def trace_record(res: dict) -> dict:
    """Bir sorunun tam iz kaydı (JSON). Mesaj dizisi adımlardan birebir kurulabilir: sistem + kullanıcı istemi,
    sonra sırayla model adımları, araç adımları (message_to_model = modele dönen metin) ve harness mesajları."""
    info = res["llm_info"]
    return dict(
        trace_version=TRACE_VERSION, qid=res.get("qid"), started_at=res["started_at"], ended_at=res["ended_at"],
        latency_seconds=res["latency"], model=info,
        generation=dict(temperature=info.get("temperature"), seed=info.get("seed"), max_tokens=info.get("max_tokens"),
                        max_tool_calls=MAX_CALLS, correction_rounds=1),
        analysis_date=res["asof"], data_ceiling=res["asof"], code=_git(),
        prompts=dict(system=system_prompt(res["asof"]), user=first_messages(res["question"], res["asof"])[1]["content"]),
        question=res["question"], tools_offered=res["tools_offered"], steps=res["steps"],
        final=dict(answer=res["answer"], action=res["action"], action_source=res["action_source"],
                   status=res["status"], retries=res["retries"], llm_calls=res["llm_calls"], error=res["error"],
                   n_tool_calls=len(res["tool_calls"]), invalid_tool_calls=res["invalid_tool_calls"],
                   validation=res["validation"]),
        first_answer=res["answer_raw"], states=res["states"])


def write_trace(res: dict, trace_dir: Path) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    f = trace_dir / f"{res['qid']}.json"
    f.write_text(json.dumps(trace_record(res), ensure_ascii=False, indent=2, default=str))
    return f


# ---------------------------------------------------------------- günlük ve tekrar oynatma
def log_question(res: dict, db: Path = QDB) -> str:
    now = datetime.now()
    created = now.isoformat(timespec="seconds")
    # Kimlikte mikrosaniye: aynı soru aynı saniyede iki kez kaydedilince (26 Kasım 2026 değerlendirmesi, sunucu
    # çökünce 0,0 sn'de biten çalıştırmalar) saniyelik damga aynı kimliği verip UNIQUE hatası çıkarıyordu.
    stamp = now.isoformat(timespec="microseconds")
    qid = hashlib.sha256(f"{res['question']}|{res['asof']}|{res['model']}|{stamp}".encode()).hexdigest()[:12]
    calls = [{k: o.get(k) for k in ("name", "arguments", "args", "ok", "error", "over_budget")} for o in res["tool_calls"]]
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
    # Çağrı hakkı bittiği için hiç çalıştırılmamış çağrılar tekrar oynatılmaz (26 Kasım 2026 bulgusu: çalıştırılınca
    # gerçek çıktı verip 'aynı değil' sayılıyordu). Eski kayıtlarda over_budget alanı yok -> hata metninden tanınır.
    ran = [(c, o) for c, o in zip(q["tool_calls"], q["outputs"])
           if not (c.get("over_budget") or c.get("error") == BUDGET_ERROR)]

    async def go():
        with data_ceiling(q["asof"]):
            async with Client(build_server()) as client:
                names = {t.name for t in (await client.list_tools()).tools}
                return [await _call_tool(client, names, dict(name=c["name"], arguments=c["arguments"]))
                        for c, _ in ran]
    again = asyncio.run(go())
    same = [a["output"] == o for a, (_, o) in zip(again, ran)]
    return dict(qid=qid, asof=q["asof"], action=q["action"], n_calls=len(again),
                n_skipped=len(q["tool_calls"]) - len(ran), same_outputs=all(same), per_call=same)


def _print(res: dict):
    print(f"\nAraç çağrıları ({len(res['tool_calls'])}, geçersiz {res['invalid_tool_calls']}):")
    for o in res["tool_calls"]:
        print(f"  {'OK ' if o['ok'] else 'HATA'} {o['name']}({o['arguments']})" + ("" if o["ok"] else f" -> {o['error']}"))
    print(f"\n{res['answer']}\n")
    v = res["validation"]
    print(f"Eylem: {res['action']} ({res['action_source']}) | durum {res['status']} | düzeltme {res['retries']} | "
          f"sayı {v['numbers']['n_numbers']}, uydurma {len(v['numbers']['unsupported'])} | {res['latency']} sn | "
          f"kayıt {res.get('qid', '-')}" + (f"\nİz kaydı: {res['trace']}" if res.get("trace") else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Soru modu: yerel model + MCP araçları + harness denetimi")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--asof", required=False, help="analiz tarihi (YYYY-AA-GG); sonrası hiçbir araçta görünmez")
    ap.add_argument("--url", default=None, help="yerel OpenAI uyumlu sunucu (varsayılan http://127.0.0.1:8080/v1)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adaptör klasörü (varsayılan models/adapters/bist-qwen3-4b-v4); 'none' = taban model")
    ap.add_argument("--seed", type=int, default=None, help="sunucuya gönderilen seed (varsayılan 42)")
    ap.add_argument("--replay", default=None, help="kayıt kimliği: araç çağrılarını yeniden çalıştır ve karşılaştır")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.replay:
        print(replay_question(a.replay))
    else:
        if not (a.question and a.asof):
            ap.error("soru ve --asof gerekli")
        _print(run_question(a.question, a.asof, LocalLLM(model=a.model, base_url=a.url, adapter=a.adapter, seed=a.seed),
                            log=not a.no_log))
