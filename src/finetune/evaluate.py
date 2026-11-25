"""Yerel model 5. adım: taban model ile ince ayarlı model (v1) görülmemiş sorularda karşılaştırılır.

Kullanıcı kararları (25 Kasım 2026): 1a/2a soru seti src/finetune/heldout.py (40 soru, yeni ifadeler, eğitimde olmayan
hisse/tarih); 3a bir araç çağrısı, araç adı ve ana parametreleri (hisse/sektör, olay/örüntü/strateji, ufuk, koşullar,
soruda geçen tarihler) altınla aynıysa doğru sayılır, fazladan geçerli parametre sorun değildir (soruda geçmeyen tarih,
ör. gösterge için geriye 14 gün, serbesttir); 4a her modelde 10 soru ikinci kez çalışır ve kayıtlı bütün araç
çağrıları yeniden oynatılır.

Metrikler (PDF §30 + hocanın duyurusu), her model için toplam ve kategori bazında:
  doğru eylem oranı (son eylem = altın eylem; tersi desteklenmeyen sonuç oranı), geçersiz araç çağrısı oranı (hata
  sorularında bilinmeyen kodla yapılan, beklenen hatalı çağrılar hariç), kanıt tamlığı (altın çağrıların karşılanan
  oranı) ve tam kanıtlı soru oranı, backtest çağrısı doğruluğu, sayı uydurma oranı (ilk cevapta, denetimden önce),
  tavsiye/hedef fiyat (ilk cevapta yakalanan ve kullanıcıya ulaşan), tekrarlanabilirlik, araç çağrısı, süre.

Sıra: sunucu adaptörsüz açılır, adaptörü LocalLLM her istekle gönderir; taban 40, v1 40, v1 tekrar 10, taban tekrar 10
(yalnızca iki model değişimi). Yarıda kalırsa kaldığı yerden sürer. Ayarlar canlı sistemle aynıdır (LocalLLM
varsayılanları, sıcaklık 0, en fazla 6 araç çağrısı, bir düzeltme).

Çıktı (reports/local_llm/eval_v1/): heldout.jsonl, raw.jsonl (her çalıştırma), question_log.sqlite (soru modu günlüğü,
araç çıktılarıyla), replay.jsonl, per_question.csv, summary.md.
Çalıştırma (proje kökü, ana venv; mlx_lm.server'ı kendisi başlatır ve kapatır):
  python -m src.finetune.heldout                # soru seti (bir kez)
  python -m src.finetune.evaluate               # ~1,5 saat
  python -m src.finetune.evaluate --summary     # yalnızca özet
"""
import argparse
import csv
import json
import os
import re
import socket
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from src.agents.llm import BASE, LOCAL_ADAPTER, LOCAL_MODEL, LocalLLM
from src.finetune.heldout import HELDOUT, MONTHS_TR, OUT, group_key, load_heldout
from src.harness import question_mode as QM
from src.mcp_server import tools as TOOLS

MODELS = {"base": BASE, "v1": LOCAL_ADAPTER}
CATS = ("s1", "s2", "s3", "s4", "s5", "other", "error")
REPEAT_N = 10
PORT = 8090
SERVER_LOG = Path("data/finetune/eval_server.log")
DATE_PARAMS = ("start", "end", "date", "asof", "oos_start")


# ---------------------------------------------------------------- puanlama (karar 3a)
def stated_dates(question: str, asof: str) -> set:
    """Soruda (ve 'Analiz tarihi' başlığında) geçen tarihler, ISO biçiminde. Biçimler: 2024-01-01, 01.01.2024,
    1 Ocak 2024, '2024 başından'."""
    out = {asof}
    out |= {f"{y}-{m}-{d}" for y, m, d in re.findall(r"(\d{4})-(\d{2})-(\d{2})", question)}
    out |= {f"{y}-{int(m):02d}-{int(d):02d}" for d, m, y in re.findall(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})", question)}
    months = {m: i + 1 for i, m in enumerate(MONTHS_TR)}
    for d, mon, y in re.findall(rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS_TR)})\s+(\d{{4}})", question):
        out.add(f"{y}-{months[mon]:02d}-{int(d):02d}")
    out |= {f"{y}-01-01" for y in re.findall(r"\b(\d{4}) başı", question)}
    return out


def _cond(c) -> str:
    return "+".join(sorted(str(c).replace(" ", "").split("+")))


def key_args(args: dict) -> dict:
    """Karşılaştırılan ana parametreler, normalleştirilmiş."""
    out = {}
    if "sector" in args or "tickers" in args:
        out["group"] = group_key(args)
    for k in ("ticker", "index"):
        if args.get(k) is not None:
            out[k] = str(args[k]).strip().upper().removesuffix(".IS")
    for k in ("event", "pattern", "strategy"):
        if args.get(k) is not None:
            out[k] = str(args[k]).strip()
    for k in ("horizon", "lookback"):
        if args.get(k) is not None:
            try:
                out[k] = int(args[k])
            except (TypeError, ValueError):
                out[k] = str(args[k])
    if args.get("conditions"):
        conds = args["conditions"] if isinstance(args["conditions"], list) else [args["conditions"]]
        out["conditions"] = tuple(sorted(_cond(c) for c in conds))
    for k in DATE_PARAMS:
        if args.get(k) is not None:
            out[k] = str(args[k])[:10]
    return out


def _args(rec: dict):
    if rec.get("args") is not None:
        return rec["args"]
    try:
        a = json.loads(rec.get("arguments") or "{}")
        return a if isinstance(a, dict) else None
    except json.JSONDecodeError:
        return None


def call_matches(gold: dict, rec: dict, stated: set) -> bool:
    """Araç adı ve altındaki her ana parametre aynı mı? Soruda geçmeyen tarih parametresi karşılaştırılmaz."""
    if rec["name"] != gold["name"]:
        return False
    m_args = _args(rec)
    if m_args is None:
        return False
    g, m = key_args(gold["args"]), key_args(m_args)
    return all(m.get(k) == v for k, v in g.items() if not (k in DATE_PARAMS and v not in stated))


def match_gold(gold: list, recs: list, stated: set) -> list:
    """Her altın çağrı için karşılandı mı (bir model çağrısı tek altın çağrıyı karşılar). Başarılı altın çağrıyı
    yalnızca başarılı çağrı karşılar; hata sorusunda bilerek başarısız olan altın çağrıyı başarısız çağrı da karşılar."""
    used, hits = set(), []
    for g in gold:
        j = next((j for j, r in enumerate(recs) if j not in used and (r["ok"] or not g["ok"])
                  and call_matches(g, r, stated)), None)
        if j is not None:
            used.add(j)
        hits.append(j is not None)
    return hits


def expected_failure(q: dict, rec: dict) -> bool:
    """Hata sorusunda, sorudaki bilinmeyen kod/ad ile yapılan çağrının hata vermesi beklenir (geçersiz sayılmaz)."""
    if q["cat"] != "error" or rec["ok"] or not q.get("bad"):
        return False
    return TOOLS._fold(q["bad"]) in TOOLS._fold(rec.get("arguments") or "")


def score(q: dict, r: dict) -> dict:
    stated = stated_dates(q["question"], q["asof"])
    recs = r["tool_calls"]
    hits = match_gold(q["gold_calls"], recs, stated)
    exp = sum(expected_failure(q, x) for x in recs)
    bt = [h for g, h in zip(q["gold_calls"], hits) if g["name"] == "run_backtest"]
    return dict(hid=q["hid"], cat=q["cat"], model=r["model"], run=r["run"], gold_action=q["gold_action"],
                action=r["action"], correct=r["action"] == q["gold_action"], status=r["status"], retries=r["retries"],
                n_calls=len(recs), invalid=sum(not x["ok"] for x in recs) - exp, expected_fail=exp,
                gold_n=len(hits), gold_hit=sum(hits), full_evidence=all(hits), bt_n=len(bt), bt_hit=sum(bt),
                first_numbers=r["first_numbers"], first_fabricated=len(r["first_unsupported"]),
                first_advice=len(r["first_advice"]), final_advice=len(r["final_advice"]),
                llm_calls=r["llm_calls"], latency=r["latency"])


def aggregate(rows: list) -> dict:
    """Bir modelin (1. çalıştırma) satırlarından metrikler."""
    s = lambda k: sum(x[k] for x in rows)
    n = len(rows)
    return dict(
        n=n,
        action_accuracy=s("correct") / n,
        unsupported_conclusion=1 - s("correct") / n,
        invalid_call_rate=s("invalid") / max(s("n_calls") - s("expected_fail"), 1),
        evidence_completeness=s("gold_hit") / max(s("gold_n"), 1),
        full_evidence=s("full_evidence") / n,
        backtest_call_correct=(s("bt_hit") / s("bt_n")) if s("bt_n") else None,
        fabricated_number_rate=s("first_fabricated") / max(s("first_numbers"), 1),
        answers_with_fabrication=sum(x["first_fabricated"] > 0 for x in rows) / n,
        advice_first=sum(x["first_advice"] > 0 for x in rows),
        advice_shown=sum(x["final_advice"] > 0 for x in rows),
        withheld=sum(x["status"] == "withheld" for x in rows),
        no_evidence=sum(x["status"] == "no_evidence" for x in rows),
        llm_error=sum(x["status"] == "llm_error" for x in rows),
        retried=sum(x["retries"] > 0 for x in rows),
        tool_calls=s("n_calls") / n,
        llm_calls=s("llm_calls") / n,
        latency_mean=s("latency") / n,
        latency_median=statistics.median(x["latency"] for x in rows),
    )


def reproducibility(raw: list) -> dict:
    """Aynı soru ikinci kez: aynı eylem, aynı araç çağrıları (ad + argüman metni), aynı cevap metni oranları."""
    by = {(r["hid"], r["model"], r["run"]): r for r in raw}
    out = {}
    for model in sorted({r["model"] for r in raw}):
        pairs = [(by[(h, m, 1)], r) for (h, m, k), r in by.items() if m == model and k == 2 and (h, m, 1) in by]
        if not pairs:
            continue
        sig = lambda r: [(c["name"], c["arguments"]) for c in r["tool_calls"]]
        out[model] = dict(n=len(pairs), same_action=sum(a["action"] == b["action"] for a, b in pairs) / len(pairs),
                          same_calls=sum(sig(a) == sig(b) for a, b in pairs) / len(pairs),
                          same_answer=sum(a["answer"] == b["answer"] for a, b in pairs) / len(pairs))
    return out


# ---------------------------------------------------------------- çalıştırma
@contextmanager
def local_server(url: str = None, port: int = PORT):
    """mlx_lm.server'ı adaptörsüz başlatır (adaptör istekle gelir) ve iş bitince kapatır; url verilirse onu kullanır."""
    if url:
        yield url
        return
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(SERVER_LOG, "w")
    p = subprocess.Popen(["venv-llm/bin/mlx_lm.server", "--model", LOCAL_MODEL, "--port", str(port)], stdout=log,
                         stderr=subprocess.STDOUT, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    try:
        for _ in range(300):
            if p.poll() is not None:
                raise RuntimeError(f"model sunucusu başlamadı, bkz. {SERVER_LOG}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("model sunucusu 300 sn içinde açılmadı")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


def repeat_subset(qs: list, n: int = REPEAT_N) -> list:
    """Kategorilerden sırayla: s1, s2, ..., error, s1, s2, ... (ilk n)."""
    by = defaultdict(list)
    for q in qs:
        by[q["cat"]].append(q)
    out = []
    while len(out) < n and any(by.values()):
        for c in CATS:
            if by[c] and len(out) < n:
                out.append(by[c].pop(0))
    return out


def run_one(q: dict, llm, label: str, run: int, db: Path) -> dict:
    """Bir soruyu soru modunda çalıştırır (günlüğe yazar) ve puanlamaya yetecek kaydı döndürür."""
    res = QM.run_question(q["question"], q["asof"], llm, db=db, log=True)
    first = (QM.validate_answer(res["answer_raw"], q["question"], res["tool_calls"]) if res["answer_raw"] is not None
             else dict(numbers=dict(n_numbers=0, unsupported=[]), advice=[]))
    return dict(hid=q["hid"], cat=q["cat"], model=label, run=run, qid=res["qid"], model_name=res["model"],
                max_tokens=getattr(llm, "max_tokens", None),
                action=res["action"], status=res["status"], action_source=res["action_source"], retries=res["retries"],
                llm_calls=res["llm_calls"], latency=res["latency"], error=res["error"],
                tool_calls=[dict(name=o["name"], arguments=o["arguments"], args=o.get("args"), ok=o["ok"],
                                 error=(o.get("error") or "")[:300], over_budget=bool(o.get("over_budget")))
                            for o in res["tool_calls"]],
                answer=res["answer"], answer_raw=res["answer_raw"],
                first_numbers=first["numbers"]["n_numbers"], first_unsupported=first["numbers"]["unsupported"],
                first_advice=first["advice"], final_advice=QM.advice_hits(res["answer"]) if res["status"] == "ok" else [])


def _jsonl(path: Path) -> list:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def _append(path: Path, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(url: str = None, models=("base", "v1"), repeat_n: int = REPEAT_N, limit: int = None, out: Path = OUT):
    qs = load_heldout(out / HELDOUT.name)[:limit]
    raw_path, db = out / "raw.jsonl", out / "question_log.sqlite"
    done = {(r["hid"], r["model"], r["run"]) for r in _jsonl(raw_path)}
    plan = [(m, 1, qs) for m in models] + [(m, 2, repeat_subset(qs, repeat_n)) for m in reversed(models)]
    todo = sum((q["hid"], m, k) not in done for m, k, items in plan for q in items)
    print(f"{todo} çalıştırma kaldı ({len(qs)} soru, modeller {', '.join(models)}, tekrar {repeat_n})", flush=True)
    if todo:
        with local_server(url) as base_url:
            for label, k, items in plan:
                llm = LocalLLM(base_url=base_url, adapter=MODELS[label])
                for q in items:
                    if (q["hid"], label, k) in done:
                        continue
                    rec = run_one(q, llm, label, k, db)
                    _append(raw_path, rec)
                    print(f"{label} #{k} {q['hid']} {q['cat']:5s} altın {q['gold_action']:<33s} model "
                          f"{rec['action']:<33s} {'✓' if rec['action'] == q['gold_action'] else '✗'} "
                          f"{rec['latency']:5.1f} sn", flush=True)
    replay_all(out)


def replay_all(out: Path = OUT):
    """4a: 1. çalıştırmaların kayıtlı araç çağrıları aynı tavanla yeniden çalışır; çıktılar aynı mı?"""
    path, db = out / "replay.jsonl", out / "question_log.sqlite"
    done = {r["qid"] for r in _jsonl(path)}
    for r in _jsonl(out / "raw.jsonl"):
        if r["run"] == 1 and r["qid"] not in done:
            rp = QM.replay_question(r["qid"], db)
            _append(path, dict(qid=r["qid"], hid=r["hid"], model=r["model"], n_calls=rp["n_calls"],
                               same_outputs=rp["same_outputs"]))


# ---------------------------------------------------------------- özet
def summarize(out: Path = OUT) -> dict:
    qs = {q["hid"]: q for q in load_heldout(out / HELDOUT.name)}
    raw = _jsonl(out / "raw.jsonl")
    rows = [score(qs[r["hid"]], r) for r in raw]
    first = [x for x in rows if x["run"] == 1]
    models = [m for m in MODELS if any(x["model"] == m for x in first)]
    overall, per_cat, confusion = {}, {}, {}
    for m in models:
        mine = [x for x in first if x["model"] == m]
        overall[m] = aggregate(mine)
        per_cat[m] = {c: aggregate([x for x in mine if x["cat"] == c]) for c in CATS if any(x["cat"] == c for x in mine)}
        confusion[m] = Counter((x["gold_action"], x["action"]) for x in mine if not x["correct"])
    rp = _jsonl(out / "replay.jsonl")
    replay = {m: sum(r["same_outputs"] for r in rp if r["model"] == m) / max(sum(r["model"] == m for r in rp), 1)
              for m in models}
    if rows:
        with open(out / "per_question.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["question"])
            w.writeheader()
            for x in sorted(rows, key=lambda x: (x["hid"], x["model"], x["run"])):
                w.writerow(dict(x, question=qs[x["hid"]]["question"]))
    res = dict(overall=overall, per_category=per_cat, confusion=confusion, reproducibility=reproducibility(raw),
               replay_same_outputs=replay, max_tokens=sorted({r.get("max_tokens") for r in raw} - {None}))
    (out / "summary.md").write_text(render(res, list(qs.values()), models))
    return res


def _f(v, pct=True):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v * 100:.1f}%" if pct else f"{v:.1f}"
    return str(v)


def render(res: dict, qs: list, models: list) -> str:
    """summary.md (İngilizce, raporlar gibi)."""
    o = res["overall"]
    head = "| metric | " + " | ".join(models) + " |\n|---|" + "---|" * len(models)
    line = lambda name, k, pct=True: f"| {name} | " + " | ".join(_f(o[m][k], pct) for m in models) + " |"
    lines = [
        "# Local model evaluation: base vs fine-tuned (v1) on held-out questions", "",
        f"Generated by `python -m src.finetune.evaluate`. Model `{LOCAL_MODEL}` (4-bit), `base` = no adapter, `v1` = "
        f"LoRA adapter `{LOCAL_ADAPTER}`. Same settings as the live question mode: temperature 0, at most "
        f"{QM.MAX_CALLS} tool calls, one correction round, max_tokens {', '.join(map(str, res['max_tokens'])) or '–'}.",
        "",
        f"**Held-out set** (`src/finetune/heldout.py`, `heldout.jsonl`): {len(qs)} questions written with phrasings, "
        "terms and a date format not used in training; no question shares category + stock/group + analysis date "
        "with any training or validation conversation. Gold tool calls and gold actions come from the same code "
        "and rules that produced the training data, so this measures whether the model applies the rules to new "
        "wording, not whether the rules are economically right. The instructor's five sample questions are in the "
        "training set and therefore not used here.", "",
        "| category | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS),
        "| questions | " + " | ".join(str(sum(q["cat"] == c for q in qs)) for c in CATS) + " |", "",
        "Gold actions: " + ", ".join(f"{a} {n}" for a, n in Counter(q["gold_action"] for q in qs).most_common()) + ".",
        "", "## Results (first run)", "", head,
        line("action accuracy", "action_accuracy"),
        line("unsupported conclusion rate", "unsupported_conclusion"),
        line("invalid tool-call rate", "invalid_call_rate"),
        line("evidence completeness (gold calls matched)", "evidence_completeness"),
        line("questions with all gold calls", "full_evidence"),
        line("backtest call correctness", "backtest_call_correct"),
        line("fabricated numbers (first answer)", "fabricated_number_rate"),
        line("answers with a fabricated number (first answer)", "answers_with_fabrication"),
        line("advice/target price in first answer", "advice_first"),
        line("advice/target price shown to user", "advice_shown"),
        line("answers withheld by the gate", "withheld"),
        line("no successful tool call", "no_evidence"),
        line("model/server errors", "llm_error"),
        line("tool calls per question", "tool_calls", False),
        line("LLM calls per question", "llm_calls", False),
        line("latency mean (s)", "latency_mean", False),
        line("latency median (s)", "latency_median", False),
        "", "Tool calls match when the tool name and every key argument of the gold call agree (stock/sector, "
        "event/pattern/strategy, horizon/lookback, conditions, dates stated in the question); extra valid arguments "
        "are allowed. Calls with the unknown code in error questions are expected to fail and are not counted as "
        "invalid.", "", "## By category (action accuracy / invalid tool-call rate / evidence completeness)", "",
        "| category | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for c in CATS:
        cells = []
        for m in models:
            a = res["per_category"][m].get(c)
            cells.append("–" if a is None else f"{_f(a['action_accuracy'])} / {_f(a['invalid_call_rate'])} / "
                                               f"{_f(a['evidence_completeness'])}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Wrong actions (gold → model: count)", ""]
    for m in models:
        wrong = res["confusion"][m].most_common()
        lines.append(f"- **{m}**: " + ("; ".join(f"{g} → {a}: {n}" for (g, a), n in wrong) if wrong else "none"))
    lines += ["", "## Reproducibility", ""]
    for m in models:
        r = res["reproducibility"].get(m)
        if r:
            lines.append(f"- **{m}**: {r['n']} questions run twice: same action {_f(r['same_action'])}, same tool "
                         f"calls {_f(r['same_calls'])}, same answer text {_f(r['same_answer'])}.")
        lines.append(f"- **{m}**: logged tool calls replayed under the same ceiling: same outputs "
                     f"{_f(res['replay_same_outputs'].get(m))}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Taban ve ince ayarlı model, görülmemiş sorular (5. adım)")
    ap.add_argument("--url", default=None, help="çalışan sunucu (adaptörsüz başlatılmış); yoksa sunucu başlatılır")
    ap.add_argument("--models", default="base,v1")
    ap.add_argument("--repeat", type=int, default=REPEAT_N)
    ap.add_argument("--limit", type=int, default=None, help="ilk N soru (hızlı deneme)")
    ap.add_argument("--summary", action="store_true", help="yalnızca özeti yeniden yaz")
    a = ap.parse_args()
    if not a.summary:
        run(a.url, tuple(a.models.split(",")), a.repeat, a.limit)
    s = summarize()
    print(json.dumps(s["overall"], ensure_ascii=False, indent=1))
    print("özet: reports/local_llm/eval_v1/summary.md")
