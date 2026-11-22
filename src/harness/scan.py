"""Evren tarama modu (PDF §33: "güçlü bir sektörün gerisinde kalan ama tarihsel olarak olumlu bir yakalama
durumuna giriyor olabilecek BIST 100 hisselerini bul").

Akış (Adım 4 kararları, 22 Kasım 2026):
  1. Kural katmanı (evidence.run_analysis, LLM yok) analiz günü 30 hissenin hepsine uygulanır: koşullar, kapı, eylem.
  2. Sorulan eylemin adayları listelenir; aday yoksa ya da azsa "en yakınlar" ve takıldıkları koşul gösterilir.
  3. En fazla `top` aday tam harness'tan geçer (LLM açıklaması, insan incelemesi, kayıt). Yakalama sorusunda
     sıra sektör gecikmesine göredir (en geride olan önce); bu bir tahmin değildir, R2'de sınanmadı.
  4. Tarama tablosu veri sürümleriyle `scans` tablosuna yazılır; replay_scan aynı taramayı yeniden hesaplar.

Çalıştırma:
  python -m src.harness.scan 2026-11-06                                   # yalnız tarama (LLM yok, anahtar yok)
  python -m src.harness.scan 2026-11-06 --action TECHNICAL_REVERSAL_CANDIDATE
  python -m src.harness.scan 2026-11-06 --run --mock --review ACCEPT --reason "demo"   # adaylar sahte modelle
  python -m src.harness.scan 2026-11-06 --run                             # adaylar Gemini ile (kullanıcı çalıştırır)
  python -m src.harness.scan --replay <scan_id>
"""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd

from src.data.universe import load_universe
from src.harness.decision_log import (DB, DEMO_DB, REVIEW_DECISIONS, connect, data_versions, dumps,
                                      procedure_version)
from src.harness.evidence import ACTIONS, CANDIDATES, analysis_day, run_analysis

CATCH_UP = "POTENTIAL_CATCH_UP_CANDIDATE"
TOP_N = 3
OUT_DIR = Path("reports/scans")
COND_COLS = ["laggard", "sector_strong", "recovering", "weak_earnings", "catch_up", "fundamental", "technical",
             "tech_conflict", "shock", "negative_news"]
SCAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id TEXT PRIMARY KEY, created_at TEXT, asof TEXT, action TEXT, top_n INTEGER,
    data_versions TEXT, procedure_version TEXT, rows TEXT, candidates TEXT, also_secondary TEXT,
    near_misses TEXT, selected TEXT, analysis_ids TEXT, fingerprint TEXT);
"""


def _core() -> list:
    u = load_universe()
    return u[u["is_core"] == 1]["ticker"].tolist()


def _why_not_catch_up(r: dict) -> list:
    """Geride kalan ve sektörü güçlü bir hissenin yakalama adayı olamamasının nedenleri."""
    out = []
    if not r["recovering"]:
        out.append("toparlanmıyor (KAMA altında ve Supertrend aşağı)")
    if r["weak_earnings"]:
        out.append("son bilanço alt üçte birde")
    if r["shock"]:
        out.append("şok")
    if r["negative_news"]:
        out.append("olumsuz KAP haberi")
    if r["gate"] in ("HARD_BLOCK", "ANALYSIS_UNSAFE"):
        out.append(f"kapı {r['gate']}")
    return out


def _why_secondary(r: dict) -> list:
    """Sorulan etiketin neden ikincil kaldığı (evidence.decide sırası: kapı -> engelleyici -> çelişki -> öncelik)."""
    out = []
    if r["gate"] in ("HARD_BLOCK", "ANALYSIS_UNSAFE", "INVESTIGATE"):
        out.append(f"kapı {r['gate']}")
    if r["shock"]:
        out.append("şok")
    if r["negative_news"]:
        out.append("olumsuz KAP haberi")
    if r["tech_conflict"]:
        out.append("çelişen teknik sinyal (aynı 3 günde boğa ve ayı olay)")
    if not out:
        out.append(f"daha öncelikli etiket: {r['action']}")
    return out


def scan_universe(asof, action: str = CATCH_UP, tickers: list = None) -> dict:
    """Kural katmanını evrene uygular. Her hisse için yalnızca analiz günü bilinen veri kullanılır (run_analysis)."""
    if action not in CANDIDATES:
        raise ValueError(f"taranabilen eylemler: {CANDIDATES}")
    day = analysis_day(asof)
    rows = []
    for t in tickers or _core():
        r = run_analysis(t, day)
        s, c = r["observed"]["sector"], r["conditions"]
        rows.append(dict(ticker=t, peer_group=r["meta"].get("sector"), bist100=(r["meta"].get("bist100") or {}).get("member"),
                         action=r["action"],
                         secondary=list(r["secondary"]), gate=r["gate"]["status"],
                         sector_lag_10=s.get("sector_lag_10"), sector_median_10=s.get("sector_median_10"),
                         **{k: bool(c[k]) for k in COND_COLS}))
    return dict(asof=str(day.date()), action=action, rows=rows, **classify(rows, action))


def classify(rows: list, action: str) -> dict:
    """Tarama satırlarından adaylar, ikincil kalanlar ve (yakalama sorusunda) en yakınlar."""
    cands = [r["ticker"] for r in rows if r["action"] == action]
    # sorulan eylem ikincil kalmış: engelleyici (REJECT_SIGNAL / INVESTIGATE) ya da daha öncelikli başka bir aday
    also = [dict(ticker=r["ticker"], action=r["action"], why=_why_secondary(r)) for r in rows
            if action in r["secondary"]]
    also_t = {x["ticker"] for x in also}
    near = []
    if action == CATCH_UP:
        for r in rows:
            if r["laggard"] and r["sector_strong"] and r["ticker"] not in cands and r["ticker"] not in also_t:
                near.append(dict(ticker=r["ticker"], sector_lag_10=r["sector_lag_10"], why=_why_not_catch_up(r)))
        near.sort(key=lambda x: x["sector_lag_10"])
    return dict(candidates=cands, also_secondary=also, near_misses=near)


def from_csv(path) -> dict:
    """Kaydedilmiş tarama tablosundan tarama (rapor üretimi için; yeniden hesaplamaz)."""
    d = pd.read_csv(path)
    d["secondary"] = d["secondary"].fillna("").apply(lambda x: [v.strip() for v in x.split(",") if v.strip()])
    rows = d.to_dict("records")
    asof, action = Path(path).stem.split("_", 2)[1:]
    return dict(asof=asof, action=action, rows=rows, **classify(rows, action))


def select(scan: dict, top: int = TOP_N) -> list:
    """Tam harness'a girecek adaylar. Yakalama: en büyük sektör gecikmesi önce; diğer eylemler: alfabetik."""
    rows = {r["ticker"]: r for r in scan["rows"]}
    c = list(scan["candidates"])
    if scan["action"] == CATCH_UP:
        c.sort(key=lambda t: rows[t]["sector_lag_10"])
    else:
        c.sort()
    return c[:top]


def scan_fingerprint(scan: dict) -> str:
    core = {k: scan[k] for k in ("asof", "action", "rows", "candidates", "also_secondary", "near_misses")}
    return hashlib.sha256(dumps(core).encode()).hexdigest()[:16]


def _connect(db: Path):
    con = connect(db)
    con.executescript(SCAN_SCHEMA)
    return con


def log_scan(scan: dict, selected: list, top: int, db: Path = DB) -> str:
    sid = uuid.uuid4().hex[:12]
    row = dict(scan_id=sid, created_at=str(pd.Timestamp.now().floor("s")), asof=scan["asof"], action=scan["action"],
               top_n=top, data_versions=dumps(data_versions()), procedure_version=procedure_version(),
               rows=dumps(scan["rows"]), candidates=dumps(scan["candidates"]),
               also_secondary=dumps(scan["also_secondary"]), near_misses=dumps(scan["near_misses"]),
               selected=dumps(selected), analysis_ids=dumps([]), fingerprint=scan_fingerprint(scan))
    with _connect(db) as con:
        con.execute(f"INSERT INTO scans ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    return sid


def link_analyses(scan_id: str, analysis_ids: list, db: Path = DB) -> None:
    """Tam harness'tan geçen adayların karar kayıtlarını taramaya bağlar."""
    with _connect(db) as con:
        con.execute("UPDATE scans SET analysis_ids = ? WHERE scan_id = ?", (dumps(analysis_ids), scan_id))


def load_scan(scan_id: str, db: Path = DB) -> dict:
    with _connect(db) as con:
        con.row_factory = lambda cur, r: {d[0]: v for d, v in zip(cur.description, r)}
        row = con.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
    if row is None:
        raise ValueError(f"tarama bulunamadı: {scan_id}")
    for k in ("data_versions", "rows", "candidates", "also_secondary", "near_misses", "selected", "analysis_ids"):
        row[k] = json.loads(row[k])
    return row


def replay_scan(scan_id: str, db: Path = DB) -> dict:
    """Kayıtlı taramayı aynı gün ve soruyla yeniden hesaplar; aday listesi ve satırlar aynı mı?"""
    old = load_scan(scan_id, db)
    new = scan_universe(old["asof"], old["action"], tickers=[r["ticker"] for r in old["rows"]])
    before = {r["ticker"]: r for r in old["rows"]}
    changed = [r["ticker"] for r in new["rows"] if dumps(r) != dumps(before.get(r["ticker"]))]
    return dict(scan_id=scan_id, same=scan_fingerprint(new) == old["fingerprint"],
                stored_candidates=old["candidates"], new_candidates=new["candidates"], changed_tickers=changed,
                changed_data_files=[f for f, h in data_versions().items() if old["data_versions"].get(f) != h],
                procedure_changed=procedure_version() != old["procedure_version"])


def scan_table(scan: dict) -> pd.DataFrame:
    d = pd.DataFrame(scan["rows"])
    d["secondary"] = d["secondary"].apply(", ".join)
    return d


def save_csv(scan: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"scan_{scan['asof']}_{scan['action']}.csv"
    scan_table(scan).to_csv(p, index=False)
    return p


def print_scan(scan: dict, selected: list) -> None:
    pd.set_option("display.width", 220)
    d = scan_table(scan)
    print(f"\nTarama {scan['asof']} | soru: {scan['action']} | {len(d)} hisse")
    print("Eylem dağılımı:", d["action"].value_counts().to_dict())
    print(f"\nAdaylar ({len(scan['candidates'])}): {', '.join(scan['candidates']) or 'yok'}")
    if scan["also_secondary"]:
        for x in scan["also_secondary"]:
            print(f"  ikincil kaldı: {x['ticker']} -> {x['action']}: {'; '.join(x['why'])}")
    for x in scan["near_misses"]:
        print(f"  en yakın: {x['ticker']} (sektör gecikmesi {x['sector_lag_10']:+.1%}): {'; '.join(x['why'])}")
    print(f"Tam harness'a girecek: {', '.join(selected) or 'yok'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evren tarama modu (PDF §33)")
    ap.add_argument("asof", nargs="?")
    ap.add_argument("--action", default=CATCH_UP, choices=CANDIDATES)
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--run", action="store_true", help="seçilen adayları tam harness'tan geçir")
    ap.add_argument("--mock", action="store_true", help="Gemini yerine sahte model (anahtar kullanılmaz)")
    ap.add_argument("--review", choices=REVIEW_DECISIONS)
    ap.add_argument("--reason", default="")
    ap.add_argument("--final", choices=ACTIONS)
    ap.add_argument("--replay", metavar="SCAN_ID")
    ap.add_argument("--demo-db", action="store_true", help="kaydı örnek günlüğe yaz")
    a = ap.parse_args()
    db = DEMO_DB if (a.mock or a.demo_db) else DB          # sahte model kayıtları gerçek günlüğe karışmaz
    if a.replay:
        print(json.dumps(replay_scan(a.replay, db), ensure_ascii=False, indent=2))
        raise SystemExit
    if not a.asof:
        ap.error("analiz günü gerekli")
    sc = scan_universe(a.asof, a.action)
    sel = select(sc, a.top)
    print_scan(sc, sel)
    print(f"tablo: {save_csv(sc)}")
    sid = log_scan(sc, sel, a.top, db)
    print(f"scan_id: {sid} ({db})")
    if a.run and sel:
        from src.agents.llm import MockLLM
        from src.harness.state_machine import run_harness
        if a.mock:
            llm = MockLLM()
        else:
            from src.agents.llm import GeminiLLM
            llm = GeminiLLM()
        review = dict(decision=a.review, reason=a.reason, final_action=a.final) if a.review else None
        ids = []
        for t in sel:
            ctx, trace = run_harness(t, sc["asof"], llm, review, db=db)
            ids.append(ctx.get("analysis_id"))
            print(f"{t}: {ctx['record']['action']} -> insan {ctx['review']['decision']} | analysis_id {ids[-1]}")
        link_analyses(sid, ids, db)
