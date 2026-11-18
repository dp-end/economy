"""Karar günlüğü, tekrar oynatma ve hafıza (PDF Bölüm 24, 27, 28). SQLite: reports/decision_log.sqlite.

Hafıza türleri:
  temporal   : fiyat, gösterge, makro, bilanço, olay geçmişi -> data/ ve reports/ dosyaları (veri sürümü hash'i kaydedilir)
  episodic   : geçmiş vakalar (durum -> kanıt -> eylem -> backtest -> insan kararı) -> analyses + human_reviews tabloları
  procedural : onaylı iş akışı sırası, eylem kümesi, kapı çıktıları, doğrulama kuralları -> PROCEDURE (sürüm hash'iyle)
Sistem hiçbir zaman AI sonucunu sessizce emre çevirmez: insan kararı ayrı tabloda, gerekçesiyle tutulur.
"""
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.harness.evidence import ACTIONS, run_analysis
from src.harness.quality_gate import GATE_OUTPUTS

DB = Path("reports/decision_log.sqlite")
DEMO_DB = Path("reports/decision_log_demo.sqlite")      # örnek kayıtlar gerçek günlüğe karışmaz
REVIEW_DECISIONS = ("ACCEPT", "MODIFY", "REJECT")
DATA_FILES = ("reports/technical_events.parquet", "reports/sector_events.parquet", "reports/earnings_events.parquet",
              "reports/kap_disclosure_dates.csv", "reports/news_events.parquet", "reports/speech_features.parquet",
              "reports/macro_panel.csv", "config/universe.csv")

PROCEDURE = dict(
    workflow=["START", "SELECT_UNIVERSE", "LOAD_MARKET_DATA", "CHECK_DATA_QUALITY", "CALCULATE_TECHNICALS",
              "RUN_SECTOR_RELATIVE_ANALYSIS", "RUN_PATTERN_AND_REVERSAL_ANALYSIS", "LOAD_FUNDAMENTALS",
              "LOAD_MACRO_CONTEXT", "LOAD_NEWS_AND_SPEECH_CONTEXT", "BUILD_EVIDENCE", "PREVIEW_STRATEGIES",
              "RUN_BACKTEST", "RISK_GATE", "GENERATE_EXPLANATION", "HUMAN_REVIEW", "LOG_AND_CLOSE"],
    actions=list(ACTIONS), gate_outputs=list(GATE_OUTPUTS), review_decisions=list(REVIEW_DECISIONS),
    rules=["Analiz günü kapanışında bilinmeyen veri kullanılmaz (event_day / available_date <= analiz günü).",
           "Eylem kurallarla seçilir; LLM yalnızca committed evidence'a dayanarak açıklar.",
           "HARD_BLOCK veya ANALYSIS_UNSAFE kapısında eylem ANALYSIS_UNSAFE olur.",
           "Şok veya olumsuz KAP haberi olan aday REJECT_SIGNAL olur.",
           "Kalite kapısı ve backtest durumları atlanamaz.",
           "İnsan kararı olmadan hiçbir sonuç gerçek işleme dönüşmez; broker bağlantısı yoktur."],
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY, created_at TEXT, ticker TEXT, sector TEXT, asof TEXT,
    data_start TEXT, data_end TEXT, data_versions TEXT, procedure_version TEXT,
    tools_called TEXT, observed TEXT, history TEXT, backtest TEXT, conditions TEXT,
    gate_status TEXT, gate TEXT, action TEXT, secondary TEXT, committed TEXT,
    explanation TEXT, fingerprint TEXT);
CREATE TABLE IF NOT EXISTS human_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT, reviewed_at TEXT, reviewer TEXT,
    decision TEXT, final_action TEXT, reason TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id));
CREATE TABLE IF NOT EXISTS harness_traces (
    analysis_id TEXT PRIMARY KEY, created_at TEXT, llm TEXT, summary TEXT, trace TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id));
"""


def _json_default(o):
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NaT:
        return None
    return str(o)


def dumps(x) -> str:
    return json.dumps(x, default=_json_default, ensure_ascii=False, sort_keys=True)


def file_hash(path: str) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "yok"


def data_versions() -> dict:
    return {f: file_hash(f) for f in DATA_FILES}


def procedure_version() -> str:
    return hashlib.sha256(dumps(PROCEDURE).encode()).hexdigest()[:12]


def fingerprint(rec: dict) -> str:
    """Analizin içerik özeti: aynı veriyle tekrar çalıştırınca aynı çıkmalı (tekrarlanabilirlik)."""
    core = {k: rec.get(k) for k in ("observed", "history", "backtest", "conditions", "gate", "action",
                                     "secondary", "committed")}
    return hashlib.sha256(dumps(core).encode()).hexdigest()[:16]


def connect(db: Path = DB) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    return con


def log_decision(rec: dict, tools_called=None, explanation: str = "", db: Path = DB) -> str:
    """Bir analizi kaydeder; analysis_id döner."""
    aid = uuid.uuid4().hex[:12]
    m = rec["meta"]
    row = dict(analysis_id=aid, created_at=str(pd.Timestamp.now().floor("s")), ticker=m["ticker"],
               sector=m.get("sector"), asof=str(pd.Timestamp(m["asof"]).date()),
               data_start=str(rec.get("backtest", {}).get("start") or ""), data_end=str(m["asof"]),
               data_versions=dumps(data_versions()), procedure_version=procedure_version(),
               tools_called=dumps(tools_called or ["build_evidence_bundle", "check_data_quality", "run_backtest"]),
               observed=dumps(rec.get("observed")), history=dumps(rec.get("history")),
               backtest=dumps(rec.get("backtest")), conditions=dumps(rec.get("conditions")),
               gate_status=rec["gate"]["status"], gate=dumps(rec["gate"]), action=rec["action"],
               secondary=dumps(rec.get("secondary")), committed=dumps(rec.get("committed")),
               explanation=explanation, fingerprint=fingerprint(rec))
    with connect(db) as con:
        con.execute(f"INSERT INTO analyses ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    return aid


def record_review(analysis_id: str, decision: str, reason: str, final_action: str = None,
                  reviewer: str = "insan", db: Path = DB) -> None:
    """İnsan incelemesi: ACCEPT / MODIFY (final_action ile) / REJECT, gerekçe zorunlu."""
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"geçersiz karar: {decision} (izinli: {REVIEW_DECISIONS})")
    if not reason or not reason.strip():
        raise ValueError("gerekçe zorunlu")
    with connect(db) as con:
        row = con.execute("SELECT action FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
        if row is None:
            raise ValueError(f"analiz bulunamadı: {analysis_id}")
        if decision == "MODIFY":
            if final_action not in ACTIONS:
                raise ValueError(f"MODIFY için geçerli eylem gerekli: {final_action}")
        else:
            final_action = row[0] if decision == "ACCEPT" else "REJECT_SIGNAL"
        con.execute("INSERT INTO human_reviews (analysis_id, reviewed_at, reviewer, decision, final_action, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (analysis_id, str(pd.Timestamp.now().floor("s")), reviewer, decision, final_action, reason))


def log_trace(analysis_id: str, llm: str, summary: dict, trace: dict, db: Path = DB) -> None:
    """Harness izi: durum sırası, araç çağrıları (izinsizler dahil), ajan notları ve sayı kontrolleri."""
    with connect(db) as con:
        con.execute("INSERT OR REPLACE INTO harness_traces VALUES (?, ?, ?, ?, ?)",
                    (analysis_id, str(pd.Timestamp.now().floor("s")), llm, dumps(summary), dumps(trace)))


def load_trace(analysis_id: str, db: Path = DB) -> dict:
    with connect(db) as con:
        row = con.execute("SELECT llm, summary, trace FROM harness_traces WHERE analysis_id = ?",
                          (analysis_id,)).fetchone()
    if row is None:
        raise ValueError(f"iz bulunamadı: {analysis_id}")
    return dict(llm=row[0], summary=json.loads(row[1]), trace=json.loads(row[2]))


def load_analysis(analysis_id: str, db: Path = DB) -> dict:
    with connect(db) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
    if row is None:
        raise ValueError(f"analiz bulunamadı: {analysis_id}")
    out = dict(row)
    for k in ("data_versions", "tools_called", "observed", "history", "backtest", "conditions", "gate",
              "secondary", "committed"):
        out[k] = json.loads(out[k]) if out[k] else None
    return out


def replay_analysis(analysis_id: str, db: Path = DB) -> dict:
    """Kayıtlı analizi aynı hisse ve günle yeniden çalıştırır; sonuç, kanıt ve kapı aynı mı?"""
    old = load_analysis(analysis_id, db)
    new = run_analysis(old["ticker"], old["asof"])
    changed_data = [f for f, h in data_versions().items() if old["data_versions"].get(f) != h]
    fp = fingerprint(new)
    return dict(analysis_id=analysis_id, same=fp == old["fingerprint"], stored_fingerprint=old["fingerprint"],
                new_fingerprint=fp, stored_action=old["action"], new_action=new["action"],
                stored_gate=old["gate_status"], new_gate=new["gate"]["status"],
                changed_data_files=changed_data, procedure_changed=procedure_version() != old["procedure_version"])


def past_cases(ticker: str = None, action: str = None, db: Path = DB) -> pd.DataFrame:
    """Episodik hafıza: geçmiş vakalar ve varsa insan kararı."""
    q = ("SELECT a.analysis_id, a.created_at, a.ticker, a.asof, a.action, a.gate_status, "
         "r.decision, r.final_action, r.reason FROM analyses a "
         "LEFT JOIN human_reviews r ON a.analysis_id = r.analysis_id WHERE 1=1")
    params = []
    if ticker:
        q, params = q + " AND a.ticker = ?", params + [ticker]
    if action:
        q, params = q + " AND a.action = ?", params + [action]
    with connect(db) as con:
        return pd.read_sql_query(q + " ORDER BY a.created_at", con, params=params)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    DEMO_DB.unlink(missing_ok=True)
    print(f"Demo kayıtları {DEMO_DB} dosyasına yazılır (gerçek günlük {DB} değişmez).")
    demo = [("ASELS", "2026-11-06", "ACCEPT", "Bilanço ve sözleşme haberleri tutarlı; backtest al-tut'u geçmiyor, sadece izlenecek.", None),
            ("TURSG", "2026-11-06", "ACCEPT", "Şok ve yönetim ayrılıkları var; inceleme gerekiyor.", None),
            ("AKBNK", "2026-11-06", "MODIFY", "Teknik olayların geçmiş başarısı %50'nin altında; aday yerine izle.", "WATCH"),
            ("ASELS", "2023-02-15", "ACCEPT", "Deprem nedeniyle piyasa kapalı; analiz yapılamaz.", None)]
    ids = []
    for t, d, dec, reason, final in demo:
        rec = run_analysis(t, d)
        aid = log_decision(rec, db=DEMO_DB)
        record_review(aid, dec, reason, final, reviewer="demo", db=DEMO_DB)
        ids.append(aid)
        print(f"{t} {d}: {rec['action']} (kapı {rec['gate']['status']}) -> insan: {dec} | id {aid}")
    print("\nTekrar oynatma (replay):")
    for aid in ids:
        r = replay_analysis(aid, db=DEMO_DB)
        print(f"  {aid}: aynı={r['same']}  eylem {r['stored_action']} -> {r['new_action']}  "
              f"değişen veri={r['changed_data_files']}")
    print("\nEpizodik hafıza (geçmiş vakalar):")
    print(past_cases(db=DEMO_DB).tail(8).to_string(index=False))
