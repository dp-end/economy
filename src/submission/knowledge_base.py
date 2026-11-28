"""Teslim: RAG dosyası (sunum duyurusu, 28 Kasım 2026) -> submission/rag/knowledge_base.json.

Duyuru: "RAG dosyası: Bilgi tabanı belgeleri ve kaynak meta verileri (JSON)". Kullanıcı kararı (28 Kasım 2026,
seçenek A): sistemde gömme (embedding) tabanlı arama yok; model bilgiye MCP araçlarıyla ulaşır ve araçlar analiz
tarihi tavanını uygular. Bu dosya o araçların dayandığı belgeleri ve kaynak meta verilerini tek yerde toplar;
modelin ya da kodun davranışını değiştirmez.

Gruplar (kullanıcının onayladığı liste):
  metin belgeleri      1 KAP bildirimleri, 2 RSS haber başlıkları, 3 konuşma bölümleri, 4 konuşmadaki sayısal
                       iddialar ve doğrulama sonucu, 5 konuşma kaynak kayıtları
  kural ve tanımlar    6 aksiyon kuralları + 8 aksiyonun tanımı (sistem istemi), 7 MCP araç tanımları,
                       8 veri sözlüğü bölümleri
  yapısal kaynaklar    9 fiyatlar, 10 bilançolar, 11 bilanço yayın tarihleri, 12 makro seriler, 13 Fed faiz
                       aralığı, 14 proje PDF'i (yalnızca künye) -- sayılar dosyaya girmez, dosya özetleri girer
Dışarıda: kendi araştırma raporlarımız (sistemin çıktısı), eğitim verisi (manifestte ayrı), Fed ses dosyası.

Çalıştırma:  python -m src.submission.knowledge_base [--pdf <proje PDF'inin yolu>]
"""
import argparse
import asyncio
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from mcp import Client

from src.agents.llm import sha256_file
from src.data.macro import EVDS_SERIES, FED_TABLE
from src.data.speech import SOURCES
from src.harness import question_mode as QM
from src.harness.evidence import speech_known_day
from src.mcp_server.server import build_server

OUT = Path("submission/rag/knowledge_base.json")
VERSION = 1
NEWS = Path("reports/news_events.parquet")
KAP_ODA = Path("data/kap_oda")
KAP_FR = Path("data/kap")
SPEECH = Path("reports/speech_features.parquet")
DICTIONARY = Path("reports/research/00_data_dictionary.md")
SPEC_PDF = Path.home() / "Okul/Ekonomi/ProjeBelgeleri/Engineering Economics Project_2026_CSE-481.pdf"
KAP_URL = re.compile(r"/Bildirim/(\d+)")

USAGE = {
    "kap": ("Kamuyu Aydınlatma Platformu'nda herkese açık bildirim; yalnızca başlık, özet, kategori ve bağlantı "
            "saklanır, ek dosyalar indirilmedi."),
    "rss": ("Yayıncının herkese açık RSS akışı; yalnızca başlık ve bağlantı saklanır, haber metni indirilmedi "
            "(ücret duvarı ya da giriş aşılmadı)."),
    "fed_20261028": ("Federal Reserve basın toplantısı (ABD kamu belgesi). Ses yerel Whisper ile metne çevrildi; "
                     "ses dosyası teslim edilmez, bağlantısı kayıtta."),
    "tcmb_20261112": ("TCMB'nin herkese açık resmi konuşma metni. YouTube ses indirmeyi bot doğrulamasıyla "
                      "engellediği için ses kullanılmadı (platform koruması aşılmadı)."),
    "internal": "Projenin kendi kodu ve belgeleri.",
}
NOTES = {
    "kap_fetched_at": "KAP kayıtlarında fetched_at, önbellek dosyasının tarihidir (data/kap_oda); tek tek indirme "
                      "zamanı kaydedilmedi.",
    "rss_fetched_at": "RSS kayıtlarında fetched_at, haberin arşive eklendiği zamandır (data/rss/archive.parquet).",
    "kap_multi_ticker": "Birden çok hisseyi ilgilendiren KAP bildirimi hisse başına ayrı kayıttır (doc_id "
                        "kap_<bildirim no>_<hisse>); araç da böyle döndürür.",
    "speech_seconds": "TCMB konuşması resmi metinden alındı (ses yok); bölümlerinde start_seconds/end_seconds "
                      "boştur. Fed bölümlerinde saniyeler Whisper zaman damgalarıdır.",
    "event_day_null": "event_day boş olan haberler fiyat verisinin son gününden sonra yayınlandı; hiçbir analiz "
                      "gününe girmez.",
}
GROUPS = [
    ("kap_disclosure", "metin", "KAP bildirimleri (başlık + özet, kategori etiketleri)", "get_news_events"),
    ("rss_headline", "metin", "RSS ekonomi haber başlıkları (AA, BloombergHT, TRT Haber)", "get_news_events"),
    ("speech_segment", "metin", "Konuşma bölümleri (metin, saniye aralığı, konu, duruş)", "get_speech_features"),
    ("speech_claim", "metin", "Konuşmalardaki sayısal iddialar ve araçla doğrulama sonucu", "get_speech_features"),
    ("speech_source", "metin", "Konuşma kaynak kayıtları (konuşmacı, kanal, bağlantılar)", "get_speech_features"),
    ("action_rule", "kural", "Aksiyon kuralları (soru modu sistem istemi)", "sistem istemi"),
    ("action_definition", "kural", "İzinli 8 aksiyonun tanımı", "sistem istemi"),
    ("tool_definition", "kural", "MCP araç tanımları (modele sunulan biçim)", "araç listesi"),
    ("data_dictionary", "tanım", "Veri sözlüğü bölümleri", "belge (modele verilmez)"),
]


# ---------------------------------------------------------------- yardımcılar
def clean(v):
    """JSON'a uygun değer: NaN/NaT -> None, zaman -> ISO metin, numpy -> Python."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat()
    if hasattr(v, "tolist"):                       # numpy dizi ya da sayı
        return clean(v.tolist())
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    return v


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def doc(doc_id, group, title, text, publisher=None, url=None, publish_time=None, event_day=None, fetched_at=None,
        ticker=None, sector=None, language="tr", tool=None, usage=None, extra=None) -> dict:
    """Bütün gruplarda aynı alanlar; gruba özgü alanlar 'extra' içinde. usage: USAGE anahtarı (not metni dosyanın
    başında bir kez yazılır; 12.905 KAP kaydında tekrarlanınca dosya ~4 MB büyüyordu)."""
    assert usage in USAGE, usage
    return clean(dict(doc_id=doc_id, group=group, title=title, text=text, sha256=text_hash(text), publisher=publisher,
                      url=url, publish_time=publish_time, event_day=event_day, fetched_at=fetched_at, ticker=ticker,
                      sector=sector, language=language, tool=tool, usage=usage, extra=extra or {}))


def file_meta(paths) -> list:
    return [dict(path=str(p), bytes=p.stat().st_size, sha256=sha256_file(p)) for p in sorted(paths)]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- 1-2 haberler
def kap_fetch_times() -> dict:
    """KAP bildirim no -> önbellek dosyasının tarihi (tek tek indirme zamanı kaydedilmedi)."""
    out = {}
    for f in KAP_ODA.glob("*.json"):
        t = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
        for it in json.loads(f.read_text()):
            out[(it.get("disclosureBasic") or {}).get("disclosureIndex")] = t
    return out


def news_docs() -> list:
    ne = pd.read_parquet(NEWS).sort_values(["publish_time", "source", "url"])
    fetched = kap_fetch_times()
    docs = []
    for r in ne.itertuples(index=False):
        extra = dict(category=r.category, categories=r.categories.split("|") if r.categories else [], scope=r.scope)
        if r.source == "KAP":
            m = KAP_URL.search(r.url or "")
            idx = int(m.group(1)) if m else None
            text = f"{r.title}. {r.summary}".strip() if r.summary else r.title
            docs.append(doc(f"kap_{idx}_{r.ticker}", "kap_disclosure", r.title, text, publisher="KAP", url=r.url,
                            publish_time=r.publish_time, event_day=r.event_day, fetched_at=fetched.get(idx),
                            ticker=r.ticker, sector=r.sector, tool="get_news_events", usage="kap",
                            extra=dict(extra, kap_index=idx)))
        else:
            docs.append(doc(f"rss_{text_hash(r.url)[:12]}", "rss_headline", r.title, r.title, publisher=r.source,
                            url=r.url, publish_time=r.publish_time, event_day=r.event_day, fetched_at=r.fetched_at,
                            ticker=r.ticker if isinstance(r.ticker, str) else None,
                            sector=r.sector if isinstance(r.sector, str) else None, tool="get_news_events",
                            usage="rss", extra=extra))
    return docs


# ---------------------------------------------------------------- 3-5 konuşmalar
def speech_docs() -> list:
    sp = pd.read_parquet(SPEECH).sort_values(["source_id", "segment_no"])
    known = speech_known_day(sp)
    docs = []
    for (i, r), k in zip(sp.iterrows(), known):
        src = SOURCES[r.source_id]
        common = dict(publisher=src["channel"], url=src["media_url"], publish_time=r.publish_time, event_day=k,
                      language=src["language"], tool="get_speech_features", usage=r.source_id)
        seg = f"{r.source_id}_s{int(r.segment_no):03d}"
        docs.append(doc(seg, "speech_segment", f"{src['title']} — bölüm {int(r.segment_no)}", r.text, **common,
                        extra=dict(source_id=r.source_id, speaker=src["speaker"], segment_no=int(r.segment_no),
                                   start_seconds=r.start, end_seconds=r.end, avg_logprob=r.avg_logprob,
                                   topics=r.topics.split("|") if r.topics else [], hawkish=int(r.hawkish),
                                   dovish=int(r.dovish), uncertain=bool(r.uncertain), segment_type=r.segment_type)))
        if r.segment_type in ("sayisal_iddia", "dogrulanmis_olgu"):
            docs.append(doc(f"{seg}_claim", "speech_claim", f"{src['speaker']}: sayısal iddia ({r.variable})", r.text,
                            **common, extra=dict(source_id=r.source_id, segment_doc_id=seg, variable=r.variable,
                                                 values=r["values"], status=r.status, tool_value=r.tool_value,
                                                 in_evidence=bool(r.in_evidence))))
    for sid, src in SOURCES.items():
        first = sp[sp.source_id == sid]
        text = f"{src['title']}. Konuşmacı: {src['speaker']}. Kanal: {src['channel']}." + (
            f" Not: {src['note']}" if src["note"] else "")
        docs.append(doc(sid, "speech_source", src["title"], text, publisher=src["channel"], url=src["page_url"],
                        publish_time=pd.Timestamp(src["publish_time"]), event_day=known[first.index].min() if len(first) else None,
                        language=src["language"], tool="get_speech_features", usage=sid,
                        extra=dict(kind=src["kind"], speaker=src["speaker"], media_url=src["media_url"],
                                   official_text_url=src["official_text_url"], n_segments=len(first))))
    return docs


# ---------------------------------------------------------------- 6-8 kurallar ve tanımlar
def rule_docs() -> list:
    system = QM.SYSTEM.split("Kurallar:", 1)[1]
    parts = re.split(r"\n(?=\d+\. )", "\n" + system.strip())
    rules = [p.strip() for p in parts if p.strip()]
    docs = [doc(f"rule_{i}", "action_rule", f"Soru modu kuralı {i}", re.sub(r"\s+", " ", t[len(f'{i}.'):]).strip(),
                publisher="proje", tool="sistem istemi", usage="internal",
                extra=dict(source_file="src/harness/question_mode.py (SYSTEM)"))
            for i, t in enumerate(rules, 1)]
    docs += [doc(f"action_{a}", "action_definition", a, f"{a}: {h}", publisher="proje", tool="sistem istemi",
                 usage="internal", extra=dict(source_file="src/harness/question_mode.py (ACTION_HELP)"))
             for a, h in QM.ACTION_HELP.items()]
    return docs


async def _tools():
    async with Client(build_server()) as client:
        return (await QM.list_tools(client))[0]


def tool_docs() -> list:
    return [doc(f"tool_{t['function']['name']}", "tool_definition", t["function"]["name"],
                t["function"]["description"], publisher="proje", tool=t["function"]["name"],
                usage="internal",
                extra=dict(parameters=t["function"]["parameters"], source_file="src/mcp_server/tools.py"))
            for t in asyncio.run(_tools())]


def dictionary_docs() -> list:
    md = DICTIONARY.read_text()
    sections = re.split(r"\n(?=## \d+\. )", md)[1:]
    docs = []
    for s in sections:
        head, body = s.split("\n", 1)
        m = re.match(r"## (\d+)\. (.+)", head)
        docs.append(doc(f"dict_{int(m.group(1)):02d}", "data_dictionary", m.group(2).strip(), body.strip(),
                        publisher="proje", language="en", tool=None, usage="internal",
                        extra=dict(source_file=str(DICTIONARY), section=int(m.group(1)))))
    return docs


# ---------------------------------------------------------------- 9-14 yapısal kaynaklar (yalnızca meta veri)
def _range(paths) -> dict:
    """Tarih indeksli parquet dosyalarının ortak kapsamı (en erken başlangıç, en geç bitiş)."""
    idx = [pd.read_parquet(p).index for p in paths]
    return dict(start=clean(min(i.min() for i in idx)), end=clean(max(i.max() for i in idx)))


def structured_sources(spec_pdf: Path) -> list:
    prices = sorted(Path("data/isyatirim").glob("*.parquet"))
    mcap = sorted(Path("data/isyatirim_mcap").glob("*"))
    fin = sorted(Path("data/fintables").glob("*.xlsx"))
    fr = sorted(KAP_FR.glob("*.json"))
    n_fr = sum(len(json.loads(f.read_text())) for f in fr)
    macro = sorted(Path("data/macro").glob("*.csv"))
    out = [
        dict(source_id="prices_isyatirim", title="Günlük fiyatlar ve XU100 (34 hisse)", publisher="İş Yatırım",
             url="https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil",
             tools=["get_stock_history", "get_index_history", "calculate_indicators", "detect_reversal_events",
                    "run_backtest"],
             coverage=_range(prices), point_in_time="o günün kapanışında bilinir; bölünme/bedelsiz düzeltmeli",
             validation="Yahoo Finance (yfinance, data/prices) ile karşılaştırıldı (veri sözlüğü §2)",
             files=file_meta(prices)),
        dict(source_id="market_cap_isyatirim", title="Günlük piyasa değeri", publisher="İş Yatırım",
             tools=["get_fundamental_metrics"], files=file_meta(mcap)),
        dict(source_id="fundamentals_fintables", title="Çeyreklik bilançolar (30 hisse)", publisher="Fintables",
             url="https://fintables.com", tools=["get_fundamental_metrics", "calculate_fundamental_change"],
             point_in_time="her çeyrek yalnızca KAP yayın gününden sonra kullanılır (kaynak 11)",
             note="Fintables TMS 29 dönemini tek birime yeniden endeksliyor (veri sözlüğü §5)", files=file_meta(fin)),
        dict(source_id="kap_financial_reports", title=f"Bilanço yayın tarihleri ({n_fr} KAP finansal rapor bildirimi)",
             publisher="KAP", url="https://www.kap.org.tr/tr/api/disclosure/list/main",
             tools=["get_fundamental_metrics", "calculate_fundamental_change"], n_items=n_fr, files=file_meta(fr)),
        dict(source_id="macro_evds", title="USD/TRY, EUR/TRY, politika faizi, TÜFE", publisher="TCMB EVDS",
             url="https://evds3.tcmb.gov.tr/", tools=["get_macro_context"],
             series={k: dict(code=v[0], frequency=v[1]) for k, v in EVDS_SERIES.items()},
             point_in_time="TÜFE, izleyen ayın 3'ünde (TÜİK yayın günü) bilinir; available_date sütunu",
             files=file_meta(macro)),
        dict(source_id="fed_funds", title="Fed federal fon hedef aralığı", publisher="Federal Reserve",
             url="https://www.federalreserve.gov/monetarypolicy/openmarket.htm", tools=["get_macro_context"],
             n_changes=len(FED_TABLE), source_file="src/data/macro.py (FED_TABLE)",
             point_in_time="karar ABD saatiyle 14:00'te, BIST kapandıktan sonra açıklanır"),
    ]
    spec = dict(source_id="project_spec", title="Engineering Economics Project 2026 (CSE-481)", publisher="ders hocası",
                note="yalnızca künye; metin telifli olduğu için dosyaya konmadı", file=spec_pdf.name)
    if spec_pdf.exists():
        from pypdf import PdfReader
        spec.update(pages=len(PdfReader(spec_pdf).pages), sha256=sha256_file(spec_pdf), bytes=spec_pdf.stat().st_size)
    out.append(spec)
    return out


# ---------------------------------------------------------------- birleştir
def build(spec_pdf: Path = SPEC_PDF) -> dict:
    docs = news_docs() + speech_docs() + rule_docs() + tool_docs() + dictionary_docs()
    ids = [d["doc_id"] for d in docs]
    assert len(ids) == len(set(ids)), "tekrarlanan doc_id"
    counts = pd.Series([d["group"] for d in docs]).value_counts().to_dict()
    return dict(
        knowledge_base_version=VERSION, created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        code_commit=git_commit(),
        description=("BIST harness bilgi tabanı: modelin MCP araçlarıyla ulaştığı metin belgeleri, soru modunun "
                     "kural ve tanım belgeleri ve yapısal veri kaynaklarının meta verileri."),
        retrieval=("Gömme (embedding) tabanlı arama yok. Model bilgiye yalnızca MCP araçlarıyla ulaşır "
                   "(get_news_events, get_speech_features ve diğerleri); araçlar analiz tarihi tavanını uygular, "
                   "event_day'i tavandan sonra olan belge modele hiç görünmez. Kural belgeleri sistem isteminin ve "
                   "araç tanımlarının kendisidir. Konuşmalardaki sayısal iddialar yalnızca araçla doğrulanınca "
                   "(status=VERIFIED) kanıta girer."),
        fields=dict(doc_id="belge kimliği", group="belge grubu", title="başlık", text="belge metni",
                    sha256="text alanının SHA-256 özeti (UTF-8)", publisher="yayıncı", url="kaynak bağlantısı",
                    publish_time="yayın zamanı (İstanbul saati)",
                    event_day="sistemin belgeyi kullanabildiği ilk işlem günü (18:00 sonrası yayın -> ertesi gün)",
                    fetched_at="indirme zamanı (nereden geldiği: notes)",
                    ticker="hisse", sector="sektör", language="dil", tool="belgeyi modele getiren araç",
                    usage="kullanım/lisans notunun anahtarı (usage_notes)", extra="gruba özgü alanlar"),
        groups=[dict(group=g, kind=k, title=t, tool=tool, count=counts.get(g, 0)) for g, k, t, tool in GROUPS],
        usage_notes=USAGE, notes=NOTES,
        excluded=["kendi araştırma raporlarımız (sistemin çıktısı; ayrıca teslim edilir)",
                  "ince ayar eğitim verisi (manifestte ayrı madde)",
                  "Fed konuşmasının ses dosyası (telif ve boyut; bağlantısı speech_source kaydında)"],
        structured_sources=structured_sources(spec_pdf), n_documents=len(docs), documents=docs)


def main():
    ap = argparse.ArgumentParser(description="RAG dosyası: submission/rag/knowledge_base.json")
    ap.add_argument("--pdf", type=Path, default=SPEC_PDF, help="proje PDF'i (yalnızca künye ve SHA-256 için)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    kb = build(a.pdf)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(kb, ensure_ascii=False, indent=1, allow_nan=False))
    print(f"{a.out}: {kb['n_documents']} belge, {a.out.stat().st_size / 1e6:.1f} MB")
    for g in kb["groups"]:
        print(f"  {g['group']:<18} {g['count']:>6}  {g['title']}")


if __name__ == "__main__":
    main()
