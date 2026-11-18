"""Konuşma katmanı: kaynak listesi, ses -> metin (Whisper), resmi metin ve transkript kalitesi.

Kaynak listesi deneylerden önce sabitlenir (SOURCES). Hocadan onaylı liste gelmediği için
yalnızca resmi kurum kanalları seçildi:
  - Fed FOMC basın toplantısı: Fed'in kendi sitesindeki açık oynatıcıdan ses indirilir, Whisper ile
    metne çevrilir; Fed'in yayımladığı resmi metinle karşılaştırılarak kalite ölçülür.
  - TCMB Enflasyon Raporu sunumu: video yalnızca YouTube'da ve YouTube bot doğrulaması istiyor.
    Platform koruması aşılmaz; TCMB'nin yayımladığı resmi konuşma metni kullanılır.
Her satırda kaynak bağlantısı, konuşmacı, kanal, yayın zamanı ve (sesli kaynakta) zaman aralığı saklanır.
"""
import difflib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.kap_dates import event_day
from src.data.prices import get_index_history

DIR = Path("data/speech")
SEGMENTS_FILE = "reports/speech_segments.parquet"
SAMPLE_RATE = 16000
WHISPER_MODEL = "small"
MIN_MATCH_RATE = 0.85        # resmi metinle örtüşme bunun altındaysa: düşük kalite uyarısı
LOW_LOGPROB = -1.0           # Whisper bu güven değerinin altındaki parçalar "düşük güvenli" sayılır
MAX_LOW_CONF_SHARE = 0.20
MAX_SEGMENT_WORDS = 40       # sesli kaynakta bir parça en fazla ~40 kelime

SOURCES = {
    "fed_20261028": dict(
        kind="video", language="en", speaker="Kevin Warsh (Fed Başkanı)", channel="Federal Reserve",
        title="FOMC Press Conference, October 28, 2026",
        publish_time="2026-10-28 21:30",          # 14:30 ABD doğu saati (EDT, UTC-4) = 21:30 İstanbul
        media_url="https://players.brightcove.net/66043936001/default_default/index.html?videoId=6407552581112",
        page_url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20261028.htm",
        official_text_url="https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20261028.pdf",
        note=""),
    "tcmb_20261112": dict(
        kind="text", language="tr", speaker="Fatih Karahan (TCMB Başkanı)", channel="TCMB",
        title="Enflasyon Raporu 2026-IV Bilgilendirme Toplantısı",
        publish_time="2026-11-12 10:30",
        media_url="https://www.youtube.com/watch?v=H6prPpAt9uU",
        page_url="https://www.tcmb.gov.tr/",
        official_text_url=("https://www.tcmb.gov.tr/wps/wcm/connect/03393ac1-4cb9-4ca5-8ed3-d852de255ff5/"
                           "KonusmaB12_11_2026.pdf?MOD=AJPERES"),
        note="Ses indirilmedi: YouTube bot doğrulaması istiyor (platform koruması aşılmaz). Resmi konuşma metni kullanıldı."),
}


# ---------- ses ----------
def audio_path(sid: str) -> Path:
    return DIR / f"{sid}.m4a"


def download_audio(sid: str) -> Path:
    """Yalnızca ses (m4a). Dosya varsa tekrar indirilmez."""
    f = audio_path(sid)
    if not f.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "-f", "bestaudio/best", "-x", "--audio-format", "m4a",
                        "-o", str(DIR / f"{sid}.%(ext)s"), SOURCES[sid]["media_url"]], check=True)
    return f


def load_audio(path: Path) -> np.ndarray:
    """ffmpeg ile 16 kHz tek kanal float32 dizi (faster-whisper'ın kendi okuyucusu av sürümüyle uyumsuz)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", str(SAMPLE_RATE), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def transcribe(sid: str, model_name: str = WHISPER_MODEL, force: bool = False) -> pd.DataFrame:
    """Whisper parçaları: start, end (video saniyesi), text, avg_logprob, no_speech_prob. Sonuç önbelleğe yazılır."""
    f = DIR / f"{sid}_whisper.parquet"
    if f.exists() and not force:
        return pd.read_parquet(f)
    from faster_whisper import WhisperModel             # ağır kütüphane: yalnızca gerektiğinde yüklenir
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(load_audio(download_audio(sid)), language=SOURCES[sid]["language"], vad_filter=True)
    df = pd.DataFrame([dict(start=s.start, end=s.end, text=s.text.strip(), avg_logprob=s.avg_logprob,
                            no_speech_prob=s.no_speech_prob) for s in segs])
    df.to_parquet(f, index=False)
    return df


def merge_to_sentences(parts: pd.DataFrame, max_words: int = MAX_SEGMENT_WORDS) -> pd.DataFrame:
    """Whisper parçalarını cümle sonuna (. ? !) ya da max_words kelimeye kadar birleştirir.
    Whisper çoğu yerde nokta koymadığı için kelime sınırı gerekir. Zaman aralığı ilk/son parçadan gelir."""
    rows, buf = [], []
    for i, r in enumerate(parts.itertuples()):
        buf.append(r)
        n_words = sum(len(b.text.split()) for b in buf)
        if re.search(r"[.?!][\"')]?$", r.text) or n_words >= max_words or i == len(parts) - 1:
            rows.append(dict(start=buf[0].start, end=buf[-1].end, text=" ".join(b.text for b in buf),
                             avg_logprob=float(np.mean([b.avg_logprob for b in buf]))))
            buf = []
    return pd.DataFrame(rows)


# ---------- resmi metin ----------
def official_text(sid: str) -> str:
    """Kurumun yayımladığı PDF metni (önbellekli)."""
    from pypdf import PdfReader
    f = DIR / f"{sid}_official.pdf"
    if not f.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(SOURCES[sid]["official_text_url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        f.write_bytes(r.content)
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(f)).pages)


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]


# ---------- kalite ----------
def words(text: str) -> list:
    t = (text or "").replace("I", "ı").replace("İ", "i").lower()
    t = t.replace("¾", " three quarters").replace("½", " half").replace("¼", " quarter")
    return re.findall(r"[a-zçğıöşü0-9]+", t)


def match_rate(transcript: str, reference: str) -> float:
    """Transkript kelimelerinin resmi metinle sırasıyla eşleşen oranı (0-1). Rakam/yazı farkı eşleşmez sayılır."""
    a, b = words(transcript), words(reference)
    if not a:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(a)


def quality(sid: str, segments: pd.DataFrame) -> dict:
    """Sesli kaynakta: resmi metinle örtüşme + düşük güvenli parça oranı. Metin kaynağında ölçüm yok."""
    if SOURCES[sid]["kind"] == "text":
        return dict(source_id=sid, method="resmi metin", match_rate=np.nan, low_conf_share=np.nan, warning="")
    mr = match_rate(" ".join(segments["text"]), official_text(sid))
    low = float((segments["avg_logprob"] < LOW_LOGPROB).mean())
    warn = []
    if mr < MIN_MATCH_RATE:
        warn.append(f"resmi metinle örtüşme düşük ({mr:.0%})")
    if low > MAX_LOW_CONF_SHARE:
        warn.append(f"düşük güvenli parça oranı yüksek ({low:.0%})")
    return dict(source_id=sid, method=f"whisper-{WHISPER_MODEL}", match_rate=round(mr, 3),
                low_conf_share=round(low, 3), warning="; ".join(warn))


# ---------- tablo ----------
def source_segments(sid: str) -> pd.DataFrame:
    """Bir kaynağın cümleleri + kaynak bilgisi. Sesli kaynakta start/end video saniyesidir."""
    src = SOURCES[sid]
    if src["kind"] == "video":
        seg = merge_to_sentences(transcribe(sid))
    else:
        seg = pd.DataFrame(dict(text=split_sentences(official_text(sid))))
        seg["start"] = seg["end"] = seg["avg_logprob"] = np.nan
    seg.insert(0, "source_id", sid)
    seg["segment_no"] = range(len(seg))
    for k in ("kind", "language", "speaker", "channel", "title", "media_url", "page_url", "official_text_url"):
        seg[k] = src[k]
    seg["publish_time"] = pd.Timestamp(src["publish_time"])
    return seg


def speech_segments(ids=None) -> pd.DataFrame:
    out = pd.concat([source_segments(s) for s in (ids or list(SOURCES))], ignore_index=True)
    out["event_day"] = event_day(out["publish_time"], get_index_history("XU100").index)
    return out


def load_speech_segments(path=SEGMENTS_FILE) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 110)
    seg = speech_segments()
    seg.to_parquet(SEGMENTS_FILE, index=False)
    print(seg.groupby("source_id").agg(cumle=("text", "size"), yayin=("publish_time", "first"),
                                       event_day=("event_day", "first")).to_string())
    print("\nKalite:")
    print(pd.DataFrame([quality(s, seg[seg["source_id"] == s]) for s in SOURCES]).to_string(index=False))
    print("\nFed ilk 5 cümle (video saniyesi):")
    f = seg[seg["source_id"] == "fed_20261028"].head(5)
    print(f[["start", "end", "text"]].round(1).to_string(index=False))
