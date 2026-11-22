"""Brief §32'deki önerilen fonksiyon adları (Adım 6, 22 Kasım 2026).

28 adın hepsi buradan çağrılabilir. 16'sı kodda aynı adla vardı ve doğrudan dışa aktarılır; 12'si projede başka
adla yapılmıştı ve burada o koda ince birer sarmalayıcıdır (yeni hesap yoktur). Eşleme tablosu: SPEC_MAP ve README.
"""
import pandas as pd

from src.backtest.engine import calculate_performance, run_backtest
from src.backtest.strategies import generate_strategy_variants
from src.data.prices import get_index_history, get_stock_history
from src.data.universe import build_sector_peer_groups, load_universe
from src.harness.decision_log import log_decision, replay_analysis
from src.harness.evidence import build_evidence_bundle
from src.harness.quality_gate import check_data_quality
from src.mcp_server.tools import (calculate_indicators, find_support_resistance, get_fundamental_metrics,
                                  get_macro_context, test_weekday_pattern)

test_weekday_pattern.__test__ = False          # pytest bu adı test sanmasın

# brief adı -> projedeki karşılığı
SPEC_MAP = {
    "load_universe": "src/data/universe.py: load_universe",
    "get_stock_history": "src/data/prices.py: get_stock_history",
    "get_index_history": "src/data/prices.py: get_index_history",
    "build_sector_peer_groups": "src/data/universe.py: build_sector_peer_groups",
    "calculate_indicators": "src/mcp_server/tools.py: calculate_indicators (src/indicators/technical.py: add_all)",
    "calculate_bollinger_events": "src/events/technical_events.py: detect_events (Bollinger types)",
    "calculate_kama_state": "src/events/technical_events.py: states (kama_state, kama_turn)",
    "calculate_supertrend_state": "src/events/technical_events.py: states (st_state)",
    "calculate_ichimoku_state": "src/events/technical_events.py: states (cloud_pos); src/indicators/technical.py: ichimoku",
    "find_support_resistance": "src/mcp_server/tools.py: find_support_resistance (src/indicators/support_resistance.py)",
    "calculate_reversal_strength": "src/events/technical_events.py: event_table (direction-adjusted forward returns)",
    "rank_sector_laggards": "src/mcp_server/tools.py: rank_sector_relative_strength (every peer group)",
    "test_weekday_pattern": "src/mcp_server/tools.py: test_weekday_pattern",
    "detect_multiday_patterns": "src/events/calendar_patterns.py: pattern_table",
    "get_fundamental_metrics": "src/mcp_server/tools.py: get_fundamental_metrics (src/data/pit_fundamentals.py)",
    "calculate_quarterly_changes": "src/mcp_server/tools.py: calculate_fundamental_change",
    "align_financial_disclosure_dates": "src/data/kap_dates.py: load_kap_table / visible_asof_kap",
    "get_macro_context": "src/mcp_server/tools.py: get_macro_context (src/data/macro.py)",
    "extract_news_events": "src/data/news_tags.py: get_news_events",
    "transcribe_finance_video_segment": "src/data/speech.py: transcribe (+ merge_to_sentences)",
    "extract_speech_features": "src/data/speech_features.py: segment_features / get_speech_features",
    "build_evidence_bundle": "src/harness/evidence.py: build_evidence_bundle",
    "generate_strategy_variants": "src/backtest/strategies.py: generate_strategy_variants",
    "run_backtest": "src/backtest/engine.py: run_backtest (MCP: src/mcp_server/tools.py: run_backtest)",
    "calculate_performance": "src/backtest/engine.py: calculate_performance",
    "check_data_quality": "src/harness/quality_gate.py: check_data_quality",
    "log_decision": "src/harness/decision_log.py: log_decision",
    "replay_analysis": "src/harness/decision_log.py: replay_analysis",
}


def _states(ticker: str) -> pd.DataFrame:
    from src.events.technical_events import states
    from src.indicators.features import full_features
    f = full_features(ticker)
    return f.join(states(f))


def calculate_bollinger_events(ticker: str) -> pd.DataFrame:
    """Bollinger olayları (alt/üst bant dönüşü, orta bant testi), olay anındaki durum ve ileri getirilerle."""
    from src.events.technical_events import event_table
    e = event_table(ticker)
    return e[e["event"].str.startswith("bb_")]


def calculate_kama_state(ticker: str) -> pd.DataFrame:
    return _states(ticker)[["Close", "kama", "kama_slope", "kama_state", "kama_turn"]]


def calculate_supertrend_state(ticker: str) -> pd.DataFrame:
    return _states(ticker)[["Close", "supertrend", "st_dir", "st_state"]]


def calculate_ichimoku_state(ticker: str) -> pd.DataFrame:
    return _states(ticker)[["Close", "tenkan", "kijun", "cloud_top", "cloud_bottom", "cloud_pos"]]


def calculate_reversal_strength(ticker: str, event: str = None) -> pd.DataFrame:
    """Olay başına yön düzeltmeli ileri getiri (%; pozitif = olayın beklediği yönde): 1, 3, 5, 10 gün."""
    from src.events.technical_events import HORIZONS, event_table
    e = event_table(ticker)
    if event:
        e = e[e["event"] == event]
    out = e[["event", "direction"]].copy()
    for h in HORIZONS:
        out[f"strength_{h}d"] = e["direction"] * e[f"fwd_{h}"] * 100
        out[f"excess_{h}d"] = e["direction"] * e[f"exc_{h}"] * 100
    return out


def rank_sector_laggards(date: str = None, horizon: int = 10) -> pd.DataFrame:
    """Bütün akran gruplarında hisse getirisi eksi akran ortancası (kendisi hariç); en geride olan önce."""
    from src.mcp_server.tools import rank_sector_relative_strength
    rows = []
    for g in sorted(load_universe()["peer_group"].unique()):
        r = rank_sector_relative_strength(g, horizon=horizon, date=date)
        rows += [dict(m, sector=g) for m in r["members"]]
    return pd.DataFrame(rows).sort_values("lag").reset_index(drop=True)


def detect_multiday_patterns(ks=(2, 3, 4), horizons=(1, 3, 5)) -> pd.DataFrame:
    """Art arda k gün düşüş/yükseliş sonrası h günlük getiri, aynı günün hisse ortalamasına göre."""
    from src.events.calendar_patterns import _data, pattern_table
    d = _data()
    return pattern_table(d["ret"], d["panel"], d["valid"], ks=ks, horizons=horizons)


def calculate_quarterly_changes(ticker: str, periods: int = 4, asof: str = None) -> dict:
    from src.mcp_server.tools import calculate_fundamental_change
    return calculate_fundamental_change(ticker, periods=periods, asof=asof)


def align_financial_disclosure_dates(date=None) -> pd.DataFrame:
    """Bilanço dönemi -> KAP yayın zamanı ve olay günü (ilk işlem günü). date verilirse o gün görünenler."""
    from src.data.kap_dates import load_kap_table, visible_asof_kap
    t = load_kap_table()
    return visible_asof_kap(t, date) if date is not None else t


def extract_news_events(ticker: str = None, start: str = None, end: str = None, asof: str = None, topic: str = None):
    """Zaman damgalı KAP/RSS haber olayları ve etiketleri; asof verilirse yalnızca o gün bilinenler."""
    from src.data.news_tags import get_news_events, load_news_events
    return get_news_events(load_news_events(), ticker=ticker, topic=topic, start=start, end=end, asof=asof)


def transcribe_finance_video_segment(source_id: str, force: bool = False) -> pd.DataFrame:
    """Kaynağın sesini Whisper ile yazıya döker ve cümlelere birleştirir (ffmpeg ve faster-whisper gerekir)."""
    from src.data.speech import merge_to_sentences, transcribe
    return merge_to_sentences(transcribe(source_id, force=force))


def extract_speech_features(source_id: str = None):
    """Konuşma özellikleri (konu, şahin/güvercin tonu, doğrulanmış iddialar); kaynak verilirse o konuşma."""
    from src.data.speech_features import get_speech_features, load_speech_features
    f = load_speech_features()
    return get_speech_features(f, source_id) if source_id else f
