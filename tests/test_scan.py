"""Adım 4: evren tarama modu (PDF §33)."""
import pytest

from src.harness.evidence import run_analysis
from src.harness.scan import CATCH_UP, link_analyses, load_scan, log_scan, replay_scan, scan_universe, select

DAY = "2026-11-06"
SUBSET = ["ASELS", "EKGYO", "ECILC", "CCOLA"]


@pytest.fixture(scope="module")
def scan():
    return scan_universe(DAY, tickers=SUBSET)


def test_scan_rows_are_the_rule_layer(scan):
    """Tarama yeni bir karar kuralı eklemez: her satırın eylemi ve kapısı run_analysis ile aynı."""
    for r in scan["rows"]:
        a = run_analysis(r["ticker"], DAY)
        assert (r["action"], r["gate"], r["secondary"]) == (a["action"], a["gate"]["status"], list(a["secondary"]))
    assert scan["candidates"] == [r["ticker"] for r in scan["rows"] if r["action"] == CATCH_UP]


def test_near_misses_say_which_condition_failed(scan):
    near = {x["ticker"]: x["why"] for x in scan["near_misses"]}
    assert near["EKGYO"] == ["son bilanço alt üçte birde"]                 # geride, sektör güçlü, toparlanıyor
    assert near["CCOLA"] == ["toparlanmıyor (KAMA altında ve Supertrend aşağı)"]
    assert "ASELS" not in near                                             # geride değil
    lags = [x["sector_lag_10"] for x in scan["near_misses"]]
    assert lags == sorted(lags)                                            # en geride olan önce


def test_select_orders_catch_up_by_sector_lag_and_caps():
    rows = [dict(ticker=t, sector_lag_10=v) for t, v in (("A", -0.06), ("B", -0.20), ("C", -0.10), ("D", -0.08))]
    sc = dict(action=CATCH_UP, rows=rows, candidates=["A", "B", "C", "D"])
    assert select(sc, top=3) == ["B", "C", "D"]
    assert select(dict(sc, action="TECHNICAL_REVERSAL_CANDIDATE"), top=2) == ["A", "B"]


def test_only_candidate_actions_can_be_scanned():
    with pytest.raises(ValueError):
        scan_universe(DAY, action="HOLD", tickers=["ASELS"])


def test_scan_log_and_replay(scan, tmp_path):
    db = tmp_path / "log.sqlite"
    sid = log_scan(scan, select(scan), 3, db)
    link_analyses(sid, ["abc123"], db)
    stored = load_scan(sid, db)
    assert stored["candidates"] == scan["candidates"] and stored["analysis_ids"] == ["abc123"]
    r = replay_scan(sid, db)
    assert r["same"] and r["changed_tickers"] == [] and r["changed_data_files"] == [] and not r["procedure_changed"]


def test_blocked_candidate_is_secondary_with_reason_not_a_near_miss():
    """2026-01-09: ARCLK yakalama koşullarını sağlıyor ama çelişen teknik sinyal yüzünden INVESTIGATE."""
    sc = scan_universe("2026-01-09", tickers=["ARCLK", "PGSUS", "TCELL"])
    assert sc["candidates"] == ["PGSUS"]
    assert sc["also_secondary"] == [dict(ticker="ARCLK", action="INVESTIGATE",
                                         why=["çelişen teknik sinyal (aynı 3 günde boğa ve ayı olay)"])]
    assert [x["ticker"] for x in sc["near_misses"]] == ["TCELL"]
