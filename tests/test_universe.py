
from src.data.universe import load_universe, build_sector_peer_groups


def test_universe_basic():
    df = load_universe()
    assert df["ticker"].is_unique
    assert (df["yahoo"] == df["ticker"] + ".IS").all()
    assert df["is_core"].sum() == 30


def test_peer_groups():
    df = load_universe()
    groups, warnings = build_sector_peer_groups(df)
    assert sum(len(v) for v in groups.values()) == len(df)
    assert len(warnings) == 1 and "Healthcare" in warnings[0]
