
from pathlib import Path
import pandas as pd

UNIVERSE_PATH = Path(__file__).resolve().parents[2] / "config" / "universe.csv"
MIN_PEERS = 3


def load_universe(path=UNIVERSE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df


def build_sector_peer_groups(df=None, min_size=MIN_PEERS):
    """Peer grubu -> ticker listesi ve küçük gruplar için uyarılar döner."""
    if df is None:
        df = load_universe()
    groups = {g: sub["ticker"].tolist() for g, sub in df.groupby("peer_group")}
    warnings = [
        f"{g}: sadece {len(t)} hisse (< {min_size}), sektör ortancası zayıf"
        for g, t in groups.items() if len(t) < min_size
    ]
    return groups, warnings