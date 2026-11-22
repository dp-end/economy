
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

MEMBERSHIP_PATH = "config/bist100_membership.csv"


def index_membership(ticker: str, day, path=MEMBERSHIP_PATH) -> dict:
    """Brief §5: endeks üyeliği her çalıştırmanın başında kontrol edilir. Dosyada analiz gününü kapsayan dönem
    varsa üye mi (True/False) ve kaynağı; yoksa member=None ('bu dönem için doğrulanmadı').
    Kaynak: Borsa İstanbul dönemsel değişiklik duyuruları (KAP)."""
    m = pd.read_csv(path, parse_dates=["period_start", "period_end"])
    d = pd.Timestamp(day)
    hit = m[(m["ticker"] == str(ticker).strip().upper()) & (m["period_start"] <= d) & (m["period_end"] >= d)]
    if hit.empty:                                      # dosya tam üye listesi değil: kayıt yoksa bilinmiyor
        return dict(member=None, period=None, source=None)
    r = hit.iloc[0]
    return dict(member=bool(r["member"]), period=f"{r['period_start'].date()}/{r['period_end'].date()}",
                source=r["source"])
