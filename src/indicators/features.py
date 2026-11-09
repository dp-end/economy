
from src.data.prices import get_stock_history
from src.indicators.technical import add_all
from src.indicators.support_resistance import sr_features
from src.indicators.relative import relative_features


def full_features(ticker: str):
    """Fiyat + teknik göstergeler + destek/direnç + göreli özellikler (tek tablo)."""
    df = get_stock_history(ticker)
    return add_all(df).join(sr_features(df)).join(relative_features(ticker))