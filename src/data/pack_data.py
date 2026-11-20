"""data/ önbelleğini tek zip arşivine paketler (temiz kurulum için; git'e girmez).

  python -m src.data.pack_data            # dist/economy_data_<son fiyat günü>.zip
Kurulumda: depo kökünde `unzip dist/economy_data_*.zip` -> data/ klasörü oluşur.

Ham ses dosyaları (*.m4a, ~31 MB) dışarıda bırakılır: transkript (fed_*_whisper.parquet) ve resmi metinler arşivde.
Üçüncü taraf verisi (Fintables, KAP, İş Yatırım) yalnızca ders değerlendirmesi için paylaşılır; yeniden dağıtılmaz.
"""
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path("data")
DIST = Path("dist")
SKIP_SUFFIXES = {".m4a", ".mp3", ".wav", ".webm"}


def files_to_pack() -> list:
    return sorted(p for p in DATA.rglob("*") if p.is_file() and p.suffix.lower() not in SKIP_SUFFIXES
                  and p.name != ".DS_Store")


def pack() -> Path:
    from src.data.prices import get_index_history
    last = get_index_history("XU100").index[-1]
    DIST.mkdir(exist_ok=True)
    out = DIST / f"economy_data_{last:%Y-%m-%d}.zip"
    files = files_to_pack()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.as_posix())
    size = sum(p.stat().st_size for p in files)
    print(f"{len(files)} dosya, {size / 1e6:.1f} MB -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(pd.Series([p.parts[1] if len(p.parts) > 2 else "." for p in files]).value_counts().to_string())
    return out


if __name__ == "__main__":
    pack()
