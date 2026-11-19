"""Araştırma raporlarını üretir (Adım 15). Anahtar gerekmez, ağa çıkmaz.

  python -m src.reports.build_reports            # tüm hazır raporlar
  python -m src.reports.build_reports --only 1   # yalnızca Rapor 1
"""
import argparse

from src.reports import r1_technical, r2_sector, r3_weekday

REPORTS = {1: r1_technical.build, 2: r2_sector.build, 3: r3_weekday.build}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Araştırma raporları")
    ap.add_argument("--only", default=",".join(map(str, REPORTS)), help="ör. 1 ya da 1,2")
    a = ap.parse_args()
    for k in map(int, a.only.split(",")):
        print("yazıldı:", REPORTS[k]())
