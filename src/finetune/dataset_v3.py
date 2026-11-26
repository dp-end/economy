"""v3 ince ayar veri seti (26 Kasım 2026, kullanıcı kararları: kural satırı, çeşitli ifadeler, nadir eylemler).

v1/v2 değerlendirmesinde (reports/local_llm/eval) hataların çoğu ya kuralın yanlış uygulanması ya da yeni ifadelere
yanlış araç çağrısıydı. v3 verisi üç yerde farklıdır:
  1. Cevaplarda eylemden önce 'Kural kontrolü' satırı (templates.finish): kuralın her koşulu araçtaki değeriyle,
     evet/hayır. Kuralın gerektirdiği oran ve işlem günü sayıları artık araç çıktısında (tools.py, v3.1).
  2. Soru ifadeleri: eğitim kalıpları + v1/v2 görülmemiş setinin kalıpları (o set artık 'görülmüş' sayılır; v3 yeni
     bir görülmemiş setle ölçülür: src/finetune/heldout_v3.py). Terimler (hafta deseni, olay, strateji adları) iki
     kalıp arasından rastgele seçilir; tarih biçimleri 10.09.2026, 2026-09-10 ve 10 Eylül 2026.
  3. Nadir eylemler: hedefli catch-up ve teknik dönüş örnekleri 20'şer yerine 40'ar; ek 20 hedefli bilanço örneği
     (son çeyrek iyileşen ve son 63 işlem gününde açıklanmış; rastgele 100 bilanço sorusunda bu eylem 27'de kalıyordu).
Kurallar, araçlar, denetim ve konuşma biçimi canlı soru moduyla aynıdır (build_dataset.run_spec).

Çıktı: data/finetune_v3/ (git dışında). Çalıştırma: python -m src.finetune.dataset_v3
"""
import argparse
import asyncio
import json
from pathlib import Path

from src.finetune import build_dataset as B
from src.finetune import heldout as H
from src.finetune import templates as TP

OUT_V3 = Path("data/finetune_v3")
SEED_V3 = 20261126
COUNTS_V3 = dict(s1=80, s2=80, s3=80, s4=80, s5=60, other=60, error=30, s1_cand=40, s3_cand=40, s4_cand=20)
CARD_V3 = Path("reports/local_llm/dataset_card_v3.md")
V3_NOTE = """**v3 changes.** (1) Every scenario answer ends with a 'Kural kontrolü' line before the action: each condition of
the rule with its value from the tool output and yes/no. The tools now return the numbers the rules need (success
rate, trading days since the last event, trading days since the last disclosure), so the model no longer has to
divide or count trading days. (2) Questions use the original phrasings plus those of the first held-out set (which is
therefore no longer held out; v3 is evaluated on a new set, `src/finetune/heldout_v3.py`). (3) More rare actions:
40 targeted catch-up, 40 targeted technical-reversal and 20 targeted fundamental-improvement examples.

"""


class Mix(dict):
    """Anahtar -> birden çok ifade; her okumada sampler'ın rastgele sayı üreteciyle biri seçilir."""

    def __init__(self, rng, *banks):
        super().__init__({k: [b[k] for b in banks] for k in banks[0]})
        self.rng = rng

    def __getitem__(self, k):
        return self.rng.choice(super().__getitem__(k))


class SamplerV3(B.Sampler):
    Q = {k: TP.Q[k] + H.Q_EVAL.get(k, []) for k in TP.Q}
    CONDS = {d: B.COND_SETS[d] + H.COND_EVAL[d] for d in B.COND_SETS}

    def __init__(self, rng):
        super().__init__(rng)
        self.WEEK = Mix(rng, TP.WEEK_PATTERNS, H.WEEK_EVAL)
        self.EVENTS = Mix(rng, TP.EVENT_TR, H.EVENT_EVAL)
        self.STRAT = Mix(rng, TP.STRAT_TR, H.STRAT_EVAL)

    def s4_cand(self) -> dict:
        """Hedefli bilanço örneği: kural aynı; yalnızca son çeyreği iyileşen ve açıklaması güncel olan hisse/tarih."""
        for _ in range(B.CAND_TRIES):
            sp = self.s4(bank=False)
            try:
                with B.data_ceiling(sp["asof"]):
                    lq = B.TOOLS.calculate_fundamental_change(ticker=sp["T"], start=sp["start"], end=sp["asof"])["last_quarter"]
            except B.TOOLS.ToolError:
                continue
            if lq and lq["improving"] and lq["trading_days_since_disclosure"] <= TP.FUND_RECENT:
                return dict(sp, targeted=True)
        raise RuntimeError("s4_cand: kuralı sağlayan örnek bulunamadı")

    def dfmt(self, iso: str) -> str:
        x = self.rng.random()
        return TP.dtr(iso) if x < 0.5 else H.yazili(iso) if x < 0.7 else iso


def build_v3(counts: dict = None, seed: int = SEED_V3, out: Path = OUT_V3) -> dict:
    return asyncio.run(B.build(counts or COUNTS_V3, seed, out, sampler=SamplerV3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v3 ince ayar veri seti")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUT_V3))
    a = ap.parse_args()
    counts = {k: max(1, round(v * a.scale)) for k, v in COUNTS_V3.items()}
    stats = build_v3(counts, out=Path(a.out))
    if Path(a.out) == OUT_V3 and a.scale == 1.0:
        B.write_card(stats, counts, CARD_V3, "data/finetune_v3/", "python -m src.finetune.dataset_v3", V3_NOTE)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
