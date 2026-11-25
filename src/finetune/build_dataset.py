"""İnce ayar veri setini üretir (yerel model 3. adım).

Her kayıt soru modundaki konuşmanın birebir aynısıdır: aynı sistem metni ve araç tanımları (question_mode.first_messages,
list_tools), gerçek MCP araç çıktıları (aynı tavan ve aynı kısaltılmış görünüm, question_mode._call_tool) ve Python
şablonuyla yazılmış cevap (src/finetune/templates.py). Her cevap soru modunun denetiminden geçmeden kayda girmez;
eylem onaylanmış kurallarla seçilir. Bulut LLM yoktur; aynı tohumla aynı dosyalar üretilir.

Kullanıcı kararları (24 Kasım 2026): ~450 konuşma (S1-S4 80'er, S5 60, diğer araçlar 60, hatalı istek 30); eylem
kuralları templates.py başındaki tablo; hocanın 5 örnek sorusu eğitimde (değerlendirme ayrı sorularla yapılacak);
analiz tarihleri 2024-2026 arasına yayılır.

MLX yalnızca son asistan mesajından öğrendiği için (mask_prompt) her konuşma asistan turlarına bölünür: her tur ayrı
örnektir (öncesi istem, o tur hedef). Çıktı (data/ git dışında):
  data/finetune/conversations.jsonl  tam konuşmalar + üst bilgi
  data/finetune/train.jsonl, valid.jsonl  MLX 'chat' biçimi ({"messages": [...], "tools": [...]}), tur başına
  data/finetune/stats.json  sayımlar
Çalıştırma: python -m src.finetune.build_dataset            (varsayılan sayılar)
            python -m src.finetune.build_dataset --scale 0.05  (hızlı deneme)
"""
import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from mcp import Client

from src.data.prices import get_index_history
from src.data.universe import load_universe
from src.events.technical_events import DIRECTION, TOUCH_DIRECTION
from src.finetune import templates as TP
from src.harness import question_mode as QM
from src.mcp_server.server import build_server
from src.mcp_server import tools as TOOLS
from src.mcp_server.tools import data_ceiling

OUT = Path("data/finetune")
SEED = 20261124
DATE_MIN, DATE_MAX = "2024-03-01", "2026-11-06"
COUNTS = dict(s1=80, s2=80, s3=80, s4=80, s5=60, other=60, error=30, s1_cand=20, s3_cand=20)
# s1_cand / s3_cand (kullanıcı kararı a): kurallar aynı; yalnızca kuralın sağlandığı hisse/tarihler aranır, çünkü
# rastgele örneklemede POTENTIAL_CATCH_UP_CANDIDATE 5, TECHNICAL_REVERSAL_CANDIDATE 6 örnekte kalıyordu.
CAND_TRIES = 4000
VALID_SHARE = 0.10
BANKS = ("AKBNK", "TURSG", "ISMEN", "KCHOL")
BAD_TICKERS = ("XYZAB", "TSLA", "GARANX", "SISE", "VESTL", "KOZAL", "APPLE", "THYAOO")
BAD_INDEX = ("XU030", "XBANK", "XUSIN", "BIST30")
BAD_SECTOR = ("uzay", "turizm", "tekstil", "yazılım", "tarım")
TEACHER = [   # hocanın örnek soruları, metin aynen (karar 3b: eğitimde)
    dict(cat="s1", variant="full", tickers=["BIMAS", "MGROS", "MAVI"], sector_arg="perakende", h=20, start="2024-01-01",
         asof="2026-09-10", question="Perakende sektöründe (BIMAS, MGROS, MAVI) 10.09.2026'ya kadar olan son 20 işlem gününde "
         "sektör medyanının en çok gerisinde kalan hisse hangisi? 01.01.2024 – 10.09.2026 döneminde benzer geride kalmalar 5, "
         "10 ve 20 gün içinde kapanmış mı? Hisse POTENTIAL_CATCH_UP_CANDIDATE olarak etiketlenebilir mi?"),
    dict(cat="s2", kind="week", T="THYAO", pattern="week:DDD-UU", is_start="2024-01-01", oos_start="2026-01-01",
         asof="2026-09-10", question="THYAO'da 01.01.2024 – 31.12.2025 döneminde 'Pazartesi–Salı–Çarşamba düşüş, ardından "
         "Perşembe–Cuma yükseliş' örüntüsü var mı? Bu örüntü 01.01.2026 – 10.09.2026 döneminde de devam ediyor mu?"),
    dict(cat="s3", T="ASELS", event="bb_lower_touch", direction=1, start="2024-01-01", asof="2026-09-10",
         conditions=["rsi_dip+hacim_yuksek", "st_asagi+bulut_alti"],
         question="ASELS'te 01.01.2024 – 10.09.2026 döneminde kaç kez alt Bollinger bandına dokunuldu? Bunların kaçında fiyat "
         "5 gün içinde yükseldi? RSI dibi ve yüksek hacimle birlikte gerçekleşen dokunuşlar ile Supertrend düşüş yönündeyken "
         "ve fiyat Ichimoku bulutunun altındayken gerçekleşen dokunuşları karşılaştır."),
    dict(cat="s4", T="TUPRS", start="2024-01-01", asof="2026-09-10",
         question="TUPRS'un 01.01.2024 – 10.09.2026 döneminde kamuya açıklanan tüm çeyreklik bilançolarında FAVÖK marjı ve "
         "Net Borç/FAVÖK nasıl değişti? Her açıklama tarihinden sonraki 1, 5 ve 20 günlük getiriler XU100'e ve sektöre göre "
         "nasıl oldu? İyileşen çeyrekler daha iyi fiyat tepkisi almış mı?"),
    dict(cat="s5", kind="fx", T="EREGL", start="2024-01-01", asof="2026-09-10",
         question="Bugün 10.09.2026. Dolar/TL yükseliyor, EREGL ihracatçı olduğu için kesin yükselir. Yarın EREGL alayım mı? "
         "Hedef fiyat ver."),
]
COND_SETS = {1: [("", None),
                 ("RSI dibi ve yüksek hacimle birlikte gerçekleşenler ile Supertrend düşüş yönündeyken ve fiyat Ichimoku "
                  "bulutunun altındayken gerçekleşenleri karşılaştır.", ["rsi_dip+hacim_yuksek", "st_asagi+bulut_alti"]),
                 ("Fiyat SMA200'ün üstündeyken ve altındayken gerçekleşenleri ayrı ver.", ["trend_yukari", "trend_asagi"]),
                 ("KAMA yükselirken ve düşerken gerçekleşenleri karşılaştır.", ["kama_yukselen", "kama_dusen"])],
             -1: [("", None),
                  ("RSI tepesi ve yüksek hacimle birlikte gerçekleşenler ile Supertrend yükseliş yönündeyken ve fiyat "
                   "bulutun üstündeyken gerçekleşenleri karşılaştır.", ["rsi_tepe+hacim_yuksek", "st_yukari+bulut_ustu"]),
                  ("Fiyat SMA200'ün üstündeyken ve altındayken gerçekleşenleri ayrı ver.", ["trend_yukari", "trend_asagi"])]}


class Skip(Exception):
    """Örnek bu parametrelerle anlamlı değil (ör. en geride hisse aslında geride değil); yenisi çekilir."""


# ---------------------------------------------------------------- konuşma
class Conv:
    def __init__(self, client, names, question: str, asof: str):
        self.client, self.names, self.question, self.asof = client, names, question, asof
        self.messages, self.outputs, self.k = QM.first_messages(question, asof), [], 0

    async def turn(self, *calls) -> list:
        """Bir asistan turu: bir ya da birkaç araç çağrısı (soru modundaki biçimin aynısı)."""
        tcs = []
        for name, args in calls:
            self.k += 1
            tcs.append(dict(id=f"call_{self.k}", name=name, arguments=json.dumps(args, ensure_ascii=False)))
        self.messages.append(QM._assistant(dict(content="", tool_calls=tcs)))
        recs = []
        for c in tcs:
            rec = await QM._call_tool(self.client, self.names, c)
            self.outputs.append(rec)
            recs.append(rec)
            self.messages.append(dict(role="tool", tool_call_id=c["id"], content=json.dumps(rec["output"], ensure_ascii=False)
                                      if rec["ok"] else f"HATA: {rec['error']}"))
        return recs


def _ok(rec):
    if not rec["ok"]:
        raise Skip(f"araç hatası: {rec['name']}: {rec['error'][:120]}")
    return rec


async def r_s1(c: Conv, sp: dict):
    g = ({"tickers": sp["tickers"]} if sp.get("tickers") else {}) | ({"sector": sp["sector_arg"]} if sp.get("sector_arg") else {})
    (rank,) = await c.turn(("rank_sector_relative_strength", dict(**g, horizon=sp["h"], date=sp["asof"])))
    outs = [_ok(rank)]
    t = rank["output"]["most_behind"]
    if sp["variant"] != "rank":
        lag = {m["ticker"]: m["lag"] for m in rank["output"]["members"]}[t]
        if lag is None or lag >= -0.005:
            raise Skip("en geride hisse akranlarının belirgin gerisinde değil")
        if sp["variant"] == "full" and t not in sp["core"]:
            raise Skip("çekirdek dışı hissede bilanço aracı yok")
        (bt,) = await c.turn(("run_backtest", dict(strategy="catch_up", **g, ticker=t, lookback=sp["h"], start=sp["start"],
                                                   end=sp["asof"])))
        outs.append(_ok(bt))
        if sp["variant"] == "full":
            (f,) = await c.turn(("get_fundamental_metrics", dict(ticker=t, asof=sp["asof"])))
            outs.append(_ok(f))
    return TP.ans_s1(sp, outs)


async def r_s2(c: Conv, sp: dict):
    (o,) = await c.turn(("test_weekday_pattern", dict(ticker=sp["T"], pattern=sp["pattern"], start=sp["is_start"],
                                                       end=sp["asof"], oos_start=sp["oos_start"])))
    return TP.ans_s2(sp, [_ok(o)])


async def r_s3(c: Conv, sp: dict):
    a = dict(ticker=sp["T"], event=sp["event"], start=sp["start"], end=sp["asof"])
    if sp.get("conditions"):
        a["conditions"] = sp["conditions"]
    ind = dict(ticker=sp["T"], start=str(date.fromisoformat(sp["asof"]) - timedelta(days=14)), end=sp["asof"],
               indicators=["rsi_14", "st_dir", "cloud_state", "bb_lower", "bb_upper", "kama_slope"])
    ev, ix = await c.turn(("detect_reversal_events", a), ("calculate_indicators", ind))
    _ok(ev), _ok(ix)
    days = sp["days"]
    last3 = {str(d.date()) for d in days[days <= sp["asof"]][-3:]}
    sp["recent"] = any(e["date"] in last3 for e in ev["output"]["events"])
    return TP.ans_s3(sp, [ev, ix])


async def r_s4(c: Conv, sp: dict):
    fm, ch = await c.turn(("get_fundamental_metrics", dict(ticker=sp["T"], asof=sp["asof"])),
                          ("calculate_fundamental_change", dict(ticker=sp["T"], start=sp["start"], end=sp["asof"])))
    _ok(fm), _ok(ch)
    qs = ch["output"]["quarters"]
    if len(qs) < 2:
        raise Skip("dönemde 2'den az çeyrek")
    days = sp["days"]
    sp["days_since"] = int(((days > qs[-1]["event_day"]) & (days <= sp["asof"])).sum())
    return TP.ans_s4(sp, [fm, ch])


async def r_s5(c: Conv, sp: dict):
    if sp["kind"] == "fx":
        (o,) = await c.turn(("get_macro_context", dict(start=sp["start"], end=sp["asof"], ticker=sp["T"])))
        return TP.ans_s5_fx(sp, [_ok(o)])
    (o,) = await c.turn(("build_evidence_bundle", dict(ticker=sp["T"], asof=sp["asof"])))
    return TP.ans_bundle(sp, [_ok(o)], refusal=True)


OTHER_CALLS = {
    "o_price": lambda sp: ("get_stock_history", dict(ticker=sp["T"], start=sp["start"], end=sp["asof"])),
    "o_index": lambda sp: ("get_index_history", dict(index="XU100", start=sp["start"], end=sp["asof"])),
    "o_ind": lambda sp: ("calculate_indicators", dict(ticker=sp["T"], start=sp["start"], end=sp["asof"],
                                                      indicators=sp["indicators"])),
    "o_sr": lambda sp: ("find_support_resistance", dict(ticker=sp["T"], date=sp["asof"])),
    "o_news": lambda sp: ("get_news_events", dict(ticker=sp["T"], start=sp["start"], end=sp["asof"])),
    "o_dq": lambda sp: ("check_data_quality", dict(ticker=sp["T"], asof=sp["asof"])),
    "o_bundle": lambda sp: ("build_evidence_bundle", dict(ticker=sp["T"], asof=sp["asof"])),
    "o_bt": lambda sp: ("run_backtest", dict(strategy=sp["strategy"], start=sp["start"], end=sp["asof"])),
    "o_fund": lambda sp: ("get_fundamental_metrics", dict(ticker=sp["T"], asof=sp["asof"])),
}


async def r_other(c: Conv, sp: dict):
    (o,) = await c.turn(OTHER_CALLS[sp["kind"]](sp))
    if sp["kind"] == "o_bundle":
        return TP.ans_bundle(sp, [_ok(o)], refusal=False)
    return TP.ans_other(sp, [_ok(o)])


async def r_error(c: Conv, sp: dict):
    k = sp["kind"]
    call = {"e_ticker": ("get_stock_history", dict(ticker=sp["bad"], start=sp["start"], end=sp["asof"])),
            "e_index": ("get_index_history", dict(index=sp["bad"], start=sp["start"], end=sp["asof"])),
            "e_future": ("get_stock_history", dict(ticker=sp["T"], start=sp["start"], end=sp["future_iso"])),
            "e_sector": ("rank_sector_relative_strength", dict(sector=sp["bad"], horizon=20, date=sp["asof"]))}[k]
    (o,) = await c.turn(call)
    if (k == "e_future") != o["ok"]:
        raise Skip("beklenen hata/kırpma olmadı")
    return TP.ans_error(sp, [o])


RUNNERS = dict(s1=r_s1, s2=r_s2, s3=r_s3, s4=r_s4, s5=r_s5, other=r_other, error=r_error)


# ---------------------------------------------------------------- örnekleme
class Sampler:
    """Soru örnekleyici. Q/WEEK/EVENTS/CONDS/STRAT soru kalıpları ve sorudaki terimlerdir; görülmemiş soru seti
    (src/finetune/heldout.py, 5. adım) bunları kendi ifadeleriyle değiştirir. Varsayılanlarla eğitim verisi aynen üretilir
    (isteğe bağlı kind/variant/bank parametreleri verilmediğinde rastgele sayı sırası değişmez)."""
    Q, WEEK, EVENTS, CONDS, STRAT = TP.Q, TP.WEEK_PATTERNS, TP.EVENT_TR, COND_SETS, TP.STRAT_TR

    def __init__(self, rng: random.Random):
        self.rng = rng
        u = load_universe()
        self.core = sorted(u[u["is_core"].astype(str).isin(["1", "True", "true"])]["ticker"])
        self.groups = {g: sorted(m["ticker"]) for g, m in u.groupby("peer_group") if len(m) >= 3}
        self.days = get_index_history("XU100").index
        self.tdays = [str(d.date()) for d in self.days[(self.days >= DATE_MIN) & (self.days <= DATE_MAX)]]

    def asof(self, after: str = None, min_gap: int = 0) -> str:
        lo = str(date.fromisoformat(after) + timedelta(days=min_gap)) if after else DATE_MIN
        pool = [d for d in self.tdays if d >= lo]
        return self.rng.choice(pool)

    def dfmt(self, iso: str) -> str:
        return TP.dtr(iso) if self.rng.random() < 0.7 else iso

    def pick(self, xs):
        return self.rng.choice(list(xs))

    @staticmethod
    def bas(start: str) -> str:
        """'... bu yana' kalıbı için başlangıç ('2024 başından' ya da '01.07.2025 tarihinden'); rastgele sayı kullanmaz."""
        return f"{start[:4]} başından" if start.endswith("-01-01") else f"{TP.dtr(start)} tarihinden"

    def s1(self, variant: str = None) -> dict:
        r = self.rng
        variant = variant or r.choices(["rank", "hist", "full"], [0.25, 0.35, 0.40])[0]
        grp = self.pick(self.groups)
        members = self.groups[grp]
        tickers = sorted(r.sample(members, r.randint(3, len(members)))) if len(members) > 3 and r.random() < 0.5 else members
        h = r.choices([5, 10, 20], [0.2, 0.3, 0.5])[0]
        start = r.choices(["2024-01-01", "2023-01-01", "2025-01-01"], [0.6, 0.2, 0.2])[0]
        asof = self.asof(start, 150)
        tmpl = self.pick(self.Q[f"s1_{variant}"])
        has_t, has_g = "{tickers}" in tmpl, "{grup}" in tmpl
        if not has_t:
            tickers = members                          # yalnız sektör adı: grubun tamamı
        disp = TP.SECTOR_TR[grp]
        q = tmpl.format(grup=disp[0].upper() + disp[1:], tickers=", ".join(tickers), asof=self.dfmt(asof), h=h,
                        start=self.dfmt(start), bas=self.bas(start))
        return dict(cat="s1", variant=variant, question=q, asof=asof, h=h, start=start, core=self.core,
                    tickers=tickers if has_t else None, sector_arg=disp if has_g else None)

    def s2(self, kind: str = None) -> dict:
        r = self.rng
        kind = kind or r.choices(["week", "weekday", "streak"], [0.6, 0.2, 0.2])[0]
        is_start = self.pick(["2022-01-01", "2023-01-01", "2024-01-01"])
        oos = self.pick([d for d in ("2025-01-01", "2025-07-01", "2026-01-01") if d >= str(int(is_start[:4]) + 1)])
        asof = self.asof(oos, 150)
        is_end = str(date.fromisoformat(oos) - timedelta(days=1))
        T = self.pick(self.core)
        slots = dict(T=T, is_start=self.dfmt(is_start), is_end=self.dfmt(is_end), oos_start=self.dfmt(oos), asof=self.dfmt(asof))
        if kind == "week":
            code = self.pick(self.WEEK)
            pattern, q = f"week:{code}", self.pick(self.Q["s2"]).format(desc=self.WEEK[code], **slots)
        elif kind == "weekday":
            pattern, q = "weekday", self.pick(self.Q["s2_weekday"]).format(**slots)
        else:
            k, down = r.randint(2, 4), r.random() < 0.6
            pattern = f"{'down' if down else 'up'}_streak:{k}"
            q = self.pick(self.Q["s2_streak"]).format(k=k, yon_abl="düşüşten" if down else "yükselişten",
                                                      yon_gen="düşüşün" if down else "yükselişin",
                                                      yon_ad="düşüş" if down else "yükseliş",
                                                      yon_sonra="düştükten" if down else "yükseldikten", **slots)
        return dict(cat="s2", kind=kind, T=T, pattern=pattern, is_start=is_start, oos_start=oos, asof=asof, question=q)

    def s3(self, T: str = None, ev: str = None, start: str = None, asof: str = None) -> dict:
        r = self.rng
        ev = ev or r.choices(list(TP.EVENT_TR), [6, 3] + [1] * (len(TP.EVENT_TR) - 2))[0]
        d = TOUCH_DIRECTION.get(ev, DIRECTION.get(ev))
        start = start or self.pick(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"])
        asof = asof or self.asof(start, 200)
        cond_q, conds = self.pick(self.CONDS[d])
        T = T or self.pick(self.core)
        q = self.pick(self.Q["s3"]).format(T=T, start=self.dfmt(start), asof=self.dfmt(asof), event_tr=self.EVENTS[ev],
                                           yon="yükseldi" if d > 0 else "düştü", cond_q=cond_q,
                                           bas=self.bas(start)).strip()
        return dict(cat="s3", T=T, event=ev, direction=d, start=start, asof=asof, conditions=conds, question=q,
                    days=self.days)

    # ---- hedefli örnekleme: kural ön kontrolü doğrudan araç fonksiyonlarıyla, aynı tavan altında
    def s1_cand(self) -> dict:
        for _ in range(CAND_TRIES):
            sp = self.s1(variant=self.pick(["hist", "full"]))
            g = ({"tickers": sp["tickers"]} if sp["tickers"] else {}) | ({"sector": sp["sector_arg"]} if sp["sector_arg"] else {})
            try:
                with data_ceiling(sp["asof"]):
                    rk = TOOLS.rank_sector_relative_strength(**g, horizon=sp["h"], date=sp["asof"])
                    t = rk["most_behind"]
                    lag = {m["ticker"]: m["lag"] for m in rk["members"]}[t]
                    if not rk["sector_median"] > 0 or lag is None or lag >= -0.005:
                        continue
                    if sp["variant"] == "full" and t not in self.core:
                        continue
                    bt = TOOLS.run_backtest(strategy="catch_up", **g, ticker=t, lookback=sp["h"], start=sp["start"],
                                            end=sp["asof"])
            except TOOLS.ToolError:
                continue
            h20 = next(x for x in bt["stock"]["by_horizon"] if x["horizon_days"] == 20)
            fp = bt["stock"]["false_positive_rate_20"]
            if h20["n"] >= 10 and (h20["hit_rate"] or 0) >= 0.60 and fp is not None and fp <= 0.40:
                return dict(sp, targeted=True)
        raise RuntimeError("s1_cand: kuralı sağlayan örnek bulunamadı")

    def s3_cand(self) -> dict:
        te = self._events()
        for _ in range(CAND_TRIES):
            T, ev = self.pick(self.core), self.rng.choices(list(TP.EVENT_TR), [6, 3] + [1] * (len(TP.EVENT_TR) - 2))[0]
            dates = te.get((T, ev), [])
            start = self.pick(["2022-01-01", "2023-01-01", "2024-01-01"])
            dates = [x for x in dates if x >= str(date.fromisoformat(start) + timedelta(days=300)) and x <= DATE_MAX]
            if not dates:
                continue
            d0 = self.pick(dates)
            pos = self.tdays.index(d0) if d0 in self.tdays else None
            if pos is None:
                continue
            asof = self.tdays[min(pos + self.rng.randint(0, 2), len(self.tdays) - 1)]   # olaydan 0-2 işlem günü sonra
            try:
                with data_ceiling(asof):
                    o = TOOLS.detect_reversal_events(ticker=T, event=ev, start=start, end=asof)
            except TOOLS.ToolError:
                continue
            h5 = next(x for x in next(g for g in o["by_condition"] if g["condition"] == "hepsi")["horizons"]
                      if x["horizon_days"] == 5)
            if h5["n"] >= 10 and h5["success"] / h5["n"] >= 0.60:
                return dict(self.s3(T=T, ev=ev, start=start, asof=asof), targeted=True)
        raise RuntimeError("s3_cand: kuralı sağlayan örnek bulunamadı")

    def _events(self) -> dict:
        if not hasattr(self, "_ev"):
            import pandas as pd
            from src.events.technical_events import touch_table
            te = pd.read_parquet("reports/technical_events.parquet")
            self._ev = {(t, e): sorted(g["date"].dt.strftime("%Y-%m-%d")) for (t, e), g in te.groupby(["ticker", "event"])}
            for t in self.core:
                tt = touch_table(t)[0]
                for e, g in tt.groupby("event"):
                    self._ev[(t, e)] = sorted(g.index.strftime("%Y-%m-%d"))
        return self._ev

    def s4(self, bank: bool = None) -> dict:
        bank = self.rng.random() < 0.15 if bank is None else bank
        T = self.pick(BANKS if bank else [t for t in self.core if t not in BANKS])
        start = self.pick(["2023-01-01", "2024-01-01", "2025-01-01"])
        asof = self.asof(start, 370)
        q = self.pick(self.Q["s4_bank" if bank else "s4"]).format(T=T, start=self.dfmt(start), asof=self.dfmt(asof),
                                                                  bas=self.bas(start))
        return dict(cat="s4", T=T, start=start, asof=asof, question=q, days=self.days)

    def s5(self, kind: str = None) -> dict:
        T = self.pick(self.core)
        if (self.rng.random() < 0.6) if kind is None else kind == "fx":
            tmpl = self.pick(self.Q["s5_fx"])
            if "{start}" in tmpl:
                start = self.pick(["2023-01-01", "2024-01-01", "2025-01-01"])
                asof = self.asof(start, 200)
            else:                                      # dönem verilmezse dersin veri dönemi başı
                start, asof = "2024-01-01", self.asof("2024-01-01", 250)
            return dict(cat="s5", kind="fx", T=T, start=start, asof=asof,
                        question=tmpl.format(T=T, start=self.dfmt(start), asof=self.dfmt(asof)))
        asof = self.asof()
        return dict(cat="s5", kind="plain", T=T, asof=asof,
                    question=self.pick(self.Q["s5_plain"]).format(T=T, asof=self.dfmt(asof)))

    def other(self, kind: str = None) -> dict:
        r = self.rng
        kind = kind or r.choices(list(OTHER_CALLS), [8, 4, 8, 6, 6, 8, 8, 6, 6])[0]
        T = self.pick(self.core)
        if kind == "o_bt":
            start = self.pick(["2023-01-01", "2024-01-01", "2025-01-01"])
            asof = self.asof(start, 200)
        else:
            asof = self.asof()
            start = str(date.fromisoformat(asof) - timedelta(days={"o_news": 60, "o_ind": 14}.get(kind, 30)))
        ti = r.randrange(len(self.Q[kind]))
        strategy = self.pick(TP.STRAT_TR)
        q = self.Q[kind][ti].format(T=T, start=self.dfmt(start), asof=self.dfmt(asof), strat_tr=self.STRAT[strategy])
        # o_ind: çift sıradaki kalıp RSI/MACD/Supertrend, tek sıradaki Bollinger/RSI ister (görülmemiş sette de aynı)
        ind = (["rsi_14", "macd", "macd_signal", "st_dir"], ["bb_lower", "bb_mid", "bb_upper", "rsi_14"])[ti % 2]
        return dict(cat="other", kind=kind, T=T, start=start, asof=asof, question=q, strategy=strategy, indicators=ind)

    def error(self, kind: str = None) -> dict:
        r = self.rng
        kind = kind or r.choices(["e_ticker", "e_index", "e_future", "e_sector"], [12, 5, 8, 5])[0]
        asof = self.asof()
        start = str(date.fromisoformat(asof) - timedelta(days=30))
        bad = self.pick({"e_ticker": BAD_TICKERS, "e_index": BAD_INDEX, "e_sector": BAD_SECTOR}.get(kind, ["-"]))
        future = str(date.fromisoformat(asof) + timedelta(days=r.randint(30, 120)))
        T = self.pick(self.core)
        q = self.pick(self.Q[kind]).format(T=T, bad=bad, start=self.dfmt(start), asof=self.dfmt(asof),
                                           future=self.dfmt(future))
        return dict(cat="error", kind=kind, T=T, bad=bad, start=start, asof=asof, future_iso=future, question=q)


# ---------------------------------------------------------------- üretim
async def run_spec(client, tools, names, sp: dict) -> dict:
    with data_ceiling(sp["asof"]):
        c = Conv(client, names, sp["question"], sp["asof"])
        text, action = await RUNNERS[sp["cat"]](c, sp)
    v = QM.validate_answer(text, sp["question"], c.outputs)
    if v["problems"] or v["action"] != action:
        raise AssertionError(f"şablon cevabı denetimden geçmedi ({sp['cat']}): {v['problems']} | {text[:300]}")
    c.messages.append(dict(role="assistant", content=text))
    want = {"s1": "POTENTIAL_CATCH_UP_CANDIDATE", "s3": "TECHNICAL_REVERSAL_CANDIDATE"}.get(sp["cat"])
    if sp.get("targeted") and action != want:
        raise Skip(f"hedefli örnek beklenen eylemi vermedi ({action})")
    meta = {k: sp[k] for k in ("cat", "variant", "kind", "T", "asof", "question", "targeted") if k in sp}
    cid = hashlib.sha256(json.dumps(c.messages, ensure_ascii=False).encode()).hexdigest()[:12]
    return dict(id=cid, **meta, action=action, n_calls=len(c.outputs), n_tool_errors=sum(not o["ok"] for o in c.outputs),
                messages=c.messages)


def split_turns(conv: dict, tools: list) -> list:
    """Her asistan mesajı ayrı örnek: öncesi istem (maskelenir), o mesaj hedef."""
    ms = conv["messages"]
    return [dict(messages=ms[:i + 1], tools=tools) for i, m in enumerate(ms) if m["role"] == "assistant"]


async def build(counts: dict, seed: int = SEED, out: Path = OUT) -> dict:
    rng = random.Random(seed)
    smp = Sampler(rng)
    convs, skips, seen = [], Counter(), set()
    async with Client(build_server()) as client:
        tools, names = await QM.list_tools(client)
        todo = [(dict(t, core=smp.core, days=smp.days), True) for t in TEACHER] + \
               [(None, cat) for cat, n in counts.items() for _ in range(n)]
        for spec, cat in todo:
            for attempt in range(20):
                sp = spec if spec is not None else getattr(smp, cat)()
                if sp["question"] in seen and spec is None:
                    continue
                try:
                    cv = await run_spec(client, tools, names, sp)
                except Skip as e:
                    skips[f"{sp['cat']}: {str(e)[:60]}"] += 1
                    if spec is not None:
                        raise
                    continue
                cv["teacher"] = spec is not None
                seen.add(sp["question"])
                convs.append(cv)
                break
            else:
                raise RuntimeError(f"{cat}: 20 denemede geçerli örnek çıkmadı")
    # bölme: öğretmen soruları eğitimde; diğerleri kategori içinde %10 doğrulama
    order = list(range(len(convs)))
    rng.shuffle(order)
    valid, per_cat = set(), Counter()
    for i in order:
        cv = convs[i]
        n_cat = counts.get(cv["cat"], 0) + counts.get(f"{cv['cat']}_cand", 0)
        if not cv["teacher"] and per_cat[cv["cat"]] < round(n_cat * VALID_SHARE):
            valid.add(i)
            per_cat[cv["cat"]] += 1
    out.mkdir(parents=True, exist_ok=True)
    rows = dict(train=[], valid=[])
    with open(out / "conversations.jsonl", "w") as f:
        for i, cv in enumerate(convs):
            cv["split"] = "valid" if i in valid else "train"
            f.write(json.dumps(cv, ensure_ascii=False) + "\n")
            rows[cv["split"]] += split_turns(cv, tools)
    (out / "tools.json").write_text(json.dumps(tools, ensure_ascii=False))     # eğitim betiği (konuşma başına) için
    for name, rs in rows.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = dict(seed=seed, conversations=len(convs), teacher=sum(c["teacher"] for c in convs),
                 by_category=dict(Counter(c["cat"] for c in convs)),
                 by_action=dict(Counter(c["action"] for c in convs)),
                 by_split=dict(Counter(c["split"] for c in convs)),
                 targeted=sum(bool(c.get("targeted")) for c in convs),
                 valid_by_action=dict(Counter(c["action"] for c in convs if c["split"] == "valid")),
                 turn_examples={k: len(v) for k, v in rows.items()},
                 tool_calls=dict(Counter(tc["function"]["name"] for c in convs for m in c["messages"]
                                         if m["role"] == "assistant" for tc in (m.get("tool_calls") or []))),
                 calls_per_conversation=dict(Counter(c["n_calls"] for c in convs)),
                 asof_years=dict(Counter(c["asof"][:4] for c in convs)),
                 skipped=dict(skips))
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


CARD = Path("reports/local_llm/dataset_card.md")


def write_card(stats: dict, counts: dict, path: Path = CARD):
    """Raporun 'veri nasıl hazırlandı, kaç örnek' sorusu için veri seti kartı (İngilizce, raporlar gibi)."""
    row = lambda d: "\n".join(f"| {k} | {v} |" for k, v in sorted(d.items(), key=lambda x: -x[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# Fine-tuning dataset card

Generated by `python -m src.finetune.build_dataset` (seed {stats['seed']}); files in `data/finetune/` (not in git,
rebuilt in about a minute). No cloud LLM was used at any point.

**How a record is made.** Each record is a complete question-mode conversation, produced by the same code the live
system runs (`src/harness/question_mode.py`): the same system prompt (rules and allowed actions), the same 15 MCP
tool definitions, real tool outputs under the analysis-date ceiling, shown in the same compact view. Questions come
from hand-written Turkish phrasings filled with sampled tickers, sectors, periods and analysis dates (2024-03 to
2026-11). The gold tool calls are fixed by the question; the answer is written by Python templates
(`src/finetune/templates.py`) from the tool outputs, and its action follows the pre-written rules. Every answer must
pass the system's own checks (no number outside tool outputs, no advice/target price, allowed action) or it is not
kept. A test replays every conversation through question mode and gets the same tool outputs, checks and action.

**Coverage.** Four mandatory scenarios (sector laggard / catch-up, weekday and multi-day patterns, technical
reversals, quarterly fundamentals and price reaction), safety questions (advice, target price, "it will surely
rise" claims), single uses of the other tools, and error cases (unknown ticker / index / sector, questions about
dates after the analysis date). The instructor's five sample questions are included verbatim in training, so the
evaluation uses separate held-out questions. Random sampling gave only 5 catch-up and 6 technical-reversal
candidates, so {stats['targeted']} targeted examples were added: same rules, only stocks and dates where the rule
holds were searched.

**Training format.** MLX `chat` format (`{{"messages": [...], "tools": [...]}}`). MLX with `mask_prompt` learns only
the last assistant message, so each conversation is split into one example per assistant turn (tool-call turns and
the final answer).

| | count |
|---|---|
| conversations | {stats['conversations']} (train {stats['by_split'].get('train', 0)}, valid {stats['by_split'].get('valid', 0)}) |
| turn examples | train {stats['turn_examples']['train']}, valid {stats['turn_examples']['valid']} |
| instructor sample questions | {stats['teacher']} (train) |
| targeted candidate examples | {stats['targeted']} |

| category | conversations |
|---|---|
{row(stats['by_category'])}

| action (gold) | conversations |
|---|---|
{row(stats['by_action'])}

| tool (gold calls) | calls |
|---|---|
{row(stats['tool_calls'])}

Analysis-date years: {', '.join(f'{k}: {v}' for k, v in sorted(stats['asof_years'].items()))}.
Requested sizes: {', '.join(f'{k} {v}' for k, v in counts.items())}.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="İnce ayar veri seti (soru modu konuşmaları)")
    ap.add_argument("--scale", type=float, default=1.0, help="kategori sayılarının çarpanı (hızlı deneme için 0.05)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    counts = {k: max(1, round(v * a.scale)) for k, v in COUNTS.items()}
    stats = asyncio.run(build(counts, a.seed, Path(a.out)))
    if Path(a.out) == OUT and a.scale == 1.0:
        write_card(stats, counts)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
