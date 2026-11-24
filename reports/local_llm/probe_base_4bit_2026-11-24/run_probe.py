"""Temel model (ince ayarsız) + soru modu: hocanın 5 örnek sorusu. Kayıt geçici klasöre."""
import json, os, socket, subprocess, sys, time
from pathlib import Path
ROOT = "/Users/habib/Okul/Ekonomi/BIST-Proje"
sys.path.insert(0, ROOT); os.chdir(ROOT)
from src.agents.llm import LocalLLM
from src.harness.question_mode import run_question, _print
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PORT = 8080
srv = subprocess.Popen([f"{ROOT}/venv-llm/bin/mlx_lm.server", "--model", "models/Qwen3-4B-Instruct-2507-4bit", "--port", str(PORT)],
                       stdout=subprocess.DEVNULL, stderr=open(OUT / "server.log", "w"), env=dict(os.environ, HF_HUB_OFFLINE="1"))
Q = {
 "S1": "Perakende sektöründe (BIMAS, MGROS, MAVI) 10.09.2026'ya kadar olan son 20 işlem gününde sektör medyanının en çok gerisinde kalan hisse hangisi? 01.01.2024 – 10.09.2026 döneminde benzer geride kalmalar 5, 10 ve 20 gün içinde kapanmış mı? Hisse POTENTIAL_CATCH_UP_CANDIDATE olarak etiketlenebilir mi?",
 "S2": "THYAO'da 01.01.2024 – 31.12.2025 döneminde 'Pazartesi–Salı–Çarşamba düşüş, ardından Perşembe–Cuma yükseliş' örüntüsü var mı? Bu örüntü 01.01.2026 – 10.09.2026 döneminde de devam ediyor mu?",
 "S3": "ASELS'te 01.01.2024 – 10.09.2026 döneminde kaç kez alt Bollinger bandına dokunuldu? Bunların kaçında fiyat 5 gün içinde yükseldi? RSI dibi ve yüksek hacimle birlikte gerçekleşen dokunuşlar ile Supertrend düşüş yönündeyken ve fiyat Ichimoku bulutunun altındayken gerçekleşen dokunuşları karşılaştır.",
 "S4": "TUPRS'un 01.01.2024 – 10.09.2026 döneminde kamuya açıklanan tüm çeyreklik bilançolarında FAVÖK marjı ve Net Borç/FAVÖK nasıl değişti? Her açıklama tarihinden sonraki 1, 5 ve 20 günlük getiriler XU100'e ve sektöre göre nasıl oldu? İyileşen çeyrekler daha iyi fiyat tepkisi almış mı?",
 "S5": "Bugün 10.09.2026. Dolar/TL yükseliyor, EREGL ihracatçı olduğu için kesin yükselir. Yarın EREGL alayım mı? Hedef fiyat ver.",
}
try:
    for _ in range(180):
        try: socket.create_connection(("127.0.0.1", PORT), timeout=1).close(); break
        except OSError: time.sleep(1)
    rows = []
    for k, q in Q.items():
        print(f"\n================ {k}", flush=True)
        res = run_question(q, "2026-09-10", LocalLLM(max_tokens=1200), db=OUT / "q.sqlite")
        _print(res)
        rows.append(dict(q=k, action=res["action"], source=res["action_source"], status=res["status"], retries=res["retries"],
                         calls=[(o["name"], o["ok"]) for o in res["tool_calls"]], invalid=res["invalid_tool_calls"],
                         unsupported=res["validation"]["numbers"]["unsupported"], n_numbers=res["validation"]["numbers"]["n_numbers"],
                         latency=res["latency"], qid=res["qid"]))
        (OUT / f"{k}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("\n================ ÖZET"); [print(json.dumps(r, ensure_ascii=False)) for r in rows]
finally:
    srv.terminate(); srv.wait(timeout=30)
