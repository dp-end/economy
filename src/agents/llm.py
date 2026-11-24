"""LLM katmanı: Gemini (openai SDK, OpenAI uyumlu uç nokta) + disk önbelleği + testler için sahte model.

LLM yalnızca yorum/açıklama yazar ve izinli araç isteyebilir; hesap yapmaz, eylem seçmez.
Anahtar yalnızca .env içinde (GEMINI_API_KEY); kod anahtarı yazdırmaz. Aynı istek ikinci kez
gelirse önbellekten döner (tekrarlanabilirlik ve ücretsiz katman kotası için).
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

CACHE = Path("data/llm_cache")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRY, RETRY_WAIT = 4, 20          # geçici hatada 20, 40, 60 sn bekle
RETRY_CODES = ("429", "500", "503")     # 429 kota, 500/503 Google tarafı geçici yoğunluk
TIMEOUT = 90                            # sn; cevap gelmezse kes ve yeniden dene (varsayılan 600 sn idi)


def clean_schema(x):
    """Gemini'nin OpenAI uyumlu ucunun reddedebileceği 'title' ve 'default' alanlarını atar (iç içe)."""
    if isinstance(x, dict):
        return {k: clean_schema(v) for k, v in x.items() if k not in ("title", "default")}
    if isinstance(x, list):
        return [clean_schema(v) for v in x]
    return x


def tool_schemas(tools, max_desc: int = 500) -> list:
    """MCP araç tanımlarını OpenAI 'tools' biçimine çevirir. max_desc: açıklama sınırı (Gemini 500; yerel model None)."""
    return [dict(type="function", function=dict(name=t.name, description=(t.description or "")[:max_desc],
                                                parameters=clean_schema(t.input_schema))) for t in tools]


def transient(e: Exception) -> bool:
    """Geçici hata mı: 429/500/503 ya da bağlantı kopması / zaman aşımı (APIConnectionError, APITimeoutError)."""
    from openai import APIConnectionError
    return isinstance(e, APIConnectionError) or any(c in str(e) for c in RETRY_CODES)


def assistant_message(r: dict) -> dict:
    """Modelin araç isteğini geçmişe ekler. Gemini 3 modelleri araç çağrısıyla bir 'thought_signature'
    gönderir (extra_content.google); sonraki istekte geri verilmezse 400 döner, bu yüzden aynen korunur."""
    calls = []
    for c in r["tool_calls"]:
        d = dict(id=c["id"], type="function", function=dict(name=c["name"], arguments=c["arguments"]))
        if c.get("extra_content"):
            d["extra_content"] = c["extra_content"]
        calls.append(d)
    return dict(role="assistant", content=r["content"] or None, tool_calls=calls)


class GeminiLLM:
    """Gerçek model. Bu sınıfı oluşturmak .env'deki anahtarı okur: komutu kullanıcı çalıştırır."""
    name = "gemini"

    def __init__(self, model: str = None, cache: Path = CACHE, temperature: float = 0.0):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.name = f"gemini:{self.model}"          # karar günlüğüne hangi modelin çalıştığı yazılır
        self.client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=BASE_URL, timeout=TIMEOUT)
        self.cache, self.temperature = cache, temperature
        self.calls = self.cache_hits = 0

    def chat(self, messages: list, tools: list = None) -> dict:
        """{'content', 'tool_calls': [{'id','name','arguments'}], 'cached', 'latency'}"""
        key = hashlib.sha256(json.dumps(dict(m=self.model, t=self.temperature, msg=messages, tools=tools),
                                        ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:24]
        f = self.cache / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return {**json.loads(f.read_text()), "cached": True, "latency": 0.0}
        t0 = time.time()
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature)
        if tools:
            kwargs["tools"] = tools
        waited = 0                                       # bekleme süresi gecikme ölçümüne katılmaz
        for i in range(MAX_RETRY):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:                       # RateLimitError, InternalServerError vb.
                if not transient(e) or i == MAX_RETRY - 1:
                    print("Gemini hata ayrıntısı:", getattr(e, "body", None) or e)   # anahtar içermez
                    raise
                wait = RETRY_WAIT * (i + 1)
                print(f"Geçici hata ({str(e)[:40]}...), {wait} sn bekleniyor, deneme {i + 2}/{MAX_RETRY}")
                time.sleep(wait)
                waited += wait
        msg = resp.choices[0].message
        out = dict(content=msg.content or "",
                   tool_calls=[dict(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}",
                                    extra_content=(c.model_extra or {}).get("extra_content"))
                               for c in (msg.tool_calls or [])])
        self.calls += 1
        self.cache.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, ensure_ascii=False))
        return {**out, "cached": False, "latency": round(time.time() - t0 - waited, 2)}


LOCAL_URL = "http://127.0.0.1:8080/v1"
LOCAL_MODEL = "models/Qwen3-4B-Instruct-2507-4bit"   # yerelde üretilen 4-bit (mlx_lm.convert -q, 4,5 bit/ağırlık)


class LocalLLM:
    """Yerel model (internet yok, anahtar yok): OpenAI uyumlu yerel sunucu (mlx_lm.server, Ollama, llama.cpp).
    Sıcaklık 0 = açgözlü çözme, aynı istem aynı cevabı verir. Önbellek yok: her cevap modelden gelir."""

    def __init__(self, model: str = None, base_url: str = None, temperature: float = 0.0, max_tokens: int = 900):
        from openai import OpenAI
        self.model = model or os.environ.get("LOCAL_MODEL", LOCAL_MODEL)
        self.base_url = base_url or os.environ.get("LOCAL_URL", LOCAL_URL)
        self.name = f"local:{self.model}"
        self.client = OpenAI(api_key="yerel", base_url=self.base_url, timeout=600)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.calls = self.cache_hits = 0

    def chat(self, messages: list, tools: list = None) -> dict:
        t0 = time.time()
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        self.calls += 1
        usage = getattr(resp, "usage", None)
        return dict(content=msg.content or "",
                    tool_calls=[dict(id=c.id or f"c{i}", name=c.function.name, arguments=c.function.arguments or "{}")
                                for i, c in enumerate(msg.tool_calls or [])],
                    cached=False, latency=round(time.time() - t0, 2),
                    prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None))


class ScriptLLM:
    """Önceden yazılmış cevapları sırayla veren sahte model (soru modu testleri ve modelsiz demo).
    Her adım: {'tool_calls': [(ad, argümanlar), ...]}, {'content': '...'} ya da {'raise': 'hata'}."""
    name = "script"

    def __init__(self, steps: list):
        self.steps, self.calls, self.cache_hits, self.log = list(steps), 0, 0, []

    def chat(self, messages: list, tools: list = None) -> dict:
        self.log.append(dict(messages=[dict(m) for m in messages], tools=[t["function"]["name"] for t in tools or []]))
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        if "raise" in step:                                  # sunucu hatası benzetimi (ör. bellek yetmedi, 404)
            raise RuntimeError(step["raise"])
        calls = [dict(id=f"c{self.calls}_{i}", name=n, arguments=json.dumps(a, ensure_ascii=False))
                 for i, (n, a) in enumerate(step.get("tool_calls", []))]
        return dict(content=step.get("content", ""), tool_calls=calls, cached=False, latency=0.0)


class MockLLM:
    """Ağa çıkmayan sahte model (testler ve anahtarsız deneme). Davranışı parametrelerle belirlenir:
    hallucinate=True -> açıklamaya kanıtta olmayan bir sayı ekler; fix_on_retry=True -> düzeltme isteğinde temizler;
    request_tool=(ad, argümanlar) -> ilk çağrıda bu aracı ister (izinsiz araç testi için)."""
    name = "mock"

    def __init__(self, hallucinate: bool = False, fix_on_retry: bool = True, request_tool: tuple = None,
                 json_action: str = "WATCH"):
        self.hallucinate, self.fix_on_retry, self.request_tool = hallucinate, fix_on_retry, request_tool
        self.json_action = json_action
        self.calls = self.cache_hits = 0
        self.log = []

    def chat(self, messages: list, tools: list = None) -> dict:
        self.calls += 1
        self.log.append(dict(messages=messages, tools=[t["function"]["name"] for t in tools or []]))
        last = messages[-1]
        if self.request_tool and last["role"] == "user" and self.calls == 1:
            name, args = self.request_tool
            return dict(content="", tool_calls=[dict(id="c1", name=name, arguments=json.dumps(args),
                                                     extra_content={"google": {"thought_signature": "MOCK_SIG"}})],
                        cached=False, latency=0.0)
        text = last.get("content") or ""
        retry = "DÜZELT" in text
        ids = re.findall(r"\bE\d+\b", text)
        body = "Kanıtlar " + ", ".join(dict.fromkeys(ids)) + " bu sonucu destekliyor." if ids else "Bağlam özeti hazır."
        if self.hallucinate and not (retry and self.fix_on_retry):
            body += " Hedef fiyat 987,65 TL."
        if "JSON" in messages[0].get("content", ""):
            body = json.dumps(dict(action=self.json_action, explanation=body), ensure_ascii=False)
        return dict(content=body, tool_calls=[], cached=False, latency=0.0)
