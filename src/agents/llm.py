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
MAX_RETRY, RETRY_WAIT = 4, 20          # 429 (kota) gelirse 20, 40, 60 sn bekle


def clean_schema(x):
    """Gemini'nin OpenAI uyumlu ucunun reddedebileceği 'title' ve 'default' alanlarını atar (iç içe)."""
    if isinstance(x, dict):
        return {k: clean_schema(v) for k, v in x.items() if k not in ("title", "default")}
    if isinstance(x, list):
        return [clean_schema(v) for v in x]
    return x


def tool_schemas(tools) -> list:
    """MCP araç tanımlarını OpenAI 'tools' biçimine çevirir."""
    return [dict(type="function", function=dict(name=t.name, description=(t.description or "")[:500],
                                                parameters=clean_schema(t.input_schema))) for t in tools]


class GeminiLLM:
    """Gerçek model. Bu sınıfı oluşturmak .env'deki anahtarı okur: komutu kullanıcı çalıştırır."""
    name = "gemini"

    def __init__(self, model: str = None, cache: Path = CACHE, temperature: float = 0.0):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=BASE_URL)
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
        for i in range(MAX_RETRY):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:                       # openai.RateLimitError vb.
                if "429" not in str(e) or i == MAX_RETRY - 1:
                    raise
                time.sleep(RETRY_WAIT * (i + 1))
        msg = resp.choices[0].message
        out = dict(content=msg.content or "",
                   tool_calls=[dict(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
                               for c in (msg.tool_calls or [])])
        self.calls += 1
        self.cache.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, ensure_ascii=False))
        return {**out, "cached": False, "latency": round(time.time() - t0, 2)}


class MockLLM:
    """Ağa çıkmayan sahte model (testler ve anahtarsız deneme). Davranışı parametrelerle belirlenir:
    hallucinate=True -> açıklamaya kanıtta olmayan bir sayı ekler; fix_on_retry=True -> düzeltme isteğinde temizler;
    request_tool=(ad, argümanlar) -> ilk çağrıda bu aracı ister (izinsiz araç testi için)."""
    name = "mock"

    def __init__(self, hallucinate: bool = False, fix_on_retry: bool = True, request_tool: tuple = None):
        self.hallucinate, self.fix_on_retry, self.request_tool = hallucinate, fix_on_retry, request_tool
        self.calls = self.cache_hits = 0
        self.log = []

    def chat(self, messages: list, tools: list = None) -> dict:
        self.calls += 1
        self.log.append(dict(messages=messages, tools=[t["function"]["name"] for t in tools or []]))
        last = messages[-1]
        if self.request_tool and last["role"] == "user" and self.calls == 1:
            name, args = self.request_tool
            return dict(content="", tool_calls=[dict(id="c1", name=name, arguments=json.dumps(args))],
                        cached=False, latency=0.0)
        text = last.get("content") or ""
        retry = "DÜZELT" in text
        ids = re.findall(r"\bE\d+\b", text)
        body = "Kanıtlar " + ", ".join(dict.fromkeys(ids)) + " bu sonucu destekliyor." if ids else "Bağlam özeti hazır."
        if self.hallucinate and not (retry and self.fix_on_retry):
            body += " Hedef fiyat 987,65 TL."
        return dict(content=body, tool_calls=[], cached=False, latency=0.0)
