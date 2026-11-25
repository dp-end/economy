"""LocalLLM adaptör seçimi (kullanıcı kararı b, 25 Kasım 2026): adaptör her istekle açıkça gönderilir.

mlx_lm.server '--adapter-path'i yalnızca model adı 'default_model' iken uygular; gerçek model yolu gelince tabanı
sessizce yükler. Bu testler sunucusuz çalışır: sahte istemci gönderilen isteği yakalar.
"""
from types import SimpleNamespace as NS

import pytest

from src.agents.llm import LOCAL_MODEL, LocalLLM


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        call = NS(id="x1", function=NS(name="get_stock_history", arguments='{"ticker": "THYAO"}'))
        msg = NS(content="", tool_calls=[call])
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=10, completion_tokens=5))


def with_fake(llm):
    fake = FakeCompletions()
    llm.client = NS(chat=NS(completions=fake))
    return fake


@pytest.fixture
def adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"")
    return str(d)


def test_adapter_sent_with_real_model_name(adapter_dir):
    llm = LocalLLM(adapter=adapter_dir)
    fake = with_fake(llm)
    out = llm.chat([dict(role="user", content="soru")], tools=[{"type": "function"}])
    assert fake.kwargs["model"] == LOCAL_MODEL                     # 'default_model' değil
    assert fake.kwargs["extra_body"] == {"adapters": adapter_dir}
    assert llm.name == f"local:{LOCAL_MODEL}+{adapter_dir}"         # soru günlüğüne hangi model olduğu yazılır
    assert out["tool_calls"][0]["name"] == "get_stock_history"


def test_base_model_sends_no_adapter():
    llm = LocalLLM(adapter="none")
    fake = with_fake(llm)
    llm.chat([dict(role="user", content="soru")])
    assert "extra_body" not in fake.kwargs
    assert llm.adapter is None and llm.name == f"local:{LOCAL_MODEL}"


def test_missing_adapter_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="adaptör bulunamadı"):
        LocalLLM(adapter=str(tmp_path / "yok"))


def test_env_selects_adapter(monkeypatch, adapter_dir):
    monkeypatch.setenv("LOCAL_ADAPTER", "none")
    assert LocalLLM().adapter is None
    monkeypatch.setenv("LOCAL_ADAPTER", adapter_dir)
    assert LocalLLM().adapter == adapter_dir
    assert LocalLLM(adapter="none").adapter is None                 # açık parametre ortam değişkeninden önce gelir
