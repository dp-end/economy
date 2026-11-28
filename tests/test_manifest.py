"""Teslim paketi (src/submission/manifest.py): submission.json + ekler."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from src.agents import llm as L
from src.submission import manifest as M


def test_entry_points_exist():
    for group in M.ENTRY_POINTS.values():
        for path, role in group:
            assert Path(path).exists(), path


def test_hyperparameters_come_from_the_training_log():
    hp = M.hyperparameters()
    assert hp["base_model"] == dict(name=L.BASE_REPO, revision=L.BASE_REVISION)
    assert hp["method"]["rank"] == 16 and hp["method"]["lora_alpha"] == 16 * hp["method"]["scale"]
    assert len(hp["method"]["layers_to_transform"]) == hp["all_config"]["num_layers"] == 16
    assert hp["data"]["train_conversations"] + hp["data"]["valid_conversations"] == 655
    assert hp["result"]["val_loss"][0]["micro_step"] == 0 and hp["result"]["val_loss_all_valid"]["conversations"] == 59


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    out = tmp_path_factory.mktemp("submission")
    (out / "traces").mkdir()
    (out / "traces" / "x.json").write_text("{}")                     # önceden üretilmiş iz kaydı silinmemeli
    return out, M.build(out, allow_dirty=True)


def test_every_listed_file_exists_with_its_hash(package):
    out, m = package
    assert (out / "submission.json").exists() and json.loads((out / "submission.json").read_text()) == m
    for f in m["files"]:
        p = out / f["path"]
        assert p.stat().st_size == f["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
    assert m["traces"]["n"] == 1 and (out / "traces" / "x.json").exists()


def test_adapters_and_model_identity(package):
    out, m = package
    mlx = L.sha256_file(M.MLX_ADAPTER / "adapters.safetensors")
    assert m["adapter"]["primary"]["sha256"] == mlx                        # iz kaydındaki adapter_sha256 ile aynı
    assert m["adapter"]["original"]["sha256"] == L.sha256_file(M.PEFT_ADAPTER / "adapter_model.safetensors")
    assert m["base_model"]["name"] == "Qwen/Qwen3-4B-Instruct-2507" and m["runtime_weights"]["sha256"] == L.WEIGHTS_SHA256
    cfg = json.loads((out / "adapter/mlx/adapter_config.json").read_text())
    assert cfg["converted_from"]["peft_dir"] == "adapter/peft" and "/Users/" not in json.dumps(cfg)
    assert m["inference"]["cloud_api"] is False and m["inference"]["temperature"] == 0.0 and m["inference"]["seed"] == 42


def test_code_zip_is_the_commit(package):
    out, m = package
    names = zipfile.ZipFile(out / "code" / "code.zip").namelist()
    assert "src/harness/question_mode.py" in names and "src/mcp_server/tools.py" in names
    assert not any(n.startswith(("data/", "models/", "venv")) or n == ".env" for n in names)   # .env.example: boş şablon
