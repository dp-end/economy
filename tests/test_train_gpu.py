"""Colab eğitimi (train_gpu.py) ve PEFT -> MLX çevirisi: torch/mlx gerektirmeyen kısımlar.

Uçtan uca kontroller (25 Kasım 2026, venv-llm, 4 katmanlı minik Qwen3): parçalı kayıp ile tek parça logits_to_keep
kaybı birebir aynı; fp16 + GradScaler çalışıyor; --resume kaldığı adımdan sürüyor; rastgele bir PEFT adaptörü MLX'e
çevrilince logit farkı en fazla 1,4e-6 (adaptörün etkisi 1,14).
"""
import math

import numpy as np
import pytest

from src.finetune.masks import target_mask
from src.finetune.peft_to_mlx import LINEAR, convert_tensors, expected_keys
from src.finetune.train_gpu import CFG, lr_schedule, target_positions


def test_target_positions_match_mlx_mask():
    for n, spans in [(10, [(3, 6)]), (20, [(0, 4), (8, 12), (15, 20)]), (7, [])]:
        m = target_mask(n, spans)
        assert target_positions(n, spans) == [j for j, v in enumerate(m) if v]


def test_lr_schedule_same_shape_as_mlx():
    at = lr_schedule(CFG, 232)                       # 464 konuşma / gradyan biriktirme 2
    lr, lo = CFG["learning_rate"], CFG["min_lr"]
    assert math.isclose(at(0), lr / 10) and math.isclose(at(CFG["warmup_steps"]), lr)
    assert math.isclose(at(231 + 1), lo) and at(5) < at(9) and at(100) > at(200)


def test_peft_keys_become_mlx_keys_transposed():
    r, d_in, d_out = 4, 6, 5
    a = np.arange(r * d_in, dtype=np.float32).reshape(r, d_in)
    b = np.arange(d_out * r, dtype=np.float32).reshape(d_out, r)
    t = convert_tensors({"base_model.model.model.layers.35.self_attn.q_proj.lora_A.weight": a,
                         "base_model.model.model.layers.35.self_attn.q_proj.lora_B.weight": b})
    assert np.array_equal(t["model.layers.35.self_attn.q_proj.lora_a"], a.T)
    assert np.array_equal(t["model.layers.35.self_attn.q_proj.lora_b"], b.T)
    x = np.ones((1, d_in), dtype=np.float32)           # PEFT: B(A x) = MLX: (x a) b
    assert np.allclose(x @ a.T @ b.T, x @ t["model.layers.35.self_attn.q_proj.lora_a"]
                       @ t["model.layers.35.self_attn.q_proj.lora_b"])
    with pytest.raises(ValueError):
        convert_tensors({"base_model.model.lm_head.lora_A.weight": a})


def test_expected_keys_like_v1_adapter():
    k = expected_keys(36, 16)
    assert len(k) == 16 * len(LINEAR) * 2 == 224          # v1 adaptöründeki anahtar sayısı
    assert "model.layers.20.mlp.down_proj.lora_a" in k and not any(".19." in x for x in k)


def test_unpack_codes_low_bits_first():
    """v3: MLX paketlemesi, her uint32 sözcükte 8 adet 4 bitlik kod, en düşük bitler ilk sütun (numpy ile; torch'ta
    aynı fonksiyon kullanılıyor, gerçek dosyada mx.dequantize ile karşılaştırması Mac'te yapıldı)."""
    from src.finetune.train_gpu import QBITS, unpack_codes
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=(3, 16), dtype=np.int64)
    words = np.zeros((3, 2), dtype=np.uint32)
    for j in range(16):
        words[:, j // 8] |= (codes[:, j].astype(np.uint32) << np.uint32(4 * (j % 8)))
    w64 = words.view(np.int32).astype(np.int64)                 # torch yolu: uint32 -> int32 görünümü -> int64
    out = unpack_codes(w64, np.arange(0, 32, QBITS, dtype=np.int64), (1 << QBITS) - 1)
    assert (out == codes).all()


def test_mlx4_source_pinned():
    from src.finetune import train_gpu as G
    assert len(G.MLX4_REVISION) == 40 and len(G.MLX4_SHA256) == 64 and (G.QBITS, G.QGROUP) == (4, 64)
