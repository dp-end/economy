"""GPU'da (PEFT) eğitilen LoRA adaptörünü MLX biçimine çevirir; Mac'te 4-bit taban üzerinde kullanılır.

PEFT:  base_model.model.model.layers.{i}.{blok}.{katman}.lora_A.weight  (rank, giriş)
       base_model.model.model.layers.{i}.{blok}.{katman}.lora_B.weight  (çıkış, rank),  ölçek = lora_alpha / r
MLX:   model.layers.{i}.{blok}.{katman}.lora_a  (giriş, rank)
       model.layers.{i}.{blok}.{katman}.lora_b  (rank, çıkış),  çıktı = scale * (x @ lora_a) @ lora_b
Yani iki matris de devrik alınır ve scale = lora_alpha / r. MLX LoRA'yı son num_layers katmanın bütün doğrusal
katmanlarına koyar (v1: 16 katman x 7 = 224 anahtar); çevrilen anahtarlar bununla birebir aynı olmalı, değilse hata.

Çalıştırma (proje kökü):  python -m src.finetune.peft_to_mlx <peft klasörü> <mlx klasörü> [--model ...]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

KEY = re.compile(r"^base_model\.model\.(model\.layers\.(\d+)\.\w+\.\w+)\.lora_([AB])\.weight$")
LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
          "mlp.up_proj", "mlp.down_proj")


def convert_tensors(peft: dict) -> dict:
    out = {}
    for k, v in peft.items():
        m = KEY.match(k)
        if not m:
            raise ValueError(f"beklenmeyen PEFT anahtarı: {k}")
        out[f"{m.group(1)}.lora_{m.group(3).lower()}"] = np.ascontiguousarray(v.T).astype(np.float32)
    return out


def expected_keys(n_total: int, num_layers: int) -> set:
    return {f"model.layers.{i}.{lin}.lora_{ab}" for i in range(n_total - num_layers, n_total) for lin in LINEAR
            for ab in "ab"}


def convert(src: Path, dst: Path, model: str, n_total_layers: int = 36) -> dict:
    from safetensors.numpy import load_file, save_file   # yalnız venv-llm'de var; testler convert_tensors'ı kullanır
    pc = json.loads((src / "adapter_config.json").read_text())
    r, alpha = pc["r"], pc["lora_alpha"]
    layers = pc.get("layers_to_transform") or list(range(n_total_layers))
    num_layers = len(layers)
    if sorted(layers) != list(range(n_total_layers - num_layers, n_total_layers)):
        raise ValueError(f"MLX yalnızca son katmanları destekler; PEFT katmanları: {layers}")
    t = convert_tensors(load_file(str(src / "adapter_model.safetensors")))
    want = expected_keys(n_total_layers, num_layers)
    if set(t) != want:
        raise ValueError(f"anahtarlar uyuşmuyor: eksik {sorted(want - set(t))[:3]}, fazla {sorted(set(t) - want)[:3]}")
    for k, v in t.items():
        if v.shape[1 if k.endswith("lora_a") else 0] != r:
            raise ValueError(f"rank uyuşmuyor: {k} {v.shape}")
    dst.mkdir(parents=True, exist_ok=True)
    save_file(t, str(dst / "adapters.safetensors"))
    lora = dict(rank=r, scale=alpha / r, dropout=pc.get("lora_dropout", 0.0))
    state = {}
    if (src / "state.json").exists():
        state = json.loads((src / "state.json").read_text())
    cfg = dict(fine_tune_type="lora", num_layers=num_layers, lora_parameters=lora, model=model,
               converted_from=dict(peft_dir=str(src), base=state.get("base") or pc.get("base_model_name_or_path"),
                                   revision=state.get("revision"), training=state.get("cfg")))
    (dst / "adapter_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    return dict(keys=len(t), num_layers=num_layers, rank=r, scale=alpha / r, out=str(dst))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PEFT LoRA -> MLX LoRA")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--model", default="models/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--layers", type=int, default=36, help="tabandaki toplam katman sayısı (Qwen3-4B: 36)")
    a = ap.parse_args()
    print(json.dumps(convert(Path(a.src), Path(a.dst), a.model, a.layers), ensure_ascii=False))
