"""Teslim paketi (sunum duyurusu, 28 Kasım 2026): submission/ klasörü + submission.json manifesti + tek zip.

Duyurunun istediği ekler ve buradaki yerleri:
  Python kaynak kodu        code/code.zip (git archive, tam commit) + manifestte araç/harness/script yolları
  ince ayar dosyası         finetune/ (konuşmalar, modele verilen train/valid, araç tanımları, hyperparameters.json,
                            veri kartı, Kaggle eğitim günlükleri)
  model ağırlıkları         adapter/mlx (canlı sistemde çalışan, asıl) + adapter/peft (Kaggle eğitim çıktısı; MLX
                            bundan src/finetune/peft_to_mlx ile çevrildi); taban model adı, sürümü, kaynağı manifestte
  RAG dosyası               rag/knowledge_base.json (src/submission/knowledge_base, her paketlemede yeniden üretilir)
  iz kayıtları              traces/ (5 senaryo; ayrı adımda üretilir, bu script yalnızca listeler, silmez)
Kullanıcı kararları (28 Kasım 2026): iki adaptör (asıl MLX), eğitim dosyalarının hepsi + hiperparametre JSON,
code.zip + GitHub commit, submission/ klasörü (git'e girmez) + script + zip.

Çalıştırma:  python -m src.submission.manifest            # commit edilmemiş değişiklik varsa durur
             python -m src.submission.manifest --allow-dirty --no-zip    # deneme
"""
import argparse
import json
import platform
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

from src.agents import llm as L
from src.harness import question_mode as QM
from src.submission import knowledge_base as K

OUT = Path("submission")
DIST = Path("dist")
VERSION = 1
REPO_URL = "https://github.com/dp-end/economy"
MLX_ADAPTER = Path("models/adapters/bist-qwen3-4b-v4")
PEFT_ADAPTER = Path("models/adapters/bist-qwen3-4b-v4-peft")     # ~/Downloads/adapter_v4/adapter_v4 kopyası
FT_DATA = Path("data/finetune_v4")
TRAIN_LOG = Path("reports/local_llm/train_v4_log.jsonl")
SMOKE_LOG = Path("reports/local_llm/train_v4_smoke_log.jsonl")
DATASET_CARD = Path("reports/local_llm/dataset_card_v4.md")
EVAL_SUMMARY = Path("reports/local_llm/eval_v4/summary.md")
TRAIN_SCRIPT = "src/finetune/train_gpu.py"

ENTRY_POINTS = {
    "tools": [("src/mcp_server/tools.py", "15 deterministik MCP aracının gövdeleri (analiz tarihi tavanı burada)"),
              ("src/mcp_server/server.py", "MCP sunucusu: araçların kaydı")],
    "harness": [("src/harness/question_mode.py", "soru modu: sistem istemi, araç döngüsü, denetim, günlük, iz kaydı"),
                ("src/agents/validate.py", "sayı denetimi: cevaptaki her sayı araç çıktısında olmalı"),
                ("src/agents/llm.py", "yerel model istemcisi (LocalLLM): adaptör, sıcaklık, seed, model kimliği"),
                ("src/harness/state_machine.py", "durum makinesi (ilk harness, PDF §23)"),
                ("src/harness/quality_gate.py", "veri kalitesi ve risk kapısı"),
                ("src/harness/evidence.py", "kanıt paketi")],
    "fine_tuning": [("src/finetune/build_dataset.py", "eğitim konuşmalarını soru modunun koduyla üretir"),
                    ("src/finetune/dataset_v4.py", "v4 veri seti (hata-düzelt, desen cümleleri)"),
                    ("src/finetune/templates.py", "cevap şablonları (sayılar yalnızca araç çıktısından)"),
                    (TRAIN_SCRIPT, "LoRA eğitimi (PyTorch + PEFT, Kaggle T4)"),
                    ("src/finetune/kaggle_train_v4.ipynb", "Kaggle defteri"),
                    ("src/finetune/peft_to_mlx.py", "PEFT adaptörü -> MLX adaptörü"),
                    ("src/finetune/evaluate.py", "taban / v1 / v4 karşılaştırması"),
                    ("src/finetune/heldout_v4.py", "görülmemiş 60 soru")],
    "submission": [("src/submission/knowledge_base.py", "RAG dosyası"),
                   ("src/submission/manifest.py", "bu paket")],
}
COMMANDS = {
    "model_server": ("HF_HUB_OFFLINE=1 venv-llm/bin/mlx_lm.server --model models/Qwen3-4B-Instruct-2507-4bit "
                     "--port 8080"),
    "ask": 'python -m src.harness.question_mode --asof 2026-09-10 "<soru>"',
    "ask_base_model": 'python -m src.harness.question_mode --adapter none --asof 2026-09-10 "<soru>"',
    "replay": "python -m src.harness.question_mode --replay <kayıt kimliği>",
    "mcp_demo": "python -m src.mcp_server.client_demo",
    "build_dataset": "python -m src.finetune.dataset_v4",
    "convert_adapter": ("python -m src.finetune.peft_to_mlx models/adapters/bist-qwen3-4b-v4-peft "
                        "models/adapters/bist-qwen3-4b-v4"),
    "evaluate": "python -m src.finetune.evaluate --out reports/local_llm/eval_v4",
    "knowledge_base": "python -m src.submission.knowledge_base",
    "package": "python -m src.submission.manifest",
    "tests": "python -m pytest -q",
}


# ---------------------------------------------------------------- yardımcılar
def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=60, check=True).stdout.strip()


def entry(out: Path, path: Path, role: str) -> dict:
    """Paketteki bir dosya: paket içi yol, görev, boyut, SHA-256."""
    return dict(path=str(path.relative_to(out)), role=role, bytes=path.stat().st_size, sha256=L.sha256_file(path))


def put(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def hyperparameters() -> dict:
    """Kaggle eğitim günlüğünden (train_v4_log.jsonl) ayarlar ve sonuçlar; elle yazılan sayı yok."""
    rows = [json.loads(line) for line in TRAIN_LOG.read_text().splitlines() if line.strip()]
    start = next(r for r in rows if r["kind"] == "start")
    done = next(r for r in rows if r["kind"] == "done")
    full = next(r for r in rows if r["kind"] == "val_full")
    cfg = start["cfg"]
    peft = json.loads((PEFT_ADAPTER / "adapter_config.json").read_text())
    stats = json.loads((FT_DATA / "stats.json").read_text())
    return dict(
        model_version="bist-qwen3-4b-v4",
        base_model=dict(name=start["model"], revision=start["revision"]),
        base_weights=dict(repo=L.WEIGHTS_REPO, revision=L.WEIGHTS_REVISION, sha256=L.WEIGHTS_SHA256,
                          note="4-bit MLX ağırlıkları eğitimde fp16'ya açıldı (dequantize); LoRA bu ağırlıkların "
                               "üzerinde eğitildi, canlı sistem aynı 4-bit dosyayla çalışır"),
        method=dict(type="LoRA", library=f"PEFT {peft['peft_version']}", task_type=peft["task_type"],
                    target_modules=sorted(peft["target_modules"]), layers_to_transform=peft["layers_to_transform"],
                    rank=peft["r"], lora_alpha=peft["lora_alpha"], scale=cfg["scale"], dropout=cfg["dropout"],
                    trainable_params=start["trainable_params"], dtype=start["dtype"]),
        optimizer=dict(name="Adam", betas=[0.9, 0.999], eps=1e-8, learning_rate=cfg["learning_rate"],
                       min_lr=cfg["min_lr"], schedule=f"{cfg['warmup_steps']} adım lr/10'dan lr'ye doğrusal ısınma, "
                                                      "sonra min_lr'ye kosinüs",
                       warmup_steps=cfg["warmup_steps"], grad_accumulation=cfg["grad_accumulation"],
                       epochs=cfg["epochs"], micro_steps=start["micro_steps"],
                       optimizer_steps=start["optimizer_steps"]),
        loss="yalnızca asistan tokenleri (araç çağrıları ve son cevap); hata-düzelt konuşmalarındaki bozuk çağrı "
             "kayba girmez",
        data=dict(train_conversations=start["train_conversations"], valid_conversations=start["valid_conversations"],
                  tokens_per_epoch=start["tokens_per_epoch"], max_seq_length=cfg["max_seq_length"],
                  dropped_over_max_seq=start["dropped_over_max_seq"], dataset_seed=stats["seed"],
                  by_category=stats["by_category"], by_action=stats["by_action"]),
        seed=cfg["seed"], all_config=cfg,
        hardware=dict(platform="Kaggle", gpu=start["gpu"], device=start["device"]),
        result=dict(minutes=done["minutes"], skipped_steps=done["skipped_steps"],
                    val_loss=[dict(micro_step=r["micro_step"], val_loss=r["val_loss"]) for r in rows
                              if r["kind"] == "val"],
                    val_loss_all_valid=dict(conversations=full["conversations"], val_loss=full["val_loss"])),
        training_script=TRAIN_SCRIPT, source_log=str(TRAIN_LOG))


def mlx_adapter_config(dst: Path):
    """MLX adaptör ayarı; 'peft_dir' kişisel klasör yolu yerine paketteki PEFT klasörünü gösterir (işlevi yok)."""
    cfg = json.loads((MLX_ADAPTER / "adapter_config.json").read_text())
    cfg.get("converted_from", {})["peft_dir"] = "adapter/peft"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    return dst


# ---------------------------------------------------------------- paket
def build(out: Path = OUT, allow_dirty: bool = False) -> dict:
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not allow_dirty:
        raise SystemExit("commit edilmemiş değişiklik var: code.zip commit'ten üretilir, önce commit et "
                         "(deneme için --allow-dirty)")
    for d in ("code", "adapter", "finetune", "rag", "evaluation"):     # traces/ ayrı adımda üretilir: silinmez
        shutil.rmtree(out / d, ignore_errors=True)
    files = []

    code = out / "code" / "code.zip"
    code.parent.mkdir(parents=True, exist_ok=True)
    git("archive", "--format=zip", "-o", str(code), commit)
    files.append(entry(out, code, "Python kaynak kodu (git archive, commit " + commit[:7] + ")"))
    files.append(entry(out, put(Path("requirements.txt"), out / "code" / "requirements.txt"), "Python bağımlılıkları"))

    mlx = [entry(out, put(MLX_ADAPTER / "adapters.safetensors", out / "adapter/mlx/adapters.safetensors"),
                 "LoRA adaptörü (MLX), canlı sistemde çalışan"),
           entry(out, mlx_adapter_config(out / "adapter/mlx/adapter_config.json"), "MLX adaptör ayarı")]
    peft = [entry(out, put(PEFT_ADAPTER / "adapter_model.safetensors", out / "adapter/peft/adapter_model.safetensors"),
                  "LoRA adaptörü (PEFT), Kaggle eğitim çıktısı"),
            entry(out, put(PEFT_ADAPTER / "adapter_config.json", out / "adapter/peft/adapter_config.json"),
                  "PEFT adaptör ayarı")]
    files += mlx + peft

    hp = out / "finetune" / "hyperparameters.json"
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(json.dumps(hyperparameters(), ensure_ascii=False, indent=1))
    ft = [entry(out, put(FT_DATA / "conversations.jsonl", out / "finetune/conversations.jsonl"),
                "655 tam konuşma (soru, araç çağrıları, araç çıktıları, cevap; kategori, aksiyon, split)"),
          entry(out, put(FT_DATA / "train.jsonl", out / "finetune/train.jsonl"),
                "eğitim örnekleri, modele verilen biçim (asistan turu başına bir örnek)"),
          entry(out, put(FT_DATA / "valid.jsonl", out / "finetune/valid.jsonl"), "doğrulama örnekleri"),
          entry(out, put(FT_DATA / "tools.json", out / "finetune/tools.json"), "eğitimde ve canlıda aynı 15 araç tanımı"),
          entry(out, put(FT_DATA / "stats.json", out / "finetune/stats.json"), "veri seti sayımları"),
          entry(out, hp, "yapılandırma ve hiperparametreler (eğitim günlüğünden)"),
          entry(out, put(DATASET_CARD, out / "finetune/dataset_card.md"), "veri kartı: kayıtların nasıl üretildiği"),
          entry(out, put(TRAIN_LOG, out / "finetune/train_log.jsonl"), "Kaggle eğitim günlüğü"),
          entry(out, put(SMOKE_LOG, out / "finetune/train_smoke_log.jsonl"), "Kaggle ön deneme günlüğü")]
    files += ft

    kb_path = out / "rag" / "knowledge_base.json"
    kb = K.build()
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    kb_path.write_text(json.dumps(kb, ensure_ascii=False, indent=1, allow_nan=False))
    rag = entry(out, kb_path, "RAG dosyası: bilgi tabanı belgeleri ve kaynak meta verileri")
    files.append(rag)

    ev = entry(out, put(EVAL_SUMMARY, out / "evaluation/eval_v4_summary.md"),
               "taban / v1 / v4 karşılaştırması (60 görülmemiş soru)")
    files.append(ev)

    traces = sorted((out / "traces").glob("*.json")) if (out / "traces").exists() else []
    tr = [entry(out, t, "iz kaydı") for t in traces]
    files += tr

    info = dict(L.LocalLLM(adapter=str(MLX_ADAPTER)).info())
    manifest = dict(
        manifest_version=VERSION, created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        project=dict(course="CSE-481 Engineering Economics", title="Integrated BIST 100 Agentic Financial Analytics "
                                                                  "Harness", team=None),
        code=dict(repository=REPO_URL, commit=commit, uncommitted_changes=dirty, archive=code.name,
                  python=platform.python_version(), server_software=info["server_software"],
                  entry_points={k: [dict(path=p, role=r) for p, r in v] for k, v in ENTRY_POINTS.items()},
                  commands=COMMANDS),
        base_model=dict(name=L.BASE_REPO, revision=L.BASE_REVISION, source=f"https://huggingface.co/{L.BASE_REPO}",
                        license="Apache-2.0"),
        runtime_weights=dict(repo=L.WEIGHTS_REPO, revision=L.WEIGHTS_REVISION, file="model.safetensors",
                             sha256=L.WEIGHTS_SHA256, source=f"https://huggingface.co/{L.WEIGHTS_REPO}",
                             quantization=info["quantization"], local_path=L.LOCAL_MODEL,
                             note="taban modelin MLX 4-bit hâli; yerel dosya bununla bit bit aynı. Teslim edilmez, "
                                  "kaynaktan indirilir."),
        adapter=dict(
            primary=dict(format="MLX (mlx-lm LoRA)", files=mlx, sha256=mlx[0]["sha256"],
                         note="canlı soru modu ve iz kayıtları bu dosyayla çalışır (iz kaydındaki adapter_sha256)"),
            original=dict(format="PEFT (Hugging Face)", files=peft, sha256=peft[0]["sha256"],
                          note="Kaggle eğitiminin çıktısı; MLX dosyası bundan çevrildi",
                          conversion=COMMANDS["convert_adapter"])),
        fine_tuning=dict(files=ft, hyperparameters=hp.name, method="LoRA, açılmış 4-bit ağırlıklar üzerinde",
                         no_cloud_llm="veri üretiminde ve eğitimde bulut LLM kullanılmadı"),
        inference=dict(offline=True, cloud_api=False, server=COMMANDS["model_server"], temperature=info["temperature"],
                       seed=info["seed"], max_tokens=info["max_tokens"], max_tool_calls=QM.MAX_CALLS,
                       correction_rounds=1),
        rag=dict(file=rag, n_documents=kb["n_documents"], groups=kb["groups"], retrieval=kb["retrieval"]),
        traces=dict(dir="traces", n=len(tr), files=tr,
                    note="5 senaryo için önceden çalıştırılmış soru-cevap iz kayıtları"),
        evaluation=dict(file=ev),
        files=files)
    (out / "submission.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def make_zip(out: Path, commit: str) -> Path:
    DIST.mkdir(exist_ok=True)
    z = DIST / f"submission_{commit[:7]}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                f.write(p, Path("submission") / p.relative_to(out))
    return z


def main():
    ap = argparse.ArgumentParser(description="Teslim paketi: submission/ + submission.json + zip")
    ap.add_argument("--allow-dirty", action="store_true", help="commit edilmemiş değişiklikle (deneme)")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    m = build(OUT, a.allow_dirty)
    total = sum(f["bytes"] for f in m["files"])
    print(f"{OUT}/submission.json: {len(m['files'])} dosya, {total / 1e6:.1f} MB, commit {m['code']['commit'][:7]}"
          + (" (commit edilmemiş değişiklik var)" if m["code"]["uncommitted_changes"] else ""))
    print(f"  adaptör MLX {m['adapter']['primary']['sha256'][:16]}…  PEFT {m['adapter']['original']['sha256'][:16]}…")
    print(f"  iz kaydı {m['traces']['n']}")
    if not a.no_zip:
        z = make_zip(OUT, m["code"]["commit"])
        print(f"  zip: {z} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
