"""Yerel model 4. adım: QLoRA eğitimi (MLX, Apple M4, internet yok).

Kullanıcı kararı (24 Kasım 2026, seçenek a): eğitim Mac'te; her konuşma tek örnek ve kayıp yalnızca asistan turlarında
(src/finetune/masks.py; araç çağrıları + son cevap), LoRA son 16 katmanda, 8.192 tokenden uzun konuşmalar dışarıda,
önce 1 epoch. Ölçülen hız (deneme 2): ~90 token/sn, yani 1 epoch ~2,4 M token için ~6-8 saat.

Taban: models/Qwen3-4B-Instruct-2507-4bit (yerelde 4-bit; LoRA 4-bit taban üzerinde = QLoRA). Çıkarımda da aynı
4-bit taban kullanılır. Ayarlar CFG'de ve adapter_config.json'a yazılır; eğitim günlüğü reports/local_llm/.

Çalıştırma (proje kökünde, venv-llm; Mac uyumasın diye caffeinate; oturum kapansa da sürsün diye nohup):
  HF_HUB_OFFLINE=1 nohup caffeinate -i venv-llm/bin/python -m src.finetune.train_mlx \\
      > reports/local_llm/train_v1.out 2>&1 &
  Kaldığı yerden: aynı komut + --resume.   Hızlı deneme: --smoke --out <geçici klasör>
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

from src.finetune.masks import assistant_spans, target_mask

DATA = Path("data/finetune")
MODEL = "models/Qwen3-4B-Instruct-2507-4bit"
OUT = Path("models/adapters/bist-qwen3-4b-v1")
LOG = Path("reports/local_llm/train_v1_log.jsonl")
# Öğrenme oranı: MLX LoRA çıktısı scale ile çarpılır (scale=20); 1e-4 ile 3 adımlık denemede genel dil bozuldu
# ("kapital kapalı ..." tekrarı). scale x lr = 20 x 2e-5 = 4e-4, PEFT'teki alışılmış alpha/r=2 x 2e-4 ile aynı büyüklük.
CFG = dict(num_layers=16, lora_parameters=dict(rank=16, scale=20.0, dropout=0.05), learning_rate=2e-5, min_lr=2e-6,
           warmup_steps=10, grad_accumulation=2, epochs=1, max_seq_length=8192, seed=0, report_every=5,
           eval_every=100, eval_conversations=15, save_every=25, max_micro_steps=0)


def encode(tok, tools, conv):
    def render(msgs, gen):
        return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=True, return_dict=False)
    ids, spans = assistant_spans(conv["messages"], render)
    return ids, target_mask(len(ids), spans), spans


def loss_fn(model, x, m):
    logits = model(x[:, :-1])
    ce = nn.losses.cross_entropy(logits, x[:, 1:]) * m
    n = m.sum()
    return ce.astype(mx.float32).sum() / n, n


def evaluate(model, items) -> float:
    model.eval()
    tot = n = 0.0
    for _, ids, m, _ in items:
        lv, nt = loss_fn(model, mx.array([ids]), mx.array([m], dtype=mx.float32))
        mx.eval(lv, nt)
        tot, n = tot + lv.item() * nt.item(), n + nt.item()
    model.train()
    return tot / n


def save(model, out: Path, cfg: dict, state: dict):
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(dict(
        fine_tune_type="lora", num_layers=cfg["num_layers"], lora_parameters=cfg["lora_parameters"], model=MODEL,
        training=cfg), ensure_ascii=False, indent=1))
    (out / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1))


def main(a):
    cfg = dict(CFG)
    for kv in a.set or []:                                     # ör. --set learning_rate=2e-5 --set eval_every=20
        k, v = kv.split("=", 1)
        cfg[k] = type(CFG[k])(float(v)) if isinstance(CFG[k], (int, float)) else v
    out, log = Path(a.out), Path(a.log)
    model, tok = load(MODEL)
    tools = json.loads((DATA / "tools.json").read_text())
    convs = [json.loads(l) for l in open(DATA / "conversations.jsonl")]
    items, dropped = [], []
    for c in convs:
        ids, m, spans = encode(tok, tools, c)
        if a.smoke and c["split"] == "train" and len(items) < 3:          # gerçek tokenizer ile maske kontrolü
            for (s, e), msg in zip(spans, [x for x in c["messages"] if x["role"] == "assistant"]):
                piece = tok.decode(ids[s:e])
                assert (msg.get("content") or "")[:40] in piece or "<tool_call>" in piece, piece[:120]
        (dropped if len(ids) > cfg["max_seq_length"] else items).append((c["split"], ids, m, c["id"]))
    train = [x for x in items if x[0] == "train"]
    valid = [x for x in items if x[0] == "valid"]
    if a.smoke:
        train, valid = sorted(train, key=lambda x: len(x[1]))[:3], sorted(valid, key=lambda x: len(x[1]))[:2]
        cfg.update(grad_accumulation=1, eval_every=3, eval_conversations=2, save_every=3, report_every=1)
    rng = random.Random(cfg["seed"])
    order = []
    for _ in range(cfg["epochs"]):
        ep = list(train)
        rng.shuffle(ep)
        order += ep
    if cfg["max_micro_steps"]:                                 # pilot: ilk N adım (aynı karıştırılmış sıra)
        order = order[:cfg["max_micro_steps"]]
    val_items = valid[:cfg["eval_conversations"]]

    model.freeze()
    linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
    start = 0
    if a.resume and (out / "state.json").exists():
        model.load_weights(str(out / "adapters.safetensors"), strict=False)
        start = json.loads((out / "state.json").read_text())["micro_step"]
    n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    grad_checkpoint(model.layers[0])
    model.train()
    opt_steps = math.ceil(len(order) / cfg["grad_accumulation"])
    w = min(cfg["warmup_steps"], max(opt_steps - 1, 1))
    sched = optim.join_schedules([optim.linear_schedule(cfg["learning_rate"] / 10, cfg["learning_rate"], w),
                                  optim.cosine_decay(cfg["learning_rate"], max(opt_steps - w, 1), cfg["min_lr"])], [w])
    opt = optim.Adam(learning_rate=sched)
    lvg = nn.value_and_grad(model, loss_fn)
    total_tok = sum(len(x[1]) for x in order)
    head = dict(kind="start", model=MODEL, train_conversations=len(train), valid_conversations=len(valid),
                dropped_over_max_seq=len(dropped), micro_steps=len(order), optimizer_steps=opt_steps,
                tokens_per_epoch=sum(len(x[1]) for x in train), trained_tokens_per_epoch=sum(sum(x[2]) for x in train),
                trainable_params=n_params, start_micro_step=start, cfg=cfg)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(head, ensure_ascii=False) + "\n")
    print(json.dumps(head, ensure_ascii=False), flush=True)

    def write(rec):
        with open(log, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    if start == 0:
        write(dict(kind="val", micro_step=0, val_loss=evaluate(model, val_items)))
    acc, win, t0, done_tok = None, [], time.time(), sum(len(x[1]) for x in order[:start])
    for k in range(start + 1, len(order) + 1):
        _, ids, m, cid = order[k - 1]
        (lv, nt), g = lvg(model, mx.array([ids]), mx.array([m], dtype=mx.float32))
        g = tree_map(lambda x: x / cfg["grad_accumulation"], g)
        acc = g if acc is None else tree_map(mx.add, acc, g)
        mx.eval(acc, lv, nt)
        if k % cfg["grad_accumulation"] == 0 or k == len(order):
            opt.update(model, acc)
            mx.eval(model.parameters(), opt.state)
            acc = None
        done_tok += len(ids)
        win.append(lv.item())
        if k % 10 == 0:
            mx.clear_cache()
        if k % cfg["report_every"] == 0 or k == len(order):
            el = time.time() - t0
            rate = (done_tok - sum(len(x[1]) for x in order[:start])) / el
            write(dict(kind="train", micro_step=k, optimizer_step=math.ceil(k / cfg["grad_accumulation"]),
                       loss=sum(win) / len(win), lr=float(sched(opt.step)), tok_per_s=round(rate, 1),
                       peak_gb=round(mx.get_peak_memory() / 1e9, 2), elapsed_min=round(el / 60, 1),
                       eta_min=round((total_tok - done_tok) / rate / 60, 1)))
            win = []
        if k % cfg["eval_every"] == 0 or k == len(order):
            write(dict(kind="val", micro_step=k, val_loss=evaluate(model, val_items)))
        if k % cfg["save_every"] == 0 or k == len(order):
            save(model, out, cfg, dict(micro_step=k, of=len(order), last_conversation=cid))
    write(dict(kind="done", micro_step=len(order), minutes=round((time.time() - t0) / 60, 1), adapter=str(out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QLoRA eğitimi (MLX), konuşma başına tek örnek")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="3 kısa konuşmayla uçtan uca deneme")
    ap.add_argument("--set", action="append", help="CFG değeri değiştir, ör. max_micro_steps=40")
    main(ap.parse_args())
