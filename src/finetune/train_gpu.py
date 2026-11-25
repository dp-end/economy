"""Yerel model 4b: LoRA eğitimi Google Colab'ın ücretsiz GPU'sunda (PyTorch + PEFT). Çıkarım yine Mac'te, internetsiz.

Kullanıcı kararları (25 Kasım 2026): Mac'teki eğitim (v1, 8 saat) bilgisayarı meşgul ettiği için eğitim bulutta;
RunPod en az 10 $ yükleme istediği için ücretsiz Colab (T4, 16 GB). Ayarlar v1 ile aynı: son 16 katman, 7 doğrusal
katman, rank 16, ölçek 20, dropout 0,05, öğrenme oranı 2e-5 -> 2e-6 kosinüs, 10 adım ısınma, gradyan biriktirme 2,
1 epoch, konuşma başına tek örnek, kayıp yalnızca asistan turlarında (masks.py). Farklar: taban 4-bit değil fp16
(T4 bf16 desteklemez; bf16 destekleyen GPU'da bf16) ve en uzun konuşma sınırı 8.192 yerine 12.288 (5.1'deki S4
kısaltmasından sonra en uzun konuşma 8.946 token, hiçbiri atılmaz).

İki aşama:
  prepare (Mac, venv-llm): konuşmalar Mac'teki tokenizer ile token kimliklerine çevrilir (çıkarımdaki tokenizer'ın
      aynısı; Colab'daki sohbet şablonu sürümüne bağlı kalınmaz) ve dist/colab_v2/ paketi yazılır: bu dosya +
      tokenized.jsonl.gz + colab_train_v2.ipynb.     venv-llm/bin/python -m src.finetune.train_gpu prepare
  train (Colab): yalnızca torch + transformers + peft ister, proje koduna bağlı değildir. Not defterindeki hücreler
      çalıştırır; adaptör her 25 adımda Google Drive'a yazılır, oturum koparsa --resume ile kaldığı yerden sürer.
Sonra adaptör Mac'e indirilir ve src/finetune/peft_to_mlx.py ile MLX biçimine çevrilir (4-bit taban üzerinde kullanılır).

Kayıp: MLX eğitimindeki (train_mlx.loss_fn) tanımın aynısı, konuşmadaki hedef tokenlerin ortalama çapraz entropisi.
Bellek için logit yalnızca hedef pozisyonlarda ve 1.024'lük parçalar hâlinde hesaplanır (parça başına ~0,6 GB;
12.288 x 151.936 tam logit ~7 GB olurdu, T4'te sığmaz). fp16'da gradyan ölçekleyici (GradScaler) kullanılır;
kayıp sonsuz/NaN çıkarsa eğitim açık bir mesajla durur (yedek plan: 4-bit QLoRA).
"""
import argparse
import gzip
import json
import math
import random
import shutil
import time
from pathlib import Path

BASE_REPO = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"   # Mac'teki models/Qwen3-4B-Instruct-2507 ile aynı sürüm
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]   # MLX'in varsayılanı: hepsi
CFG = dict(num_layers=16, rank=16, scale=20.0, dropout=0.05, learning_rate=2e-5, min_lr=2e-6, warmup_steps=10,
           grad_accumulation=2, epochs=1, max_seq_length=12288, seed=0, report_every=5, eval_every=100,
           eval_conversations=15, save_every=25, max_micro_steps=0)
CHUNK = 1024                                                    # kayıp parçası (hedef token)
PACK = Path("dist/colab_v2")
NOTEBOOK = Path(__file__).with_name("colab_train_v2.ipynb")


# ---------------------------------------------------------------- prepare (Mac)
def prepare(a):
    from transformers import AutoTokenizer
    from src.finetune.masks import assistant_spans
    data = Path(a.data)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tools = json.loads((data / "tools.json").read_text())

    def render(msgs, gen):
        return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=True, return_dict=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = tot = tgt = 0
    with gzip.open(out / "tokenized.jsonl.gz", "wt") as f:
        for line in open(data / "conversations.jsonl"):
            c = json.loads(line)
            ids, spans = assistant_spans(c["messages"], render)
            f.write(json.dumps(dict(id=c["id"], cat=c["cat"], split=c["split"], ids=ids, spans=spans)) + "\n")
            n, tot, tgt = n + 1, tot + len(ids), tgt + len(target_positions(len(ids), spans))
    shutil.copy(__file__, out / "train_gpu.py")
    shutil.copy(NOTEBOOK, out / NOTEBOOK.name)
    print(json.dumps(dict(conversations=n, tokens=tot, target_tokens=tgt, out=str(out)), ensure_ascii=False))


# ---------------------------------------------------------------- train (Colab)
def lr_schedule(cfg, opt_steps):
    """MLX'teki takvimin aynısı: lr/10'dan lr'ye doğrusal ısınma (w adım), sonra lr'den min_lr'ye kosinüs."""
    lr, lo = cfg["learning_rate"], cfg["min_lr"]
    w = min(cfg["warmup_steps"], max(opt_steps - 1, 1))
    decay = max(opt_steps - w, 1)

    def at(step):                                              # step: 0'dan başlayan optimizer adımı
        if step < w:
            return lr / 10 + (lr - lr / 10) * step / w
        s = min(step - w, decay)
        return lo + (lr - lo) * 0.5 * (1 + math.cos(math.pi * s / decay))
    return at


def target_positions(n, spans):
    """Hedefi asistan aralığında olan pozisyonlar: j konumundaki logit j+1. tokeni tahmin eder (masks.target_mask)."""
    return sorted({k - 1 for a, b in spans for k in range(max(a, 1), b)})


def train(a):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM

    cfg = dict(CFG)
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        cfg[k] = type(CFG[k])(float(v)) if isinstance(CFG[k], (int, float)) else v
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.dtype == "auto":
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16) \
            if dev == "cuda" else torch.float32
    else:
        dtype = getattr(torch, a.dtype)
    torch.manual_seed(cfg["seed"])
    out, log = Path(a.out), Path(a.log)

    convs = [json.loads(l) for l in gzip.open(a.data, "rt")]
    items, dropped = [], []
    for c in convs:
        (dropped if len(c["ids"]) > cfg["max_seq_length"] else items).append(c)
    train_c = [c for c in items if c["split"] == "train"]
    valid_c = [c for c in items if c["split"] == "valid"]
    if a.smoke:                                                # uçtan uca deneme: en uzun 2 (bellek) + en kısa 4
        by_len = sorted(train_c, key=lambda c: len(c["ids"]))
        train_c = by_len[-2:] + by_len[:4]
        valid_c = sorted(valid_c, key=lambda c: len(c["ids"]))[:2]
        cfg.update(eval_every=3, eval_conversations=2, save_every=3, report_every=1)
    rng = random.Random(cfg["seed"])
    order = []
    for _ in range(cfg["epochs"]):
        ep = list(train_c)
        rng.shuffle(ep)
        order += ep
    if cfg["max_micro_steps"]:
        order = order[:cfg["max_micro_steps"]]
    val_items = valid_c[:cfg["eval_conversations"]]

    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=dtype,
                                                 attn_implementation="sdpa", device_map={"": dev})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    model.enable_input_require_grads()
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers - cfg["num_layers"], n_layers))
    lcfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["scale"] * cfg["rank"], lora_dropout=cfg["dropout"],
                      target_modules=TARGETS, layers_to_transform=layers, bias="none", task_type="CAUSAL_LM")
    start = 0
    if a.resume and (out / "state.json").exists():            # oturum koptuysa: Drive'daki son kayıttan
        model = PeftModel.from_pretrained(model, str(out), is_trainable=True)
        start = json.loads((out / "state.json").read_text())["micro_step"]
    else:
        model = get_peft_model(model, lcfg)                    # adaptör ağırlıkları float32 (MLX v1 gibi)
    lm = model.get_base_model()                                # LoRA katmanları içine yerleşmiş Qwen3ForCausalLM
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    def chunk_ce(h, t):
        return F.cross_entropy(lm.lm_head(h).float(), t, reduction="sum")

    def loss_of(c):
        ids = torch.tensor([c["ids"]], device=dev)
        keep = torch.tensor(target_positions(len(c["ids"]), c["spans"]), device=dev)
        h = lm.model(input_ids=ids[:, :-1]).last_hidden_state[0, keep]
        tgt = ids[0, keep + 1]
        tot = sum(checkpoint(chunk_ce, h[i:i + CHUNK], tgt[i:i + CHUNK], use_reentrant=False)
                  for i in range(0, len(keep), CHUNK))
        return tot / len(keep), len(keep)

    def evaluate(cs):
        model.eval()
        tot = n = 0.0
        with torch.no_grad():
            for c in cs:
                lv, nt = loss_of(c)
                tot, n = tot + lv.item() * nt, n + nt
        model.train()
        return tot / n

    opt_steps = math.ceil(len(order) / cfg["grad_accumulation"])
    sched = lr_schedule(cfg, opt_steps)
    off = start // cfg["grad_accumulation"]
    opt = torch.optim.Adam(params, lr=sched(off), betas=(0.9, 0.999), eps=1e-8)   # MLX optim.Adam ile aynı
    scaler = torch.amp.GradScaler(dev, enabled=dtype == torch.float16)
    total_tok = sum(len(c["ids"]) for c in order)

    def write(rec):
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    def save(k, cid):
        model.save_pretrained(str(out))
        meta = dict(micro_step=k, of=len(order), last_conversation=cid, base=a.model, revision=a.revision,
                    dtype=str(dtype), cfg=cfg, lora_layers=layers, targets=TARGETS)
        (out / "state.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    write(dict(kind="start", model=a.model, revision=a.revision, device=dev, dtype=str(dtype),
               gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None, train_conversations=len(train_c),
               valid_conversations=len(valid_c), dropped_over_max_seq=len(dropped), micro_steps=len(order),
               optimizer_steps=opt_steps, tokens_per_epoch=sum(len(c["ids"]) for c in train_c),
               trainable_params=n_params, start_micro_step=start, cfg=cfg))
    if start >= len(order):
        write(dict(kind="done", micro_step=start, note="eğitim zaten bitmiş", adapter=str(out)))
        return
    if start == 0:
        write(dict(kind="val", micro_step=0, val_loss=evaluate(val_items)))
    model.train()
    win, skipped, t0 = [], 0, time.time()
    done_tok = base_tok = sum(len(c["ids"]) for c in order[:start])
    for k in range(start + 1, len(order) + 1):
        c = order[k - 1]
        lv, _ = loss_of(c)
        if not math.isfinite(lv.item()):
            write(dict(kind="error", micro_step=k, conversation=c["id"], tokens=len(c["ids"]), dtype=str(dtype),
                       message="kayıp sonsuz/NaN: fp16 taşması olabilir; eğitim durdu (son kayıt Drive'da)."))
            raise SystemExit(1)
        scaler.scale(lv / cfg["grad_accumulation"]).backward()
        if k % cfg["grad_accumulation"] == 0 or k == len(order):
            step = (k - 1) // cfg["grad_accumulation"]
            for g in opt.param_groups:
                g["lr"] = sched(step)
            before = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(opt)                                   # gradyan taşarsa bu adım atlanır, ölçek küçülür
            scaler.update()
            if before is not None and scaler.get_scale() < before:
                skipped += 1
            opt.zero_grad(set_to_none=True)
        done_tok += len(c["ids"])
        win.append(lv.item())
        if k % cfg["report_every"] == 0 or k == len(order):
            el = time.time() - t0
            rate = (done_tok - base_tok) / el
            write(dict(kind="train", micro_step=k, optimizer_step=math.ceil(k / cfg["grad_accumulation"]),
                       loss=sum(win) / len(win), lr=opt.param_groups[0]["lr"], tok_per_s=round(rate, 1),
                       peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None,
                       skipped_steps=skipped, elapsed_min=round(el / 60, 1),
                       eta_min=round((total_tok - done_tok) / rate / 60, 1)))
            win = []
        if k % cfg["eval_every"] == 0 or k == len(order):
            write(dict(kind="val", micro_step=k, val_loss=evaluate(val_items)))
        if k % cfg["save_every"] == 0 or k == len(order):
            save(k, c["id"])
    write(dict(kind="val_full", micro_step=len(order), conversations=len(valid_c), val_loss=evaluate(valid_c)))
    write(dict(kind="done", micro_step=len(order), minutes=round((time.time() - t0) / 60, 1), skipped_steps=skipped,
               adapter=str(out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LoRA eğitimi, bulut GPU (PyTorch + PEFT)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="Mac: konuşmaları tokenize et, Colab paketini yaz")
    p.add_argument("--data", default="data/finetune")
    p.add_argument("--tokenizer", default="models/Qwen3-4B-Instruct-2507-4bit")
    p.add_argument("--out", default=str(PACK))
    t = sub.add_parser("train", help="Colab: eğitim")
    t.add_argument("--data", default="tokenized.jsonl.gz")
    t.add_argument("--model", default=BASE_REPO)
    t.add_argument("--revision", default=BASE_REVISION)
    t.add_argument("--out", default="adapter_v2")
    t.add_argument("--log", default="train_v2_log.jsonl")
    t.add_argument("--device", default=None)
    t.add_argument("--dtype", default="auto", help="auto | float16 | bfloat16 | float32")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--smoke", action="store_true", help="6 konuşmayla (en uzun 2 dahil) uçtan uca deneme")
    t.add_argument("--set", action="append", help="CFG değeri değiştir, ör. max_micro_steps=4")
    args = ap.parse_args()
    {"prepare": prepare, "train": train}[args.cmd](args)
