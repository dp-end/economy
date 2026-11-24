"""Konuşma başına tek örnek için kayıp maskesi (yerel model 4. adım, kullanıcı kararı a).

MLX'in hazır eğiticisi yalnızca son asistan mesajından öğrenir; bu yüzden 3. adımda konuşmalar turlara bölünmüştü ve
aynı ~3.400 tokenlik istem her turda yeniden işleniyordu (4,42 M token). Burada konuşma bir kez işlenir ve kayıp
yalnızca asistan turlarının tokenlerinde hesaplanır (araç çağrıları ve son cevap; sistem, kullanıcı ve araç çıktıları
maskelenir): 2,79 M token. MLX'e bağlı değildir; 'render' tokenizer'ın sohbet şablonudur.
"""


def assistant_spans(messages: list, render) -> tuple:
    """(token kimlikleri, [(başlangıç, bitiş), ...]) — her asistan mesajının şablondaki token aralığı.

    render(mesajlar, add_generation_prompt) -> token listesi. Aralık, mesajdan önceki istem + '<|im_start|>assistant\\n'
    sonrasından mesajın '<|im_end|>' sonuna kadardır (model durmayı da öğrenir). Şablonun önek tutarlı olması şarttır
    (ilk k mesajın tokenleri tam konuşmanın ilk tokenleri); değilse hata verilir, sessizce yanlış maske üretilmez.
    """
    full = list(render(messages, False))
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        head = list(render(messages[:i], True))
        upto = list(render(messages[:i + 1], False))
        if full[:len(head)] != head or full[:len(upto)] != upto:
            raise ValueError(f"sohbet şablonu önek tutarlı değil (mesaj {i})")
        spans.append((len(head), len(upto)))
    return full, spans


def target_mask(n_tokens: int, spans: list) -> list:
    """Hedef (bir sonraki token) maskesi, uzunluk n_tokens - 1: hedef k+1. token bir asistan aralığındaysa 1."""
    m = [0] * (n_tokens - 1)
    for a, b in spans:
        for k in range(max(a, 1), b):
            m[k - 1] = 1
    return m
