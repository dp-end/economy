# İlk deneme: temel model (ince ayarsız) + soru modu, 24 Kasım 2026

- Model: Qwen3-4B-Instruct-2507, yerelde 4-bit (mlx_lm.convert -q, 4,5 bit/ağırlık), mlx_lm.server, internet yok.
- Sorular: hocanın 5 örnek sorusu (S1–S5), analiz tarihi 2026-09-10.
- Harness sürümü: commit 59ecb9b (bu denemeden sonra düzeltilen 3 sorun aşağıda). Resmî karşılaştırma değildir;
  temel model / ince ayarlı model karşılaştırması 5. adımda sabit soru setiyle yapılacak.

| Soru | Geçerli araç çağrısı | Gözlenen hata |
|---|---|---|
| S1 | 3/4 | `horizon=1`; catch_up'ta lookback 5 (doğrusu 20) ve uydurma eşik 0,05; cevapta kavram karışıklığı |
| S2 | 0/2 | desen `week:PTT-PCU` (doğrusu `week:DDD-UU`); harness ANALYSIS_UNSAFE |
| S3 | 1/3 | koşul adları uyduruldu (`high_volume`, `supertrend_down`); 36 dokunuş günü = olay sandı (doğrusu 15); "yükselen 0" (doğrusu 11) |
| S4 | 1/5 | `period="Q1"` dört kez; cevapta çeyrek adları yanlış, sektör ve XU100 farkları karıştı |
| S5 | 4/7 | 2 günlük tarih aralığı (doğrusu 2024–2026); uydurma gösterge adları; tavsiye reddi zayıf |

Toplam: 21 çağrının 12'si geçersiz (%57). Sayı denetiminde uydurma sayı 0, ama anlam hataları var: sayı denetimi tek
başına yetmez, 5. adımda doğru cevaplarla karşılaştırma gerekir.

Bu denemeden sonra düzeltilenler: (1) get_macro_context kısa aralıkta ticker ile çöküyordu (fx_relation en az 60 gün);
(2) başarılı araç çağrısı yoksa cevap metni artık gösterilmiyor (S2'de "0 örüntü" uydurulmuştu); (3) eylem satırından
sonra kopyalanan istem metni kesiliyor. Önceki deneme (sıkıştırılmamış model) Metal bellek yetmezliğiyle düşmüştü.
