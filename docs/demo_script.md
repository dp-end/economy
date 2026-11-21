# Canlı demo senaryosu (5 dakika, API anahtarı gerekmez)

Sunumun son slaytı ("Live demo") bu sırayı izler. Bütün komutlar depo kökünden, venv açıkken çalışır.
Sahte model (`--mock`) yalnızca `reports/decision_log_demo.sqlite` dosyasına yazar; gerçek karar günlüğü değişmez.

## Hazırlık (sunumdan önce, 2 dk)

```bash
cd ~/Okul/Ekonomi/BIST-Proje
source venv/bin/activate
pytest -q                     # 161 passed beklenir (~1 dk); sunumdan önce bir kez
clear
```

- Terminal yazı boyutunu büyüt (Cmd +), pencereyi projeksiyona tam ekran yap.
- Yedek olarak `reports/research/final_report.pdf` ve Rapor 6'yı (`research_reports.pdf`) açık tut.

## 1. Tam harness: bir aday (≈ 90 sn)

```bash
python -m src.harness.state_machine ASELS 2026-11-06 --mock --review MODIFY --final WATCH --reason "demo: backtest al-tutu gecmiyor"
```

Gösterilecekler:
- `Durumlar:` satırı: 17 durum sırayla; LLM sırayı seçemiyor.
- `Özet:` satırı: araç çağrısı sayısı, kontrol edilen sayılar, desteksiz sayı = 0.
- Ajan notları (market, fundamental_macro, …, orchestrator): sahte model yalnızca kanıt numaralarına atıf yapıyor.
- Son satır: `analysis_id` ve "(sahte model: kayıt … demo …)" — gerçek günlük değişmedi.

Söylenecek: "Etiketi kurallar seçti: FUNDAMENTAL_IMPROVEMENT_CANDIDATE. Ben insan olarak WATCH'a çevirdim, gerekçesiyle; sistem buna izin veriyor ve gerekçeyi saklıyor."

## 2. Güvenli başarısızlık: bilinmeyen hisse (≈ 45 sn)

```bash
python -m src.harness.state_machine YOKBOYLE 2026-11-06 --mock --review ACCEPT --reason "boyle bir kod yok"
```

Gösterilecekler: analiz durumları atlanıyor ama `CHECK_DATA_QUALITY → RISK_GATE → GENERATE_EXPLANATION → HUMAN_REVIEW →
LOG_AND_CLOSE` yine çalışıyor; kapı `HARD_BLOCK`, eylem `ANALYSIS_UNSAFE`.

## 3. MCP araçları (≈ 60 sn)

```bash
python -m src.mcp_server.client_demo
```

Gösterilecekler: sunucu ayrı süreç (stdio) olarak başlıyor, 15 araç listeleniyor, her biri bir kez çağrılıyor; sondaki
hatalı çağrılar reddediliyor: bilinmeyen hisse, bilinmeyen strateji (`AL_SAT`), gelecekteki tarih (`2030-01-01`).

## 4. Karar günlüğü, replay ve hafıza (≈ 60 sn)

```bash
python -m src.harness.decision_log
```

Gösterilecekler: 4 demo analizi yazılıyor, her biri tekrar oynatılıyor (`aynı=True`, `değişen veri=[]`), en altta
epizodik hafıza tablosu (insan kararları ve gerekçeleri).

Söylenecek: "Gerçek günlükteki iki Gemini kaydını bugünkü kodla oynattığımda eylem aynı, veri aynı ama prosedür
değişmiş çıkıyor — replay 'veri mi değişti, yöntem mi?' sorusunu ayırıyor (Rapor 6, Tablo 13)."

## 5. Geleceğe bakma testleri (≈ 30 sn)

```bash
pytest -q tests/test_harness.py tests/test_state_machine.py -k "speech or memory or direction"
```

Beklenen: `5 passed`. Söylenecek: "Bu testler bulduğumuz hataların testleri: kapanıştan sonraki konuşma, geleceği gören
hafıza, ters hesaplanan isabet oranı. Düzeltme olmadan başarısız oluyorlar."

## Yedek plan

| Sorun | Yapılacak |
|---|---|
| Terminal / venv açılmıyor | `final_report.pdf` §7 (Tablo 4–5) ve Rapor 6 §2–4'ü göster |
| Bir komut hata veriyor | atla, sonraki adıma geç; ekran görüntüsü yerine Rapor 6 tablolarını göster |
| "Gerçek Gemini çıktısı?" sorusu | `reports/experiments/raw.jsonl` (60 satır) ve Rapor 6 Tablo 8–10 |
| Zaman azaldı | yalnızca adım 1 ve 2 (≈ 2 dk) |

Demodan sonra temizlik (isteğe bağlı): `git checkout reports/decision_log_demo.sqlite`.
