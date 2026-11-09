import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1) Kütüphaneler
import pandas, numpy, scipy, statsmodels, yfinance, pyarrow, matplotlib, mcp
print("Kütüphaneler: OK")

# 2) Anahtarlar tanımlı mı (değerleri yazdırmıyoruz)
for k in ["GEMINI_API_KEY", "EVDS_API_KEY"]:
    print(k, "tanımlı" if os.getenv(k) else "EKSİK")

# 3) Gemini çağrısı
client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
resp = client.chat.completions.create(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    messages=[{"role": "user", "content": "Tek cümleyle Türkçe merhaba de."}],
)
print("Gemini:", resp.choices[0].message.content)
