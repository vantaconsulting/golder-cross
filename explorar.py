"""
explorar.py — Fase 1
Prueba suelta: confirma que Polygon (acciones + crypto) y Telegram
responden bien ANTES de construir nada más automatizado.

Corre: python explorar.py
"""

import requests
from config import POLYGON_API_KEY, POLYGON_BASE_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, check_env


def probar_polygon_ticker(ticker="AAPL"):
    print(f"\n--- Probando Polygon: precio previo de {ticker} ---")
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
    r = requests.get(url, params={"apiKey": POLYGON_API_KEY})
    print("Status:", r.status_code)
    print(r.json())


def probar_polygon_grouped_daily(fecha="2024-06-10"):
    """Trae TODO el mercado en un solo request (grouped daily bars)."""
    print(f"\n--- Probando Polygon: grouped daily bars ({fecha}) ---")
    url = f"{POLYGON_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{fecha}"
    r = requests.get(url, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"})
    print("Status:", r.status_code)
    data = r.json()
    resultados = data.get("results", [])
    print(f"Tickers regresados: {len(resultados)}")
    if resultados:
        print("Ejemplo:", resultados[0])


def probar_polygon_crypto(ticker="X:BTCUSD"):
    print(f"\n--- Probando Polygon: crypto {ticker} ---")
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
    r = requests.get(url, params={"apiKey": POLYGON_API_KEY})
    print("Status:", r.status_code)
    print(r.json())


def probar_telegram():
    print("\n--- Probando Telegram ---")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "✅ Conexión de prueba OK — sistema Golden Cross"
    })
    print("Status:", r.status_code)
    print(r.json())


if __name__ == "__main__":
    if not check_env():
        print("Llena tu archivo .env antes de continuar (copia .env.example).")
    else:
        probar_polygon_ticker("AAPL")
        probar_polygon_grouped_daily()
        probar_polygon_crypto("X:BTCUSD")
        probar_telegram()
