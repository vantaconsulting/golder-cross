"""
obtener_universo.py — Fase 3
Baja el universo de acciones comunes (sin ETFs) de NASDAQ + NYSE desde
Polygon y llena la tabla `tickers`. Opcionalmente agrega las 10 criptomonedas.

Corre: python obtener_universo.py                  (solo acciones)
       python obtener_universo.py --incluir-crypto  (acciones + las 10 cryptos)
"""

import time
import requests
from config import get_connection, POLYGON_API_KEY, POLYGON_BASE_URL, CRYPTO_TICKERS


def obtener_acciones():
    """Pagina /v3/reference/tickers filtrando type=CS (common stock) en NASDAQ y NYSE."""
    acciones = []
    url = f"{POLYGON_BASE_URL}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "type": "CS",           # common stock -> excluye ETFs
        "active": "true",
        "limit": 1000,
        "apiKey": POLYGON_API_KEY,
    }

    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        for item in data.get("results", []):
            exch = item.get("primary_exchange", "")
            # Polygon usa códigos MIC: XNAS (Nasdaq), XNYS (NYSE)
            if exch in ("XNAS", "XNYS"):
                acciones.append({
                    "ticker": item["ticker"],
                    "tipo": "stock",
                    "exchange": "NASDAQ" if exch == "XNAS" else "NYSE",
                })

        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": POLYGON_API_KEY}  # next_url ya trae los demás params
            time.sleep(0.25)  # ser gentil con el rate limit
        else:
            url = None

    return acciones


def guardar_tickers(acciones, incluir_crypto=False):
    conn = get_connection()
    cur = conn.cursor()

    for a in acciones:
        cur.execute("""
            INSERT INTO tickers (ticker, tipo, exchange, activo)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(ticker) DO UPDATE SET
                exchange = excluded.exchange,
                activo = 1
        """, (a["ticker"], a["tipo"], a["exchange"]))

    if incluir_crypto:
        for c in CRYPTO_TICKERS:
            cur.execute("""
                INSERT INTO tickers (ticker, tipo, exchange, activo)
                VALUES (?, 'crypto', 'crypto', 1)
                ON CONFLICT(ticker) DO UPDATE SET activo = 1
            """, (c,))

    conn.commit()
    total = cur.execute("SELECT COUNT(*) AS n FROM tickers").fetchone()["n"]
    print(f"✅ Universo guardado. Total de tickers en la tabla: {total}")
    if not incluir_crypto:
        print("   (crypto NO incluido — corre con --incluir-crypto cuando lo quieras agregar)")
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--incluir-crypto", action="store_true",
                         help="También agrega las 10 criptomonedas a la tabla tickers")
    args = parser.parse_args()

    print("Descargando universo NASDAQ + NYSE (solo acciones comunes)...")
    acciones = obtener_acciones()
    print(f"Acciones encontradas: {len(acciones)}")
    guardar_tickers(acciones, incluir_crypto=args.incluir_crypto)
