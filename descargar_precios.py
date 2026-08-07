"""
descargar_precios.py — Fase 5
Baja el precio de HOY (acciones vía grouped daily bars + crypto) y lo
agrega a precios_diarios. Este es el script que se automatiza a diario.

Corre: python descargar_precios.py
"""

from datetime import date
import requests
from config import get_connection, POLYGON_API_KEY, POLYGON_BASE_URL, CRYPTO_TICKERS


def descargar_acciones_hoy():
    conn = get_connection()
    cur = conn.cursor()
    hoy = date.today().isoformat()

    tickers_activos = {
        row["ticker"] for row in cur.execute(
            "SELECT ticker FROM tickers WHERE tipo='stock' AND activo=1"
        ).fetchall()
    }

    url = f"{POLYGON_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{hoy}"
    r = requests.get(url, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"})

    if r.status_code != 200:
        print(f"⚠️  Polygon respondió {r.status_code}: {r.text[:200]}")
        conn.close()
        return 0

    data = r.json().get("results", [])
    filas = [(item["T"], hoy, item["c"]) for item in data if item["T"] in tickers_activos]

    cur.executemany("""
        INSERT OR REPLACE INTO precios_diarios (ticker, fecha, cierre)
        VALUES (?, ?, ?)
    """, filas)
    conn.commit()
    conn.close()
    print(f"✅ Acciones: {len(filas)} precios guardados para {hoy}")
    return len(filas)


def descargar_crypto_hoy():
    conn = get_connection()
    cur = conn.cursor()
    hoy = date.today().isoformat()
    guardados = 0

    for ticker in CRYPTO_TICKERS:
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        r = requests.get(url, params={"apiKey": POLYGON_API_KEY})
        if r.status_code != 200:
            continue
        resultados = r.json().get("results", [])
        if not resultados:
            continue
        cierre = resultados[0]["c"]
        cur.execute("""
            INSERT OR REPLACE INTO precios_diarios (ticker, fecha, cierre)
            VALUES (?, ?, ?)
        """, (ticker, hoy, cierre))
        guardados += 1

    conn.commit()
    conn.close()
    print(f"✅ Crypto: {guardados} precios guardados para {hoy}")
    return guardados


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo", choices=["acciones", "crypto", "todo"], default="acciones",
                         help="Qué descargar (default: acciones)")
    args = parser.parse_args()

    if args.solo in ("acciones", "todo"):
        descargar_acciones_hoy()
    if args.solo in ("crypto", "todo"):
        descargar_crypto_hoy()
