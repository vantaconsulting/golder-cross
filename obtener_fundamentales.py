"""
obtener_fundamentales.py — extiende la tabla `tickers`
Descarga market cap e industria (SIC description) de cada ticker desde el
endpoint de referencia de Polygon (/v3/reference/tickers/{ticker}) y los
guarda en la base de datos. Se corre UNA sola vez (o cada varios meses,
market cap cambia pero no tan rápido como para justificar correrlo a diario).

Nota: esto NO consume tu límite de datos EOD — es un endpoint distinto,
de referencia/fundamentales.

Corre: python obtener_fundamentales.py
       python obtener_fundamentales.py AAPL MSFT NVDA   (subset rápido)
"""

import time
import requests
from config import get_connection, POLYGON_API_KEY, POLYGON_BASE_URL


def agregar_columnas_si_faltan(conn):
    cur = conn.cursor()
    columnas_existentes = {row[1] for row in cur.execute("PRAGMA table_info(tickers)").fetchall()}
    if "market_cap" not in columnas_existentes:
        cur.execute("ALTER TABLE tickers ADD COLUMN market_cap REAL")
    if "industria" not in columnas_existentes:
        cur.execute("ALTER TABLE tickers ADD COLUMN industria TEXT")
    conn.commit()


def obtener_detalle_ticker(ticker):
    url = f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}"
    r = requests.get(url, params={"apiKey": POLYGON_API_KEY})
    if r.status_code != 200:
        return None, None
    data = r.json().get("results", {})
    market_cap = data.get("market_cap")
    industria = data.get("sic_description")
    return market_cap, industria


def correr(tickers=None):
    conn = get_connection()
    agregar_columnas_si_faltan(conn)

    if tickers is None:
        tickers = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM tickers WHERE activo = 1 AND tipo = 'stock'"
        ).fetchall()]

    print(f"Descargando market cap + industria para {len(tickers)} tickers...")
    actualizados = 0
    sin_dato = 0

    for i, ticker in enumerate(tickers):
        market_cap, industria = obtener_detalle_ticker(ticker)
        if market_cap is not None or industria is not None:
            conn.execute(
                "UPDATE tickers SET market_cap = ?, industria = ? WHERE ticker = ?",
                (market_cap, industria, ticker)
            )
            actualizados += 1
        else:
            sin_dato += 1

        if i % 100 == 0:
            conn.commit()
            print(f"  [{i}/{len(tickers)}] {ticker} -> market_cap={market_cap}, industria={industria}")

        time.sleep(0.05)  # ser gentil con el rate limit, aunque el plan sea ilimitado

    conn.commit()
    conn.close()
    print(f"\n✅ Actualizados: {actualizados}. Sin dato disponible: {sin_dato}.")


if __name__ == "__main__":
    import sys
    correr(tickers=sys.argv[1:] if len(sys.argv) > 1 else None)
