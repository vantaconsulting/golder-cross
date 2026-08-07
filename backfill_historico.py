"""
backfill_historico.py — Fase 4 (crítico)
Descarga 1-2 años de precios diarios para TODO el universo (acciones vía
grouped daily bars + crypto ticker por ticker) y llena `precios_diarios`.

Se corre 1 sola vez (o cuando quieras extender el historial).
Sin este paso, EMA200 no se puede calcular de forma confiable.

Corre: python backfill_historico.py --anios 2
"""

import argparse
import time
from datetime import date, timedelta
import requests
from config import get_connection, POLYGON_API_KEY, POLYGON_BASE_URL, CRYPTO_TICKERS


def rango_dias_habiles(anios):
    fin = date.today()
    inicio = fin - timedelta(days=int(anios * 365.25))
    dias = []
    d = inicio
    while d <= fin:
        if d.weekday() < 5:  # lunes-viernes (aproximado, Polygon regresa vacío si no hubo mercado)
            dias.append(d.isoformat())
        d += timedelta(days=1)
    return dias


def backfill_acciones(anios):
    conn = get_connection()
    cur = conn.cursor()
    dias = rango_dias_habiles(anios)
    print(f"Descargando grouped daily bars para {len(dias)} días hábiles...")

    tickers_activos = {
        row["ticker"] for row in cur.execute(
            "SELECT ticker FROM tickers WHERE tipo='stock' AND activo=1"
        ).fetchall()
    }

    insertados = 0
    for i, fecha in enumerate(dias):
        url = f"{POLYGON_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{fecha}"
        r = requests.get(url, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"})
        if r.status_code != 200:
            time.sleep(0.3)
            continue

        data = r.json().get("results", [])
        filas = [
            (item["T"], fecha, item["c"])
            for item in data if item["T"] in tickers_activos
        ]
        cur.executemany("""
            INSERT OR IGNORE INTO precios_diarios (ticker, fecha, cierre)
            VALUES (?, ?, ?)
        """, filas)
        insertados += len(filas)

        if i % 20 == 0:
            conn.commit()
            print(f"  [{i}/{len(dias)}] {fecha} -> {len(filas)} tickers guardados (acum: {insertados})")

        time.sleep(0.2)  # respeta rate limit del plan Starter

    conn.commit()
    conn.close()
    print(f"✅ Backfill de acciones terminado. Filas insertadas: {insertados}")


def backfill_crypto(anios):
    conn = get_connection()
    cur = conn.cursor()
    fin = date.today()
    inicio = fin - timedelta(days=int(anios * 365.25))

    for ticker in CRYPTO_TICKERS:
        print(f"Descargando histórico de {ticker}...")
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{inicio.isoformat()}/{fin.isoformat()}"
        r = requests.get(url, params={"apiKey": POLYGON_API_KEY, "adjusted": "true", "limit": 50000})
        if r.status_code != 200:
            print(f"  ⚠️  Error en {ticker}: {r.status_code}")
            continue

        resultados = r.json().get("results", [])
        filas = []
        for item in resultados:
            fecha = date.fromtimestamp(item["t"] / 1000).isoformat()
            filas.append((ticker, fecha, item["c"]))

        cur.executemany("""
            INSERT OR IGNORE INTO precios_diarios (ticker, fecha, cierre)
            VALUES (?, ?, ?)
        """, filas)
        conn.commit()
        print(f"  ✅ {ticker}: {len(filas)} días guardados")
        time.sleep(0.3)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anios", type=float, default=2.0, help="Años de historia a descargar")
    parser.add_argument("--solo", choices=["acciones", "crypto", "todo"], default="acciones",
                         help="Qué descargar (default: acciones)")
    args = parser.parse_args()

    if args.solo in ("acciones", "todo"):
        backfill_acciones(args.anios)
    if args.solo in ("crypto", "todo"):
        backfill_crypto(args.anios)
