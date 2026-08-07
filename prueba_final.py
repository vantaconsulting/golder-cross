"""
prueba_final.py

LA prueba consolidada antes de pasar a operar con dinero real. Combina TODO
lo validado en la sesión de hoy sobre un universo YA FILTRADO:
  - Excluye micro-cap (<$300M) — fue la categoría con retorno Y win rate
    negativos, sin compensación de ningún tipo.
  - Excluye industrias "basura" (por default: REITs — 37.5% de acierto,
    peor que azar, retorno mediana negativo).

Sobre ese universo filtrado corre DOS cosas:

  1. PROXY RÁPIDO: % de cruces confirmados que alcanzan 5%/10% de ganancia
     dentro de 30/60 días (igual lógica que analizar_velocidad_ganancia.py,
     pero ya filtrado y solo con los umbrales/ventanas que pediste).

  2. PRUEBA FINAL COMPLETA: la estrategia combinada validada hoy (entrada
     ANTICIPADA real día-por-día, filtrada por score de ticker >= 70%,
     saliendo con el objetivo ADAPTATIVO calibrado por ticker) — pero
     ahora solo sobre el universo filtrado, no sobre todo el mercado.

Pendientes que quedan FUERA de esta prueba (avisados, no resueltos):
  - Comparación contra buy-and-hold de SPY (dependencia del edge con bull rally)
  - Stop loss

Corre: python prueba_final.py
       python prueba_final.py --market-cap-minimo 300000000
"""

import numpy as np
import pandas as pd
from config import get_connection
import estrategia_combinada

MIN_HISTORIA = 250
UMBRAL_SALTO_SOSPECHOSO = 0.50
MARKET_CAP_MINIMO_DEFAULT = 300_000_000
INDUSTRIAS_EXCLUIDAS_DEFAULT = ["REAL ESTATE INVESTMENT TRUSTS"]
UMBRALES_PROXY = [5, 10]
VENTANAS_PROXY = [30, 60]
SCORE_MINIMO_DEFAULT = 70.0
CAPITAL_POR_TRADE_DEFAULT = 1000


def obtener_universo_filtrado(conn, market_cap_minimo, industrias_excluidas):
    columnas = {row[1] for row in conn.execute("PRAGMA table_info(tickers)").fetchall()}
    if "market_cap" not in columnas:
        print("⚠️  Falta correr obtener_fundamentales.py antes de filtrar por market cap/industria.")
        return None

    filas = conn.execute("""
        SELECT ticker, market_cap, industria FROM tickers
        WHERE activo = 1 AND tipo = 'stock'
    """).fetchall()

    incluidos, excluidos_cap, excluidos_industria = [], 0, 0
    for r in filas:
        if r["market_cap"] is not None and r["market_cap"] < market_cap_minimo:
            excluidos_cap += 1
            continue
        if r["industria"] in industrias_excluidas:
            excluidos_industria += 1
            continue
        incluidos.append(r["ticker"])

    print(f"Universo filtrado: {len(incluidos)} tickers "
          f"(excluidos por micro-cap: {excluidos_cap}, por industria basura: {excluidos_industria})")
    return incluidos


def encontrar_cruces_dorados(df):
    diff = df["ema50"] - df["ema200"]
    signo = np.sign(diff)
    cambio = signo.diff()
    return list(df.index[cambio == 2])


def hay_salto_en_tramo(df, idx_inicio, idx_fin):
    return bool(df["salto_sospechoso"].iloc[idx_inicio:idx_fin + 1].any())


def proxy_ticker(conn, ticker):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA:
        return []

    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO

    cruces = encontrar_cruces_dorados(df)
    max_ventana = max(VENTANAS_PROXY)
    eventos = []

    for idx_c in cruces:
        idx_fin_disponible = min(idx_c + max_ventana, len(df) - 1)
        dias_disponibles = idx_fin_disponible - idx_c
        if hay_salto_en_tramo(df, idx_c, idx_fin_disponible):
            continue
        precio_entrada = df["cierre"].iloc[idx_c]
        if precio_entrada == 0:
            continue
        tramo = df["cierre"].iloc[idx_c:idx_fin_disponible + 1]
        ganancia_acumulada = ((tramo.cummax() - precio_entrada) / precio_entrada * 100).reset_index(drop=True)
        eventos.append({"dias_disponibles": dias_disponibles, "ganancia_acumulada": ganancia_acumulada})

    return eventos


def correr_proxy(conn, tickers):
    print("\n" + "=" * 90)
    print(f"1. PROXY RÁPIDO — universo filtrado, umbrales {UMBRALES_PROXY}%, ventanas {VENTANAS_PROXY} días")
    print("=" * 90)

    todos_eventos = []
    for i, ticker in enumerate(tickers):
        todos_eventos.extend(proxy_ticker(conn, ticker))
        if i % 300 == 0:
            print(f"  [{i}/{len(tickers)}] procesados...")

    if not todos_eventos:
        print("⚠️  No hay cruces suficientes en el universo filtrado.")
        return

    print(f"\nTotal de cruces dorados confirmados (universo filtrado): {len(todos_eventos)}")
    header = f"{'Ventana':<12}" + "".join([f">={u}%".ljust(10) for u in UMBRALES_PROXY]) + "N elegibles"
    print(header)
    for ventana in VENTANAS_PROXY:
        elegibles = [e for e in todos_eventos if e["dias_disponibles"] >= ventana]
        n = len(elegibles)
        fila = f"{ventana} días{'':<6}"
        for u in UMBRALES_PROXY:
            alcanzaron = sum(1 for e in elegibles if e["ganancia_acumulada"].iloc[:ventana + 1].max() >= u) if n else 0
            fila += (f"{alcanzaron/n*100:.1f}%" if n else "N/A").ljust(10)
        fila += f"{n}"
        print(fila)


def correr(market_cap_minimo=MARKET_CAP_MINIMO_DEFAULT,
           industrias_excluidas=None,
           score_minimo=SCORE_MINIMO_DEFAULT,
           capital=CAPITAL_POR_TRADE_DEFAULT):
    if industrias_excluidas is None:
        industrias_excluidas = INDUSTRIAS_EXCLUIDAS_DEFAULT

    conn = get_connection()
    tickers = obtener_universo_filtrado(conn, market_cap_minimo, industrias_excluidas)
    if not tickers:
        conn.close()
        return

    correr_proxy(conn, tickers)
    conn.close()

    print("\n" + "=" * 90)
    print("2. PRUEBA FINAL COMPLETA — anticipado + score ticker >= "
          f"{score_minimo}% + objetivo adaptativo, universo filtrado")
    print("=" * 90)
    estrategia_combinada.correr(tickers=tickers, score_minimo=score_minimo, capital=capital)

    print("\n" + "=" * 90)
    print("PENDIENTES QUE QUEDARON FUERA DE ESTA PRUEBA (avisados, no resueltos):")
    print("  - Comparación contra buy-and-hold de SPY (dependencia del edge con bull rally)")
    print("  - Stop loss")
    print("=" * 90)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-cap-minimo", type=float, default=MARKET_CAP_MINIMO_DEFAULT)
    parser.add_argument("--score-minimo", type=float, default=SCORE_MINIMO_DEFAULT)
    parser.add_argument("--capital", type=float, default=CAPITAL_POR_TRADE_DEFAULT)
    args = parser.parse_args()

    correr(
        market_cap_minimo=args.market_cap_minimo,
        score_minimo=args.score_minimo,
        capital=args.capital,
    )
