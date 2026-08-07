"""
analizar_por_grupo.py

Corre la estrategia ganadora de hoy (golden cross confirmado + objetivo
adaptativo) sobre TODO el universo, pero en vez de calcular confiabilidad
por TICKER individual (que casi siempre tiene muy pocos cruces propios,
2-5, para confiar estadísticamente), agrupa los resultados por:
  - Categoría de market cap (micro / small / mid / large)
  - Industria (SIC description de Polygon)

Esto responde tu pregunta: "¿el edge es más fuerte en ciertos tamaños de
empresa o sectores?" — con muestras mucho más grandes por grupo que las
que teníamos por ticker individual, y quizás encontrando un patrón más
generalizable ("las small-caps de tecnología funcionan mejor") en vez de
depender del historial propio, casi siempre corto, de una sola acción.

Requiere haber corrido antes obtener_fundamentales.py (para tener
market_cap e industria en la tabla tickers).

También calcula, por cada operación: MOMENTUM (retorno de los 20 días
antes del cruce) y VOLATILIDAD (desviación estándar de retornos diarios,
proxy de "qué tan penny-stock se comporta" en ausencia de ATR real, ya
que no tenemos high/low históricos guardados todavía).

Corre: python analizar_por_grupo.py
       python analizar_por_grupo.py --capital 1000
"""

import numpy as np
import pandas as pd
from config import get_connection

MIN_HISTORIA = 250
UMBRAL_SALTO_SOSPECHOSO = 0.50
CAPTURA_RATIO_ADAPTATIVO = 0.6
OBJETIVO_FIJO_DEFAULT_PCT = 15.0
MIN_CRUCES_PREVIOS_ADAPTATIVO = 2
CAPITAL_POR_TRADE_DEFAULT = 1000
DIAS_MOMENTUM = 20

BUCKETS_MARKET_CAP = [
    (0, 300_000_000, "1. Micro-cap (<$300M)"),
    (300_000_000, 2_000_000_000, "2. Small-cap ($300M-$2B)"),
    (2_000_000_000, 10_000_000_000, "3. Mid-cap ($2B-$10B)"),
    (10_000_000_000, float("inf"), "4. Large-cap (>$10B)"),
]


def bucket_market_cap(market_cap):
    if market_cap is None or pd.isna(market_cap):
        return "0. Sin dato"
    for lo, hi, nombre in BUCKETS_MARKET_CAP:
        if lo <= market_cap < hi:
            return nombre
    return "0. Sin dato"


def encontrar_cruces(df, col_rapida, col_lenta):
    sub = df.dropna(subset=[col_rapida, col_lenta]).reset_index()
    diff = sub[col_rapida] - sub[col_lenta]
    signo = np.sign(diff)
    cambio = signo.diff()
    cruces = []
    for i in range(1, len(sub)):
        if cambio.iloc[i] == 2:
            cruces.append((sub["index"].iloc[i], "dorado"))
        elif cambio.iloc[i] == -2:
            cruces.append((sub["index"].iloc[i], "muerte"))
    return cruces


def hay_salto_en_tramo(df, idx_inicio, idx_fin):
    return bool(df["salto_sospechoso"].iloc[idx_inicio:idx_fin + 1].any())


def calcular_mfe_pct(df, idx_entrada, idx_fin):
    precio_entrada = df["cierre"].iloc[idx_entrada]
    precio_max = df["cierre"].iloc[idx_entrada:idx_fin + 1].max()
    return (precio_max - precio_entrada) / precio_entrada * 100


def procesar_ticker(conn, ticker, market_cap, industria, capital_por_trade):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA:
        return []

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["retorno_diario"] = df["cierre"].pct_change()
    df["cambio_pct"] = df["retorno_diario"].abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO

    cruces = encontrar_cruces(df, "ema50", "ema200")
    historial_mfe = []
    resultados = []

    for j, (idx_c, tipo_c) in enumerate(cruces):
        if tipo_c != "dorado":
            continue
        idx_limite = cruces[j + 1][0] if j + 1 < len(cruces) else len(df) - 1
        if hay_salto_en_tramo(df, idx_c, idx_limite):
            continue

        if len(historial_mfe) >= MIN_CRUCES_PREVIOS_ADAPTATIVO:
            objetivo_pct = max(float(np.median(historial_mfe)) * CAPTURA_RATIO_ADAPTATIVO, 2.0)
        else:
            objetivo_pct = OBJETIVO_FIJO_DEFAULT_PCT

        mfe = calcular_mfe_pct(df, idx_c, idx_limite)
        historial_mfe.append(mfe)

        precio_entrada = df["cierre"].iloc[idx_c]
        if precio_entrada == 0:
            continue

        idx_salida = idx_limite
        for idx in range(idx_c, idx_limite + 1):
            ganancia = (df["cierre"].iloc[idx] - precio_entrada) / precio_entrada * 100
            if ganancia >= objetivo_pct:
                idx_salida = idx
                break

        precio_salida = df["cierre"].iloc[idx_salida]
        retorno_pct = (precio_salida - precio_entrada) / precio_entrada * 100

        # momentum: retorno acumulado de los DIAS_MOMENTUM días previos al cruce
        idx_inicio_mom = max(0, idx_c - DIAS_MOMENTUM)
        precio_inicio_mom = df["cierre"].iloc[idx_inicio_mom]
        momentum_pct = (precio_entrada - precio_inicio_mom) / precio_inicio_mom * 100 if precio_inicio_mom > 0 else None

        # volatilidad: std de retornos diarios en los DIAS_MOMENTUM dias previos (proxy de ATR)
        vol_ventana = df["retorno_diario"].iloc[idx_inicio_mom:idx_c]
        volatilidad_pct = float(vol_ventana.std() * 100) if len(vol_ventana) > 5 else None

        resultados.append({
            "ticker": ticker,
            "market_cap_bucket": bucket_market_cap(market_cap),
            "industria": industria if industria else "Sin dato",
            "momentum_pct": momentum_pct,
            "volatilidad_pct": volatilidad_pct,
            "retorno_pct": retorno_pct,
            "ganancia_usd": capital_por_trade * (retorno_pct / 100),
        })

    return resultados


def correr(capital=CAPITAL_POR_TRADE_DEFAULT):
    conn = get_connection()

    columnas = {row[1] for row in conn.execute("PRAGMA table_info(tickers)").fetchall()}
    if "market_cap" not in columnas:
        print("⚠️  No hay columna market_cap en la tabla tickers. Corre obtener_fundamentales.py primero.")
        return

    tickers_info = conn.execute(
        "SELECT ticker, market_cap, industria FROM tickers WHERE activo = 1 AND tipo = 'stock'"
    ).fetchall()

    print(f"Corriendo estrategia adaptativa agrupada sobre {len(tickers_info)} tickers...")
    todos = []
    for i, row in enumerate(tickers_info):
        res = procesar_ticker(conn, row["ticker"], row["market_cap"], row["industria"], capital)
        todos.extend(res)
        if i % 200 == 0:
            print(f"  [{i}/{len(tickers_info)}] {row['ticker']}: {len(res)} operaciones")
    conn.close()

    if not todos:
        print("⚠️  No hay resultados.")
        return

    df = pd.DataFrame(todos)

    print("\n" + "=" * 100)
    print("RESULTADOS POR CATEGORÍA DE MARKET CAP")
    print("=" * 100)
    resumen_cap = df.groupby("market_cap_bucket").agg(
        operaciones=("retorno_pct", "count"),
        pct_ganadoras=("retorno_pct", lambda s: (s > 0).mean() * 100),
        retorno_promedio_pct=("retorno_pct", "mean"),
        retorno_mediana_pct=("retorno_pct", "median"),
        ganancia_usd_total=("ganancia_usd", "sum"),
    ).round(2)
    resumen_cap["capital_invertido"] = (resumen_cap["operaciones"] * capital).round(2)
    resumen_cap["roi_pct"] = (resumen_cap["ganancia_usd_total"] / resumen_cap["capital_invertido"] * 100).round(2)
    print(resumen_cap.sort_index().to_string())

    print("\n" + "=" * 100)
    print("RESULTADOS POR INDUSTRIA — TODAS, ordenadas de PEOR a MEJOR retorno mediana")
    print("(antes solo veíamos el top 15 por volumen, lo cual podía esconder industrias malas con pocas operaciones)")
    print("=" * 100)
    resumen_ind_completo = df.groupby("industria").agg(
        operaciones=("retorno_pct", "count"),
        pct_ganadoras=("retorno_pct", lambda s: (s > 0).mean() * 100),
        retorno_promedio_pct=("retorno_pct", "mean"),
        retorno_mediana_pct=("retorno_pct", "median"),
        ganancia_usd_total=("ganancia_usd", "sum"),
    ).round(2)
    resumen_ind_completo["roi_pct"] = (resumen_ind_completo["ganancia_usd_total"] / (resumen_ind_completo["operaciones"] * capital) * 100).round(2)
    # solo industrias con muestra minima razonable para no confundir ruido estadistico con señal real
    MIN_OPERACIONES_CONFIABLE = 30
    con_muestra_suficiente = resumen_ind_completo[resumen_ind_completo["operaciones"] >= MIN_OPERACIONES_CONFIABLE]
    print(f"(mostrando solo industrias con >= {MIN_OPERACIONES_CONFIABLE} operaciones, para no confundir ruido con señal)")
    print(con_muestra_suficiente.sort_values("retorno_mediana_pct").to_string())

    print("\n" + "=" * 100)
    print("RESULTADOS POR INDUSTRIA (top 15 por número de operaciones, vista original)")
    print("=" * 100)
    print(resumen_ind_completo.sort_values("operaciones", ascending=False).head(15).to_string())

    print("\n" + "=" * 100)
    print("MOMENTUM Y VOLATILIDAD: ¿se relacionan con el resultado?")
    print("=" * 100)
    df_mom = df.dropna(subset=["momentum_pct"])
    df_mom["momentum_bucket"] = pd.cut(df_mom["momentum_pct"], bins=[-100, 0, 10, 25, 1000],
                                        labels=["Negativo", "0-10%", "10-25%", ">25%"])
    print(df_mom.groupby("momentum_bucket", observed=True).agg(
        operaciones=("retorno_pct", "count"),
        pct_ganadoras=("retorno_pct", lambda s: (s > 0).mean() * 100),
        retorno_promedio_pct=("retorno_pct", "mean"),
    ).round(2).to_string())

    df_vol = df.dropna(subset=["volatilidad_pct"])
    df_vol["vol_bucket"] = pd.cut(df_vol["volatilidad_pct"], bins=[0, 2, 4, 6, 100],
                                   labels=["Baja (<2%)", "Media (2-4%)", "Alta (4-6%)", "Muy alta (>6%)"])
    print("\n" + df_vol.groupby("vol_bucket", observed=True).agg(
        operaciones=("retorno_pct", "count"),
        pct_ganadoras=("retorno_pct", lambda s: (s > 0).mean() * 100),
        retorno_promedio_pct=("retorno_pct", "mean"),
    ).round(2).to_string())

    print("\nLectura rápida:")
    print("- Si 'Muy alta' volatilidad tiene retorno% más alto PERO 'pct_ganadoras' más bajo, confirma")
    print("  tu intuición: penny stocks suben más cuando aciertan, pero aciertan menos seguido.")
    print("- Busca en la tabla de market cap si el edge se concentra en un tamaño específico —")
    print("  si es así, podríamos filtrar el universo entero a ese rango en vez de operar todo.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=CAPITAL_POR_TRADE_DEFAULT)
    args = parser.parse_args()
    correr(capital=args.capital)