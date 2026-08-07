"""
backtest_objetivos_fijos.py

Simula la operación COMPLETA (no solo si tocó el umbral, sino la venta real):
compra en cada golden cross CONFIRMADO (EMA, sin filtro de score, sin
anticipar), vende automático en cuanto el precio sube 5% / 10% / 15% / 20% /
25% desde la compra, o en el death cross si nunca lo alcanza. Reporta, para
cada umbral: días de hold, ganancia realizada, y cartera final.

Corre: python backtest_objetivos_fijos.py
       python backtest_objetivos_fijos.py --capital 1000
       python backtest_objetivos_fijos.py AAPL MSFT NVDA
"""

import numpy as np
import pandas as pd
from config import get_connection

MIN_HISTORIA = 250
UMBRAL_SALTO_SOSPECHOSO = 0.50
OBJETIVOS_PCT = [5, 10, 15, 20, 25]
CAPITAL_POR_TRADE_DEFAULT = 1000


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


def simular_salida(df, idx_entrada, idx_limite, objetivo_pct):
    precio_entrada = df["cierre"].iloc[idx_entrada]
    for idx in range(idx_entrada, idx_limite + 1):
        ganancia = (df["cierre"].iloc[idx] - precio_entrada) / precio_entrada * 100
        if ganancia >= objetivo_pct:
            return idx, "objetivo_alcanzado"
    return idx_limite, "cruce_contrario_o_fin"


def procesar_ticker(conn, ticker):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA:
        return []

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO

    cruces = encontrar_cruces(df, "ema50", "ema200")
    resultados = []

    for j, (idx_c, tipo_c) in enumerate(cruces):
        if tipo_c != "dorado":
            continue
        idx_limite = cruces[j + 1][0] if j + 1 < len(cruces) else len(df) - 1
        if hay_salto_en_tramo(df, idx_c, idx_limite):
            continue

        precio_entrada = df["cierre"].iloc[idx_c]
        if precio_entrada == 0:
            continue

        for objetivo in OBJETIVOS_PCT:
            idx_salida, motivo = simular_salida(df, idx_c, idx_limite, objetivo)
            precio_salida = df["cierre"].iloc[idx_salida]
            retorno_pct = (precio_salida - precio_entrada) / precio_entrada * 100
            fecha_entrada = df["fecha"].iloc[idx_c]
            fecha_salida = df["fecha"].iloc[idx_salida]
            dias = max((fecha_salida - fecha_entrada).days, 0)

            resultados.append({
                "ticker": ticker,
                "objetivo_pct": objetivo,
                "fecha_entrada": str(fecha_entrada.date()),
                "dias_en_operacion": dias,
                "retorno_pct": retorno_pct,
                "motivo_salida": motivo,
            })

    return resultados


def correr(tickers=None, capital=CAPITAL_POR_TRADE_DEFAULT):
    conn = get_connection()
    if tickers is None:
        tickers = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM tickers WHERE activo = 1 AND tipo = 'stock'"
        ).fetchall()]

    print(f"Simulando cierre automático en {OBJETIVOS_PCT}% sobre {len(tickers)} tickers "
          f"(golden cross EMA confirmado, sin filtros)...")
    todos = []
    for i, ticker in enumerate(tickers):
        res = procesar_ticker(conn, ticker)
        todos.extend(res)
        if i % 200 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(res)} simulaciones")
    conn.close()

    if not todos:
        print("⚠️  No hay resultados.")
        return

    df = pd.DataFrame(todos)
    df["ganancia_usd"] = capital * (df["retorno_pct"] / 100)

    print("\n" + "=" * 110)
    print(f"RESULTADOS POR OBJETIVO FIJO DE TOMA DE GANANCIA (${capital:,.0f}/operación, sin componer)")
    print("=" * 110)

    resumen = df.groupby("objetivo_pct").agg(
        operaciones=("retorno_pct", "count"),
        pct_alcanzo_objetivo=("motivo_salida", lambda s: (s == "objetivo_alcanzado").mean() * 100),
        dias_promedio=("dias_en_operacion", "mean"),
        dias_mediana=("dias_en_operacion", "median"),
        retorno_promedio_pct=("retorno_pct", "mean"),
        retorno_mediana_pct=("retorno_pct", "median"),
        pct_ganadoras=("retorno_pct", lambda s: (s > 0).mean() * 100),
        ganancia_usd_total=("ganancia_usd", "sum"),
    ).round(2)
    resumen["capital_invertido"] = (resumen["operaciones"] * capital).round(2)
    resumen["cartera_final"] = (resumen["capital_invertido"] + resumen["ganancia_usd_total"]).round(2)
    resumen["roi_pct"] = (resumen["ganancia_usd_total"] / resumen["capital_invertido"] * 100).round(2)

    print(resumen.to_string())
    print("=" * 110)
    print("\nLectura rápida:")
    print("- 'pct_alcanzo_objetivo' = % de operaciones que SÍ tocaron el umbral antes del death cross.")
    print("  El resto salió antes (en el death cross) sin llegar a la meta, a veces con pérdida.")
    print("- 'cartera_final' = capital_invertido + ganancia_usd_total — cuánto tendrías si hubieras puesto")
    print(f"  ${capital:,.0f} en CADA una de esas señales (capital no simultáneo, sumado a través del tiempo).")
    print("- 'dias_mediana' te dice qué tan rápido rota tu capital con cada umbral — más bajo = más veces")
    print("  puedes reciclar el mismo capital en un año real.")
    print("- Compara 'roi_pct' entre umbrales para ver el punto óptimo entre ambicioso y realista.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--capital", type=float, default=CAPITAL_POR_TRADE_DEFAULT)
    args = parser.parse_args()
    correr(tickers=args.tickers if args.tickers else None, capital=args.capital)
