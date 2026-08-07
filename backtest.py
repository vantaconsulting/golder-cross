"""
backtest.py — Fase 11 (v2)

Cambios respecto a v1:
  - SOLO LONG. Los shorts se quitaron por completo (perdían dinero en las
    4 combinaciones del baseline y con riesgo de cola enorme sin stop loss).
  - RETORNO ANUALIZADO por operación, además del retorno crudo. El retorno
    crudo NO es comparable entre operaciones de distinta duración (una
    operación +13% en 3 semanas no es lo mismo que +13% en 10 meses).
  - 3 TÉCNICAS DE SALIDA comparadas lado a lado:
      1. cruce_contrario   -> la original: espera al death cross, sin importar
                              cuánto tarde ni cuánto se le devuelva a la ganancia.
      2. objetivo_fijo     -> toma ganancia en cuanto el precio sube
                              OBJETIVO_FIJO_DEFAULT_PCT% desde la entrada (o en
                              el death cross si nunca lo alcanza).
      3. objetivo_adaptativo_ticker -> el objetivo de toma de ganancia se
                              CALCULA para cada ticker usando SOLO sus propios
                              cruces PASADOS (walk-forward, sin ver el futuro):
                              toma la mediana de la ganancia máxima que alcanzó
                              ese ticker en cruces anteriores antes de revertir,
                              y usa un % de esa mediana como objetivo. Si el
                              ticker no tiene suficiente historia propia todavía
                              (< MIN_CRUCES_PREVIOS_ADAPTATIVO cruces previos),
                              usa el objetivo fijo de respaldo.

Corre: python backtest.py --capital 1000
       python backtest.py AAPL MSFT NVDA --capital 1000   (subset rápido)
"""

import pandas as pd
import numpy as np
from config import get_connection

MIN_HISTORIA = 250
UMBRAL_SALTO_SOSPECHOSO = 0.50       # 50% en un día = probable split/error de datos
OBJETIVO_FIJO_DEFAULT_PCT = 15.0     # % de ganancia para tomar utilidad (estrategia "objetivo_fijo")
CAPTURA_RATIO_ADAPTATIVO = 0.6       # qué fracción de la ganancia máxima histórica del ticker se busca capturar
MIN_CRUCES_PREVIOS_ADAPTATIVO = 2    # cruces previos del mismo ticker necesarios para calibrar; si no hay, usa el default


def cargar_precios(conn, ticker):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def calcular_medias(df):
    df = df.copy()
    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["sma50"] = df["cierre"].rolling(window=50).mean()
    df["sma200"] = df["cierre"].rolling(window=200).mean()
    return df


def marcar_saltos_sospechosos(df):
    """Marca fechas con salto >50% en un día (reverse split / delisting / error de datos)."""
    df = df.copy()
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO
    return df


def encontrar_cruces(df, col_rapida, col_lenta):
    """Regresa lista de (index_en_df, tipo) para todos los cruces históricos, en orden cronológico."""
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


def hay_salto_sospechoso_en_tramo(df, idx_inicio, idx_fin):
    tramo = df["salto_sospechoso"].iloc[idx_inicio:idx_fin + 1]
    return bool(tramo.any())


def calcular_mfe_pct(df, idx_entrada, idx_fin):
    """Maximum Favorable Excursion: ganancia % máxima alcanzada entre entrada y idx_fin (long)."""
    precio_entrada = df["cierre"].iloc[idx_entrada]
    tramo = df["cierre"].iloc[idx_entrada:idx_fin + 1]
    precio_max = tramo.max()
    return (precio_max - precio_entrada) / precio_entrada * 100


def simular_salida(df, idx_entrada, idx_limite, objetivo_pct=None):
    """
    idx_limite = índice del siguiente cruce (cualquier tipo) o fin de datos si no hay más.
    Si objetivo_pct se especifica, sale el primer día que la ganancia desde la entrada
    lo alcanza; si nunca lo alcanza, sale en idx_limite (cruce contrario / fin de datos).
    Regresa (idx_salida, motivo).
    """
    precio_entrada = df["cierre"].iloc[idx_entrada]
    if objetivo_pct is not None:
        for idx in range(idx_entrada, idx_limite + 1):
            ganancia = (df["cierre"].iloc[idx] - precio_entrada) / precio_entrada * 100
            if ganancia >= objetivo_pct:
                return idx, "objetivo_alcanzado"
    return idx_limite, "cruce_contrario_o_fin_datos"


def calcular_retorno_anualizado(retorno_pct, dias):
    dias = max(dias, 1)
    return ((1 + retorno_pct / 100) ** (365 / dias) - 1) * 100


def procesar_ticker(conn, ticker):
    df = cargar_precios(conn, ticker)
    if len(df) < MIN_HISTORIA:
        return [], 0

    df = calcular_medias(df)
    df = marcar_saltos_sospechosos(df)
    resultados = []
    excluidas_por_salto = 0

    for promedio_tipo, (col_r, col_l) in [("EMA", ("ema50", "ema200")), ("SMA", ("sma50", "sma200"))]:
        cruces = encontrar_cruces(df, col_r, col_l)
        historial_mfe = []  # se va llenando SOLO con cruces ya pasados de este ticker (walk-forward)

        for i, (idx_cruce, tipo_cruce) in enumerate(cruces):
            if tipo_cruce != "dorado":
                continue  # solo long: no abrimos operación en death cross

            idx_limite = cruces[i + 1][0] if i + 1 < len(cruces) else len(df) - 1

            if hay_salto_sospechoso_en_tramo(df, idx_cruce, idx_limite):
                excluidas_por_salto += 1
                continue

            # objetivo adaptativo: mediana de MFE de cruces PREVIOS de este mismo ticker
            if len(historial_mfe) >= MIN_CRUCES_PREVIOS_ADAPTATIVO:
                objetivo_adaptativo = float(np.median(historial_mfe)) * CAPTURA_RATIO_ADAPTATIVO
                objetivo_adaptativo = max(objetivo_adaptativo, 2.0)  # piso mínimo, evita objetivos ridículamente chicos
            else:
                objetivo_adaptativo = OBJETIVO_FIJO_DEFAULT_PCT

            # registra el MFE de ESTE cruce para calibrar cruces FUTUROS del mismo ticker
            mfe_este_cruce = calcular_mfe_pct(df, idx_cruce, idx_limite)
            historial_mfe.append(mfe_este_cruce)

            for estrategia_entrada, idx_entrada in [
                ("confirmado", idx_cruce),
                ("anticipado", max(0, idx_cruce - 3)),
            ]:
                if idx_entrada < idx_cruce and hay_salto_sospechoso_en_tramo(df, idx_entrada, idx_cruce):
                    continue  # el tramo de anticipación también debe estar limpio

                for estrategia_salida, objetivo_pct in [
                    ("cruce_contrario", None),
                    ("objetivo_fijo", OBJETIVO_FIJO_DEFAULT_PCT),
                    ("objetivo_adaptativo_ticker", objetivo_adaptativo),
                ]:
                    idx_salida, motivo = simular_salida(df, idx_entrada, idx_limite, objetivo_pct)

                    precio_entrada = df["cierre"].iloc[idx_entrada]
                    precio_salida = df["cierre"].iloc[idx_salida]
                    if precio_entrada == 0:
                        continue
                    retorno_pct = (precio_salida - precio_entrada) / precio_entrada * 100

                    fecha_entrada = df["fecha"].iloc[idx_entrada]
                    fecha_salida = df["fecha"].iloc[idx_salida]
                    dias_en_operacion = max((fecha_salida - fecha_entrada).days, 0)
                    retorno_anualizado_pct = calcular_retorno_anualizado(retorno_pct, dias_en_operacion)

                    resultados.append({
                        "ticker": ticker,
                        "promedio_tipo": promedio_tipo,
                        "estrategia": estrategia_entrada,
                        "direccion": "long",
                        "estrategia_salida": estrategia_salida,
                        "fecha_entrada": str(fecha_entrada.date()),
                        "fecha_salida": str(fecha_salida.date()),
                        "dias_en_operacion": dias_en_operacion,
                        "retorno_pct": retorno_pct,
                        "retorno_anualizado_pct": retorno_anualizado_pct,
                        "fue_falso_positivo": 0,
                    })

    return resultados, excluidas_por_salto


def correr_backtest_completo(tickers=None, capital_por_trade=1000):
    conn = get_connection()

    if tickers is None:
        tickers = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM tickers WHERE activo = 1 AND tipo = 'stock'"
        ).fetchall()]

    print(f"Corriendo backtest (solo LONG) sobre {len(tickers)} tickers...")
    todos_resultados = []
    total_excluidas = 0

    for i, ticker in enumerate(tickers):
        res, excluidas = procesar_ticker(conn, ticker)
        todos_resultados.extend(res)
        total_excluidas += excluidas
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(res)} operaciones simuladas")

    if total_excluidas > 0:
        print(f"\n⚠️  Se excluyeron {total_excluidas} operaciones por cruzar un salto de precio "
              f">±{UMBRAL_SALTO_SOSPECHOSO*100:.0f}% en un día (probable split/error de datos).")

    if not todos_resultados:
        print("⚠️  No hay suficiente historia para correr el backtest todavía.")
        conn.close()
        return

    # limpiar tabla de resultados anteriores de esta corrida antes de insertar (evita duplicar históricos viejos)
    conn.execute("DELETE FROM backtest_resultados")
    conn.executemany("""
        INSERT INTO backtest_resultados
            (ticker, estrategia, direccion, promedio_tipo, estrategia_salida,
             fecha_entrada, fecha_salida, dias_en_operacion, retorno_pct,
             retorno_anualizado_pct, fue_falso_positivo)
        VALUES (:ticker, :estrategia, :direccion, :promedio_tipo, :estrategia_salida,
                :fecha_entrada, :fecha_salida, :dias_en_operacion, :retorno_pct,
                :retorno_anualizado_pct, :fue_falso_positivo)
    """, todos_resultados)
    conn.commit()
    conn.close()

    imprimir_resumen(todos_resultados, capital_por_trade)


def imprimir_resumen(resultados, capital_por_trade=1000):
    df = pd.DataFrame(resultados)
    df["ganancia_usd"] = capital_por_trade * (df["retorno_pct"] / 100)

    print("\n" + "=" * 115)
    print(f"RESUMEN DEL BACKTEST (solo LONG) — capital fijo de ${capital_por_trade:,.0f} USD por operación")
    print("=" * 115)

    resumen = df.groupby(["promedio_tipo", "estrategia", "estrategia_salida"]).agg(
        operaciones=("retorno_pct", "count"),
        operaciones_ganadoras=("retorno_pct", lambda s: int((s > 0).sum())),
        operaciones_perdedoras=("retorno_pct", lambda s: int((s <= 0).sum())),
        pct_ganadoras=("retorno_pct", lambda s: (s > 0).mean() * 100),
        ganancia_promedio_de_ganadoras_pct=("retorno_pct", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
        perdida_promedio_de_perdedoras_pct=("retorno_pct", lambda s: s[s <= 0].mean() if (s <= 0).any() else 0),
        dias_promedio_hold=("dias_en_operacion", "mean"),
        dias_mediana_hold=("dias_en_operacion", "median"),
        retorno_promedio_pct=("retorno_pct", "mean"),
        retorno_mediana_pct=("retorno_pct", "median"),
        retorno_anualizado_mediano_pct=("retorno_anualizado_pct", "median"),
        peor_operacion_pct=("retorno_pct", "min"),
        mejor_operacion_pct=("retorno_pct", "max"),
        ganancia_usd_TOTAL=("ganancia_usd", "sum"),
    ).round(2)

    resumen["capital_invertido_total"] = (resumen["operaciones"] * capital_por_trade).round(2)
    resumen["roi_total_pct"] = (
        resumen["ganancia_usd_TOTAL"] / resumen["capital_invertido_total"] * 100
    ).round(2)

    print(resumen.to_string())
    print("=" * 115)
    print("\nLectura rápida:")
    print("- 'operaciones_ganadoras' / 'operaciones_perdedoras' + sus promedios por separado son el")
    print("  desglose real: compara 'ganancia_promedio_de_ganadoras_pct' contra 'perdida_promedio_de")
    print("  _perdedoras_pct' — si las ganadoras ganan mucho más de lo que pierden las perdedoras,")
    print("  la estrategia puede ser rentable aunque el % de acierto no sea alto.")
    print("- ESTE AGREGADO ES GLOBAL: junta TODAS las operaciones de TODOS los tickers de una vez,")
    print("  no es 'el retorno de una acción'. Las acciones con subidón que ves en TradingView SÍ están")
    print("  incluidas aquí — pero también miles de señales mediocres o fallidas de otras acciones que")
    print("  normalmente no revisas manualmente, y eso jala el promedio hacia abajo.")
    print("- 'retorno_promedio_pct' / 'retorno_mediana_pct' = por OPERACIÓN individual, NO anual.")
    print("- 'retorno_anualizado_mediano_pct' = ese mismo retorno por operación, extrapolado matemáticamente")
    print("  a 1 año completo — NO es un año calendario real observado.")
    print("- La fila 'objetivo_fijo' ES 'gano 15% y me retiro' — ya está en la tabla.")
    print("- Compara las 3 filas de 'estrategia_salida' (misma fila de promedio_tipo/estrategia) para ver")
    print("  qué momento de salida da mejor resultado.")

    return resumen


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*", help="Tickers específicos para probar rápido, ej: AAPL MSFT")
    parser.add_argument("--capital", type=float, default=1000, help="Capital fijo por operación en USD (default: 1000)")
    args = parser.parse_args()

    correr_backtest_completo(
        tickers=args.tickers if args.tickers else None,
        capital_por_trade=args.capital,
    )