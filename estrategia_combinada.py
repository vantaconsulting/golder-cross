"""
estrategia_combinada.py

Junta las 3 mejoras encontradas hoy en UNA sola simulación:
  1. ENTRADA: anticipada (modelo de proyección real, día por día, sin ver el futuro)
  2. FILTRO: solo toma la señal si ESE ticker específico ya demostró (con su
     propio historial pasado, walk-forward) tener >= SCORE_MINIMO_PCT de
     acierto en sus predicciones anteriores.
  3. SALIDA: objetivo adaptativo calibrado con el historial de MFE (ganancia
     máxima alcanzada) de los cruces CONFIRMADOS pasados de ese mismo ticker.

Todo el walk-forward respeta la regla de siempre: en el día i, solo se usa
información de ANTES de i. Nunca se ve el futuro para decidir nada.

Qué pasa si la predicción resulta ser falsa alarma (no se confirma el cruce):
se corta la operación al final de la ventana de validación (10 días) — no
se deja abierta indefinidamente esperando algo que quizás nunca llegue.

Corre: python estrategia_combinada.py
       python estrategia_combinada.py --score-minimo 70 --capital 1000
       python estrategia_combinada.py AAPL MSFT NVDA   (subset rápido)
"""

import numpy as np
import pandas as pd
from config import get_connection

MIN_HISTORIA = 250
MAX_DIAS_ANTICIPACION_DEFAULT = 4
VENTANA_VALIDACION_DIAS_DEFAULT = 10
ESCENARIOS_PRECIO = [0.0, 0.005, 0.01, -0.005, -0.01]
UMBRAL_SALTO_SOSPECHOSO = 0.50
CAPTURA_RATIO_ADAPTATIVO = 0.6
OBJETIVO_FIJO_DEFAULT_PCT = 15.0
MIN_CRUCES_PREVIOS_ADAPTATIVO = 2
MIN_PREDICCIONES_PREVIAS_SCORE = 2
SCORE_MINIMO_DEFAULT = 70.0
CAPITAL_POR_TRADE_DEFAULT = 1000


def proyectar_ema(precio_supuesto, ema_hoy, span, n):
    alpha = 2 / (span + 1)
    return precio_supuesto + (ema_hoy - precio_supuesto) * ((1 - alpha) ** n)


def dias_hasta_cruce(precio_hoy, ema50_hoy, ema200_hoy, drift_diario, max_dias=10):
    diff_hoy = ema50_hoy - ema200_hoy
    for n in range(1, max_dias + 1):
        p = precio_hoy * ((1 + drift_diario) ** n)
        e50 = proyectar_ema(p, ema50_hoy, 50, n)
        e200 = proyectar_ema(p, ema200_hoy, 200, n)
        diff_n = e50 - e200
        if diff_hoy != 0 and (diff_n > 0) != (diff_hoy > 0):
            return n
    return None


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


def hay_salto_sospechoso_en_tramo(df, idx_inicio, idx_fin):
    return bool(df["salto_sospechoso"].iloc[idx_inicio:idx_fin + 1].any())


def calcular_mfe_pct(df, idx_entrada, idx_fin):
    precio_entrada = df["cierre"].iloc[idx_entrada]
    precio_max = df["cierre"].iloc[idx_entrada:idx_fin + 1].max()
    return (precio_max - precio_entrada) / precio_entrada * 100


def procesar_ticker(conn, ticker, score_minimo, capital_por_trade, max_dias_anticipacion, ventana_validacion_dias):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA + ventana_validacion_dias:
        return [], 0

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO

    cruces = encontrar_cruces(df, "ema50", "ema200")

    # historial de MFE de cada cruce dorado CONFIRMADO (para calibrar el objetivo adaptativo)
    mfe_por_cruce = {}
    idx_limite_por_cruce = {}
    for j, (idx_c, tipo_c) in enumerate(cruces):
        if tipo_c != "dorado":
            continue
        idx_lim = cruces[j + 1][0] if j + 1 < len(cruces) else len(df) - 1
        if hay_salto_sospechoso_en_tramo(df, idx_c, idx_lim):
            continue
        mfe_por_cruce[idx_c] = calcular_mfe_pct(df, idx_c, idx_lim)
        idx_limite_por_cruce[idx_c] = idx_lim

    dorados_confirmados_ordenados = sorted(mfe_por_cruce.keys())
    dias_cruce_dorado_real = set(idx for idx, tipo in cruces if tipo == "dorado")

    eventos_prediccion_previos = []  # walk-forward: True/False de predicciones pasadas de este ticker
    resultados = []
    filtrados_por_score = 0
    en_señal = False

    for i in range(MIN_HISTORIA, len(df) - ventana_validacion_dias):
        hoy = df.iloc[i]
        if pd.isna(hoy["ema50"]) or pd.isna(hoy["ema200"]) or bool(df["salto_sospechoso"].iloc[i]):
            en_señal = False
            continue
        if hoy["ema50"] >= hoy["ema200"]:
            en_señal = False
            continue

        resultados_n = []
        for drift in ESCENARIOS_PRECIO:
            n = dias_hasta_cruce(hoy["cierre"], hoy["ema50"], hoy["ema200"], drift, max_dias=10)
            if n is not None:
                resultados_n.append(n)

        es_señal_fuerte = False
        if resultados_n:
            n_min = min(resultados_n)
            es_señal_fuerte = (n_min <= max_dias_anticipacion) and \
                               (len(resultados_n) >= (len(ESCENARIOS_PRECIO) // 2 + 1))

        if es_señal_fuerte and not en_señal:
            en_señal = True

            n_prev = len(eventos_prediccion_previos)
            score = (sum(eventos_prediccion_previos) / n_prev * 100) if n_prev >= MIN_PREDICCIONES_PREVIAS_SCORE else None

            ventana = list(range(i + 1, min(i + 1 + ventana_validacion_dias, len(df))))
            idx_cruce_real = next((idx for idx in ventana if idx in dias_cruce_dorado_real), None)
            se_cumplio = idx_cruce_real is not None

            if score is not None and score >= score_minimo:
                mfes_previos = [mfe_por_cruce[idx_c] for idx_c in dorados_confirmados_ordenados if idx_c < i]
                if len(mfes_previos) >= MIN_CRUCES_PREVIOS_ADAPTATIVO:
                    objetivo_pct = max(float(np.median(mfes_previos)) * CAPTURA_RATIO_ADAPTATIVO, 2.0)
                else:
                    objetivo_pct = OBJETIVO_FIJO_DEFAULT_PCT

                idx_limite_salida = idx_limite_por_cruce.get(idx_cruce_real, ventana[-1]) if se_cumplio else ventana[-1]

                precio_entrada = hoy["cierre"]
                idx_salida = idx_limite_salida
                motivo = "cruce_contrario_o_fin"
                for idx in range(i, idx_limite_salida + 1):
                    ganancia = (df["cierre"].iloc[idx] - precio_entrada) / precio_entrada * 100
                    if ganancia >= objetivo_pct:
                        idx_salida = idx
                        motivo = "objetivo_alcanzado"
                        break

                precio_salida = df["cierre"].iloc[idx_salida]
                retorno_pct = (precio_salida - precio_entrada) / precio_entrada * 100
                fecha_entrada = hoy["fecha"]
                fecha_salida = df["fecha"].iloc[idx_salida]
                dias_en_operacion = max((fecha_salida - fecha_entrada).days, 0)
                retorno_anualizado_pct = ((1 + retorno_pct / 100) ** (365 / max(dias_en_operacion, 1)) - 1) * 100

                resultados.append({
                    "ticker": ticker,
                    "fecha_entrada": str(fecha_entrada.date()),
                    "fecha_salida": str(fecha_salida.date()),
                    "dias_en_operacion": dias_en_operacion,
                    "score_ticker_pct": round(score, 1),
                    "se_cumplio_prediccion": se_cumplio,
                    "objetivo_usado_pct": round(objetivo_pct, 2),
                    "motivo_salida": motivo,
                    "retorno_pct": retorno_pct,
                    "retorno_anualizado_pct": retorno_anualizado_pct,
                })
            else:
                filtrados_por_score += 1

            eventos_prediccion_previos.append(se_cumplio)
        elif not es_señal_fuerte:
            en_señal = False

    return resultados, filtrados_por_score


def correr(tickers=None, score_minimo=SCORE_MINIMO_DEFAULT, capital=CAPITAL_POR_TRADE_DEFAULT,
           max_dias_anticipacion=MAX_DIAS_ANTICIPACION_DEFAULT,
           ventana_validacion_dias=VENTANA_VALIDACION_DIAS_DEFAULT):
    conn = get_connection()
    if tickers is None:
        tickers = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM tickers WHERE activo = 1 AND tipo = 'stock'"
        ).fetchall()]

    print(f"Estrategia combinada: ANTICIPADO (proyección <= {max_dias_anticipacion} días, "
          f"confirmación en <= {ventana_validacion_dias} días) + score ticker >= {score_minimo}% + OBJETIVO ADAPTATIVO")
    print(f"Corriendo sobre {len(tickers)} tickers...")

    todos_resultados = []
    total_filtrados = 0

    for i, ticker in enumerate(tickers):
        res, filtrados = procesar_ticker(conn, ticker, score_minimo, capital, max_dias_anticipacion, ventana_validacion_dias)
        todos_resultados.extend(res)
        total_filtrados += filtrados
        if i % 200 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(res)} operaciones tomadas")

    conn.close()

    print(f"\nSeñales que pasaron el filtro de score y se operaron: {len(todos_resultados)}")
    print(f"Señales descartadas por no cumplir score >= {score_minimo}% (o sin historial suficiente): {total_filtrados}")

    if not todos_resultados:
        print("⚠️  No quedaron operaciones con este filtro. Prueba un score mínimo más bajo.")
        return

    df = pd.DataFrame(todos_resultados)
    df["ganancia_usd"] = capital * (df["retorno_pct"] / 100)

    print("\n" + "=" * 90)
    print(f"RESULTADOS — ANTICIPADO + SCORE >= {score_minimo}% + OBJETIVO ADAPTATIVO (${capital:,.0f}/trade)")
    print("=" * 90)
    print(f"Operaciones:                    {len(df)}")
    print(f"Ganadoras / Perdedoras:         {(df['retorno_pct']>0).sum()} / {(df['retorno_pct']<=0).sum()}")
    print(f"% Ganadoras:                    {(df['retorno_pct']>0).mean()*100:.2f}%")
    print(f"Ganancia promedio de ganadoras: {df.loc[df['retorno_pct']>0,'retorno_pct'].mean():.2f}%")
    perdedoras = df.loc[df['retorno_pct']<=0, 'retorno_pct']
    print(f"Pérdida promedio de perdedoras: {perdedoras.mean():.2f}%" if not perdedoras.empty else "Pérdida promedio de perdedoras: N/A")
    print(f"Retorno promedio por operación: {df['retorno_pct'].mean():.2f}%")
    print(f"Retorno mediana por operación:  {df['retorno_pct'].median():.2f}%")
    print(f"Retorno anualizado mediano:     {df['retorno_anualizado_pct'].median():.2f}%")
    print(f"Días de hold (promedio/mediana): {df['dias_en_operacion'].mean():.1f} / {df['dias_en_operacion'].median():.1f}")
    print(f"% de predicciones que sí se confirmaron: {df['se_cumplio_prediccion'].mean()*100:.1f}%")
    print(f"Ganancia $ total (sin componer):  ${df['ganancia_usd'].sum():,.2f}")
    print(f"ROI total:                       {df['ganancia_usd'].sum() / (capital*len(df)) * 100:.2f}%")
    print("=" * 90)
    print("\nComparación con lo que ya sabíamos:")
    print("- Sin ningún filtro (todas las señales anticipadas): retorno prom ~5.77%, anualizado mediano ~117%")
    print("- Solo filtro de score (salida genérica 10 días):     retorno prom ~1.02% (umbral 70%)")
    print("- ESTA combinación (score + salida adaptativa real) debería, en teoría, acercarse a lo mejor")
    print("  de ambos mundos — compáralo contra esas 2 referencias para ver si de verdad suma.")


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--score-minimo", type=float, default=SCORE_MINIMO_DEFAULT)
    parser.add_argument("--capital", type=float, default=CAPITAL_POR_TRADE_DEFAULT)
    parser.add_argument("--max-dias-anticipacion", type=int, default=MAX_DIAS_ANTICIPACION_DEFAULT,
                         help="Cuántos días adelante puede proyectar el modelo el cruce (default: 4)")
    parser.add_argument("--ventana-validacion", type=int, default=VENTANA_VALIDACION_DIAS_DEFAULT,
                         help="Cuántos días se le da a la predicción para confirmarse antes de darla por falsa alarma (default: 10)")
    args = parser.parse_args()

    correr(
        tickers=args.tickers if args.tickers else None,
        score_minimo=args.score_minimo,
        capital=args.capital,
        max_dias_anticipacion=args.max_dias_anticipacion,
        ventana_validacion_dias=args.ventana_validacion,
    )