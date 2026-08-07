"""
evaluar_predicciones.py

Corre el modelo de proyección (la fórmula cerrada de EMA de modelo_proyeccion.py)
DÍA POR DÍA sobre el histórico, viendo SOLO datos hasta ese día (sin trampa,
sin ver el futuro), y compara cada predicción contra lo que realmente pasó
después: ¿el golden cross proyectado se cumplió, o fue una falsa alarma?

Esto es distinto al "anticipado" de backtest.py (que aproxima entrando N días
antes de un cruce que YA SABEMOS que ocurrió). Aquí sí puede fallar de verdad.

Método:
  - Para cada ticker, cada día que el precio está por debajo del EMA200
    (o sea, antes de un golden cross), corre la proyección con varios
    supuestos de precio futuro (igual que modelo_proyeccion.py).
  - Si proyecta un cruce en <= MAX_DIAS_ANTICIPACION días Y la mayoría de
    los escenarios de precio coinciden (señal "fuerte"), se registra como
    una PREDICCIÓN.
  - Se valida contra el futuro real (que sí tenemos, por ser backtest):
    ¿hubo un golden cross real dentro de los siguientes VENTANA_VALIDACION_DIAS?
  - Solo se cuenta una predicción NUEVA cuando la señal fuerte "empieza"
    (evita contar el mismo evento varias veces mientras se mantiene la señal).

Corre: python evaluar_predicciones.py
       python evaluar_predicciones.py AAPL MSFT NVDA   (subset rápido)
"""

import numpy as np
import pandas as pd
from config import get_connection

MIN_HISTORIA = 250
MAX_DIAS_ANTICIPACION = 4
VENTANA_VALIDACION_DIAS = 10   # cuántos días hacia adelante se le da chance de cumplirse
ESCENARIOS_PRECIO = [0.0, 0.005, 0.01, -0.005, -0.01]
UMBRAL_SALTO_SOSPECHOSO = 0.50


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


def evaluar_ticker(conn, ticker):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA + VENTANA_VALIDACION_DIAS:
        return []

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO

    diff = df["ema50"] - df["ema200"]
    signo = np.sign(diff)
    cambio = signo.diff()
    dias_cruce_dorado_real = set(df.index[cambio == 2])

    predicciones = []
    en_señal = False

    for i in range(MIN_HISTORIA, len(df) - VENTANA_VALIDACION_DIAS):
        hoy = df.iloc[i]
        if pd.isna(hoy["ema50"]) or pd.isna(hoy["ema200"]) or bool(df["salto_sospechoso"].iloc[i]):
            en_señal = False
            continue

        if hoy["ema50"] >= hoy["ema200"]:
            en_señal = False  # ya cruzó o ya está arriba, no hay nada que anticipar
            continue

        resultados_n = []
        for drift in ESCENARIOS_PRECIO:
            n = dias_hasta_cruce(hoy["cierre"], hoy["ema50"], hoy["ema200"], drift, max_dias=10)
            if n is not None:
                resultados_n.append(n)

        es_señal_fuerte = False
        n_min = None
        if resultados_n:
            n_min = min(resultados_n)
            es_señal_fuerte = (n_min <= MAX_DIAS_ANTICIPACION) and \
                               (len(resultados_n) >= (len(ESCENARIOS_PRECIO) // 2 + 1))

        if es_señal_fuerte and not en_señal:
            # nueva predicción (primer día que aparece la señal fuerte para este evento)
            ventana = range(i + 1, min(i + 1 + VENTANA_VALIDACION_DIAS, len(df)))
            dias_reales = [idx - i for idx in ventana if idx in dias_cruce_dorado_real]
            se_cumplio = len(dias_reales) > 0

            predicciones.append({
                "ticker": ticker,
                "fecha_prediccion": str(hoy["fecha"].date()),
                "n_min_proyectado": n_min,
                "se_cumplio": se_cumplio,
                "dias_reales_hasta_cruce": dias_reales[0] if se_cumplio else None,
            })
            en_señal = True
        elif not es_señal_fuerte:
            en_señal = False

    return predicciones


def correr_evaluacion(tickers=None):
    conn = get_connection()
    if tickers is None:
        tickers = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM tickers WHERE activo = 1 AND tipo = 'stock'"
        ).fetchall()]

    print(f"Evaluando precisión de predicciones sobre {len(tickers)} tickers...")
    todas_predicciones = []

    for i, ticker in enumerate(tickers):
        preds = evaluar_ticker(conn, ticker)
        todas_predicciones.extend(preds)
        if i % 200 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(preds)} predicciones evaluadas")

    conn.close()

    if not todas_predicciones:
        print("⚠️  No se generaron predicciones (revisa que haya suficiente historia).")
        return

    df = pd.DataFrame(todas_predicciones)
    total = len(df)
    cumplidas = int(df["se_cumplio"].sum())
    falsas = total - cumplidas
    pct_acierto = cumplidas / total * 100

    print("\n" + "=" * 70)
    print("PRECISIÓN DEL MODELO DE PROYECCIÓN (día por día, sin ver el futuro)")
    print("=" * 70)
    print(f"Total de predicciones hechas:        {total}")
    print(f"Se cumplieron (cruce real ocurrió):  {cumplidas}  ({pct_acierto:.1f}%)")
    print(f"Falsas alarmas (nunca cruzó):         {falsas}  ({100 - pct_acierto:.1f}%)")

    cumplidas_df = df[df["se_cumplio"]]
    if not cumplidas_df.empty:
        print(f"\nDe las que SÍ se cumplieron:")
        print(f"  Días proyectados (promedio):  {cumplidas_df['n_min_proyectado'].mean():.1f}")
        print(f"  Días reales hasta el cruce (promedio): {cumplidas_df['dias_reales_hasta_cruce'].mean():.1f}")

    print("=" * 70)
    print("\nLectura rápida:")
    print("- Este es el % de acierto REAL del modelo de proyección, sin la ventaja de ver el futuro.")
    print("- Compáralo contra lo que asumía 'anticipado' en backtest.py (que por diseño nunca falla) —")
    print("  si este % de acierto es bajo, la versión de backtest.py fue demasiado optimista y el")
    print("  retorno de las filas 'anticipado' probablemente esté sobreestimado en la práctica real.")


if __name__ == "__main__":
    import sys
    correr_evaluacion(tickers=sys.argv[1:] if len(sys.argv) > 1 else None)
