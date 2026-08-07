"""
monetizar_predicciones.py

Extiende evaluar_predicciones.py: para cada predicción (día por día, sin ver
el futuro), simula el retorno $ real de DOS formas para el mismo evento:
  (a) ENTRAR EN LA PREDICCIÓN de inmediato (aunque a veces sea falsa alarma)
  (b) ENTRAR SOLO SI SE CONFIRMA el cruce (esperando, como haría alguien
      escéptico del modelo de proyección)
Y las compara en dólares — para saber si anticipar vale la pena en la
práctica, no solo qué tan seguido acierta.

También calcula un SCORE POR TICKER: la tasa de acierto histórica de
ESE ticker específico en sus predicciones pasadas (walk-forward, sin ver
el futuro), para filtrar y confiar en anticipar solo en acciones que ya
demostraron ser predecibles.

Corre: python monetizar_predicciones.py
       python monetizar_predicciones.py AAPL MSFT NVDA
"""

import numpy as np
import pandas as pd
from config import get_connection

MIN_HISTORIA = 250
MAX_DIAS_ANTICIPACION = 4
VENTANA_VALIDACION_DIAS = 10
ESCENARIOS_PRECIO = [0.0, 0.005, 0.01, -0.005, -0.01]
UMBRAL_SALTO_SOSPECHOSO = 0.50
CAPITAL_POR_TRADE_DEFAULT = 1000
MIN_PREVIAS_SCORE_DEFAULT = 2


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

    eventos = []
    en_señal = False

    for i in range(MIN_HISTORIA, len(df) - VENTANA_VALIDACION_DIAS):
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
            es_señal_fuerte = (n_min <= MAX_DIAS_ANTICIPACION) and \
                               (len(resultados_n) >= (len(ESCENARIOS_PRECIO) // 2 + 1))

        if es_señal_fuerte and not en_señal:
            ventana = list(range(i + 1, min(i + 1 + VENTANA_VALIDACION_DIAS, len(df))))
            idx_cruce_real = next((idx for idx in ventana if idx in dias_cruce_dorado_real), None)
            se_cumplio = idx_cruce_real is not None

            precio_prediccion = hoy["cierre"]

            # (a) entrar en la predicción: sale en el cruce real si llegó; si no, corta al final de la ventana
            idx_salida_predicho = idx_cruce_real if se_cumplio else ventana[-1]
            precio_salida_predicho = df["cierre"].iloc[idx_salida_predicho]
            retorno_predicho_pct = (precio_salida_predicho - precio_prediccion) / precio_prediccion * 100

            # (b) entrar solo si se confirma: mismo punto de salida, para comparar manzanas con manzanas
            retorno_confirmado_pct = None
            if se_cumplio:
                precio_confirmado = df["cierre"].iloc[idx_cruce_real]
                precio_salida_conf = df["cierre"].iloc[ventana[-1]]
                retorno_confirmado_pct = (precio_salida_conf - precio_confirmado) / precio_confirmado * 100

            eventos.append({
                "ticker": ticker,
                "fecha_prediccion": str(hoy["fecha"].date()),
                "se_cumplio": se_cumplio,
                "retorno_predicho_pct": retorno_predicho_pct,
                "retorno_confirmado_pct": retorno_confirmado_pct,
            })
            en_señal = True
        elif not es_señal_fuerte:
            en_señal = False

    return eventos


def calcular_score_por_ticker(df):
    """Walk-forward: % de aciertos de ESE ticker en sus predicciones ANTERIORES (nunca futuro)."""
    df = df.sort_values(["ticker", "fecha_prediccion"]).reset_index(drop=True)
    scores, n_previas = [], []
    for _, grupo in df.groupby("ticker"):
        acumulado_ok = grupo["se_cumplio"].shift(1).expanding().sum()
        acumulado_total = grupo["se_cumplio"].shift(1).expanding().count()
        tasa = acumulado_ok / acumulado_total * 100
        scores.extend(tasa.tolist())
        n_previas.extend(acumulado_total.tolist())
    df["score_historico_ticker_pct"] = scores
    df["predicciones_previas_ticker"] = n_previas
    return df


def correr(tickers=None, capital=CAPITAL_POR_TRADE_DEFAULT, min_previas=MIN_PREVIAS_SCORE_DEFAULT):
    conn = get_connection()
    if tickers is None:
        tickers = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM tickers WHERE activo = 1 AND tipo = 'stock'"
        ).fetchall()]

    print(f"Monetizando predicciones sobre {len(tickers)} tickers...")
    todos = []
    for i, ticker in enumerate(tickers):
        eventos = evaluar_ticker(conn, ticker)
        todos.extend(eventos)
        if i % 200 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(eventos)} eventos")
    conn.close()

    if not todos:
        print("⚠️  No se generaron eventos.")
        return

    df = pd.DataFrame(todos)
    df["ganancia_predicho_usd"] = capital * (df["retorno_predicho_pct"] / 100)

    print("\n" + "=" * 90)
    print("MONETIZACIÓN: ¿VALE LA PENA ENTRAR EN LA PREDICCIÓN, O ESPERAR CONFIRMACIÓN?")
    print("=" * 90)
    print(f"Total de eventos: {len(df)}")
    print(f"  Se cumplieron:  {int(df['se_cumplio'].sum())} ({df['se_cumplio'].mean()*100:.1f}%)")
    print(f"  Falsas alarmas: {int((~df['se_cumplio']).sum())} ({(~df['se_cumplio']).mean()*100:.1f}%)")

    print(f"\n--- ENTRANDO EN LA PREDICCIÓN (incluye las falsas alarmas) ---")
    print(f"Retorno promedio por operación: {df['retorno_predicho_pct'].mean():.2f}%")
    print(f"Retorno mediana por operación:  {df['retorno_predicho_pct'].median():.2f}%")
    print(f"% de operaciones ganadoras:     {(df['retorno_predicho_pct'] > 0).mean()*100:.2f}%")
    print(f"Ganancia $ total (${capital:,.0f}/trade, sin componer): ${df['ganancia_predicho_usd'].sum():,.2f}")

    solo_cumplidas = df[df["se_cumplio"]]
    solo_falsas = df[~df["se_cumplio"]]
    print(f"\n  De las que SÍ se cumplieron: retorno promedio {solo_cumplidas['retorno_predicho_pct'].mean():.2f}%")
    if not solo_falsas.empty:
        costo_falsas = capital * (solo_falsas["retorno_predicho_pct"] / 100).sum()
        print(f"  De las FALSAS ALARMAS (el costo real del {(~df['se_cumplio']).mean()*100:.1f}%): "
              f"retorno promedio {solo_falsas['retorno_predicho_pct'].mean():.2f}%")
        print(f"  Costo/ganancia total en $ de las falsas alarmas: ${costo_falsas:,.2f}")

    conf = solo_cumplidas.dropna(subset=["retorno_confirmado_pct"])
    if not conf.empty:
        print(f"\n--- COMPARACIÓN DIRECTA (mismas señales, solo las que sí cruzaron) ---")
        print(f"Retorno promedio ENTRANDO EN LA PREDICCIÓN:  {conf['retorno_predicho_pct'].mean():.2f}%")
        print(f"Retorno promedio ESPERANDO CONFIRMACIÓN:      {conf['retorno_confirmado_pct'].mean():.2f}%")
        diferencia = conf["retorno_predicho_pct"].mean() - conf["retorno_confirmado_pct"].mean()
        print(f"Diferencia (positivo = SÍ vale la pena anticipar): {diferencia:+.2f} puntos porcentuales")
    print("=" * 90)

    df_score = calcular_score_por_ticker(df)
    con_historial = df_score[df_score["predicciones_previas_ticker"] >= min_previas]

    print(f"\n--- SCORE DE CONFIABILIDAD POR TICKER (walk-forward, mínimo {min_previas} predicciones previas) ---")
    print(f"{'Umbral de score previo':<25}{'Eventos':<12}{'% del total':<14}{'Retorno prom %':<18}{'% se cumplió':<15}")
    total = len(df)
    for umbral in [0, 40, 50, 60, 70, 80, 90, 100]:
        filtrado = con_historial[con_historial["score_historico_ticker_pct"] >= umbral]
        if len(filtrado) == 0:
            continue
        print(f">= {umbral}%{'':<19}{len(filtrado):<12}{len(filtrado)/total*100:<14.1f}"
              f"{filtrado['retorno_predicho_pct'].mean():<18.2f}{filtrado['se_cumplio'].mean()*100:<15.1f}")

    print("\nLectura rápida:")
    print("- Si el retorno 'ENTRANDO EN LA PREDICCIÓN' > 'ESPERANDO CONFIRMACIÓN', anticipar SÍ vale la")
    print("  pena en dinero, a pesar de las falsas alarmas — el tiempo ganado compensa el riesgo.")
    print("- La tabla de score por ticker muestra si confiar más en acciones con buen historial propio")
    print("  de predicciones cumplidas mejora el resultado (igual lógica que filtrar_calidad_señal.py,")
    print("  pero aplicada a qué tan predecible es CADA ticker, no a la salida de la operación).")


if __name__ == "__main__":
    import sys
    correr(tickers=sys.argv[1:] if len(sys.argv) > 1 else None)
