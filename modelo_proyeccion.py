"""
modelo_proyeccion.py — Fase 7
Aplica la fórmula cerrada de proyección de EMA para estimar en cuántos
días (n) ocurriría el cruce, bajo varios supuestos de precio futuro.
Si el n estimado es <= 4 bajo un escenario razonable, registra el cruce
como 'anticipado'.

Corre: python modelo_proyeccion.py
"""

import pandas as pd
from config import get_connection
from calcular_emas import registrar_cruce

MAX_DIAS_ANTICIPACION = 4
ESCENARIOS_PRECIO = [0.0, 0.005, 0.01, -0.005, -0.01]  # plano, +0.5%, +1%, -0.5%, -1% por día


def proyectar_ema(precio_supuesto, ema_hoy, span, n):
    alpha = 2 / (span + 1)
    return precio_supuesto + (ema_hoy - precio_supuesto) * ((1 - alpha) ** n)


def dias_hasta_cruce(precio_hoy, ema50_hoy, ema200_hoy, drift_diario, max_dias=10):
    """
    Para un supuesto de drift diario del precio, busca el primer n (1..max_dias)
    donde EMA50 proyectada cruza EMA200 proyectada. Regresa n o None.
    """
    diff_hoy = ema50_hoy - ema200_hoy

    for n in range(1, max_dias + 1):
        p_supuesto = precio_hoy * ((1 + drift_diario) ** n)
        ema50_n = proyectar_ema(p_supuesto, ema50_hoy, 50, n)
        ema200_n = proyectar_ema(p_supuesto, ema200_hoy, 200, n)
        diff_n = ema50_n - ema200_n

        # cambio de signo respecto a hoy = cruce proyectado en el día n
        if diff_hoy != 0 and (diff_n > 0) != (diff_hoy > 0):
            return n
    return None


def evaluar_ticker(conn, ticker):
    df = pd.read_sql_query("""
        SELECT p.fecha, p.cierre, e.ema50, e.ema200
        FROM precios_diarios p
        JOIN ema_historico e ON e.ticker = p.ticker AND e.fecha = p.fecha
        WHERE p.ticker = ?
        ORDER BY p.fecha
    """, conn, params=(ticker,))

    if df.empty:
        return None

    hoy = df.iloc[-1]
    if pd.isna(hoy["ema50"]) or pd.isna(hoy["ema200"]):
        return None

    resultados_n = []
    for drift in ESCENARIOS_PRECIO:
        n = dias_hasta_cruce(hoy["cierre"], hoy["ema50"], hoy["ema200"], drift)
        if n is not None:
            resultados_n.append(n)

    if not resultados_n:
        return None  # ningún escenario proyecta cruce en el horizonte

    n_min = min(resultados_n)
    n_max = max(resultados_n)

    if n_min <= MAX_DIAS_ANTICIPACION:
        tipo = "dorado" if hoy["ema50"] < hoy["ema200"] else "muerte"
        return {
            "ticker": ticker,
            "tipo": tipo,
            "fecha": hoy["fecha"],
            "n_min": n_min,
            "n_max": n_max,
            "escenarios_que_cruzan": len(resultados_n),
            "total_escenarios": len(ESCENARIOS_PRECIO),
        }
    return None


def procesar_todos():
    conn = get_connection()
    tickers = [r["ticker"] for r in conn.execute(
        "SELECT ticker FROM tickers WHERE activo = 1"
    ).fetchall()]

    anticipados = []
    for ticker in tickers:
        res = evaluar_ticker(conn, ticker)
        if res:
            # Señal "fuerte" = la mayoría de los escenarios de precio coinciden en el cruce
            es_senal_fuerte = res["escenarios_que_cruzan"] >= (res["total_escenarios"] // 2 + 1)
            if es_senal_fuerte:
                registrar_cruce(conn, ticker, res["fecha"], res["tipo"], "EMA", metodo="anticipado")
                anticipados.append(res)

    conn.commit()
    conn.close()

    print(f"✅ Cruces anticipados detectados: {len(anticipados)}")
    for a in anticipados:
        print(f"  {a['ticker']}: {a['tipo']} en {a['n_min']}-{a['n_max']} días "
              f"({a['escenarios_que_cruzan']}/{a['total_escenarios']} escenarios de acuerdo)")

    return anticipados


if __name__ == "__main__":
    procesar_todos()
