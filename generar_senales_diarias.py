"""
generar_senales_diarias.py

Corre TODOS LOS DÍAS (vía Railway cron). Aplica EXACTAMENTE el criterio
validado en el prospecto:
  - Universo: NASDAQ+NYSE, market cap >= $300M, sin REITs
  - Entrada: predicción anticipada (proyección EMA <= 4 días), solo si el
    ticker tiene >= SCORE_MINIMO% de acierto en sus predicciones PASADAS
    (walk-forward real, nunca ve el futuro — ni siquiera el de HOY, que
    todavía no se sabe si se cumplirá).
  - Objetivo aproximado: calibrado con el historial de ganancia máxima de
    los cruces CONFIRMADOS pasados de ese mismo ticker (mismo método que
    la salida adaptativa validada).

Arma un mensaje de Telegram con cada señal del día: empresa, industria,
market cap, precio de cruce, y % aproximado esperado — y lo manda.

Corre: python generar_senales_diarias.py
"""

import numpy as np
import pandas as pd
import requests
from config import get_connection, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

MIN_HISTORIA = 250
MAX_DIAS_ANTICIPACION = 4
VENTANA_VALIDACION_DIAS = 10
ESCENARIOS_PRECIO = [0.0, 0.005, 0.01, -0.005, -0.01]
UMBRAL_SALTO_SOSPECHOSO = 0.50
CAPTURA_RATIO_ADAPTATIVO = 0.6
OBJETIVO_FIJO_DEFAULT_PCT = 15.0
MIN_CRUCES_PREVIOS_ADAPTATIVO = 2
MIN_PREDICCIONES_PREVIAS_SCORE = 2
SCORE_MINIMO = 70.0
MARKET_CAP_MINIMO = 300_000_000
INDUSTRIAS_EXCLUIDAS = [
    "REAL ESTATE INVESTMENT TRUSTS",
    "SERVICES-BUSINESS SERVICES, NEC",
    "MISCELLANEOUS ELECTRICAL MACHINERY, EQUIPMENT & SUPPLIES",
    "OPERATIVE BUILDERS",
    "LABORATORY ANALYTICAL INSTRUMENTS",
    "INDUSTRIAL ORGANIC CHEMICALS",
    "RETAIL-AUTO DEALERS & GASOLINE STATIONS",
    "WATER SUPPLY",
]


def formatear_market_cap(mc):
    if mc is None:
        return "Sin dato"
    if mc >= 10_000_000_000:
        return f"${mc/1e9:.1f}B (Large-cap)"
    if mc >= 2_000_000_000:
        return f"${mc/1e9:.1f}B (Mid-cap)"
    return f"${mc/1e6:.0f}M (Small-cap)"


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


def hay_salto_en_tramo(df, idx_inicio, idx_fin):
    return bool(df["salto_sospechoso"].iloc[idx_inicio:idx_fin + 1].any())


def calcular_mfe_pct(df, idx_entrada, idx_fin):
    precio_entrada = df["cierre"].iloc[idx_entrada]
    precio_max = df["cierre"].iloc[idx_entrada:idx_fin + 1].max()
    return (precio_max - precio_entrada) / precio_entrada * 100


def dias_hasta_objetivo(df, idx_entrada, idx_fin, objetivo_pct):
    """Cuántos días tomó, EN ESE cruce pasado, llegar al objetivo (o hasta el límite si nunca llegó)."""
    precio_entrada = df["cierre"].iloc[idx_entrada]
    for idx in range(idx_entrada, idx_fin + 1):
        ganancia = (df["cierre"].iloc[idx] - precio_entrada) / precio_entrada * 100
        if ganancia >= objetivo_pct:
            return idx - idx_entrada
    return idx_fin - idx_entrada


def evaluar_ticker_hoy(conn, ticker, market_cap, industria):
    """
    Regresa (señal_completa, candidato_hoy):
      - señal_completa: dict con la señal que SÍ pasó todo el criterio (o None)
      - candidato_hoy: dict con info de que hoy hubo actividad de proyección
        (aunque no haya pasado el filtro), o None si el ticker ni siquiera
        está en fase de "acercándose" a un cruce.
    """
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA + VENTANA_VALIDACION_DIAS + 1:
        return None, None

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO

    cruces = encontrar_cruces(df, "ema50", "ema200")
    mfe_por_cruce = {}
    idx_limite_por_cruce = {}
    for j, (idx_c, tipo_c) in enumerate(cruces):
        if tipo_c != "dorado":
            continue
        idx_lim = cruces[j + 1][0] if j + 1 < len(cruces) else len(df) - 1
        if hay_salto_en_tramo(df, idx_c, idx_lim):
            continue
        mfe_por_cruce[idx_c] = calcular_mfe_pct(df, idx_c, idx_lim)
        idx_limite_por_cruce[idx_c] = idx_lim
    dorados_confirmados_ordenados = sorted(mfe_por_cruce.keys())
    dias_cruce_dorado_real = set(idx for idx, tipo in cruces if tipo == "dorado")

    ultimo_idx = len(df) - 1
    limite_validable = ultimo_idx - VENTANA_VALIDACION_DIAS  # último índice con futuro suficiente para saber si se cumplió

    eventos_prediccion_previos = []
    en_señal = False
    candidato_hoy = None

    for i in range(MIN_HISTORIA, len(df)):
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
        n_min = None
        if resultados_n:
            n_min = min(resultados_n)
            es_señal_fuerte = (n_min <= MAX_DIAS_ANTICIPACION) and \
                               (len(resultados_n) >= (len(ESCENARIOS_PRECIO) // 2 + 1))

        if es_señal_fuerte and not en_señal:
            en_señal = True

            if i <= limite_validable:
                # evento histórico: ya sabemos si se cumplió -> alimenta el score para el futuro
                ventana = list(range(i + 1, min(i + 1 + VENTANA_VALIDACION_DIAS, len(df))))
                idx_cruce_real = next((idx for idx in ventana if idx in dias_cruce_dorado_real), None)
                eventos_prediccion_previos.append(idx_cruce_real is not None)
            elif i == ultimo_idx:
                # es HOY: evaluamos si pasa el filtro, sin saber todavía si se cumplirá
                n_prev = len(eventos_prediccion_previos)
                score = (sum(eventos_prediccion_previos) / n_prev * 100) if n_prev >= MIN_PREDICCIONES_PREVIAS_SCORE else None

                # candidato: hoy SÍ hay proyección fuerte de cruce, se guarda pase o no el filtro
                candidato_hoy = {
                    "ticker": ticker,
                    "n_min_dias": n_min,
                    "score_pct": round(score, 1) if score is not None else None,
                }

                if score is not None and score >= SCORE_MINIMO:
                    mfes_previos = [mfe_por_cruce[idx_c] for idx_c in dorados_confirmados_ordenados if idx_c < i]
                    if len(mfes_previos) >= MIN_CRUCES_PREVIOS_ADAPTATIVO:
                        objetivo_pct = max(float(np.median(mfes_previos)) * CAPTURA_RATIO_ADAPTATIVO, 2.0)
                    else:
                        objetivo_pct = OBJETIVO_FIJO_DEFAULT_PCT

                    # HOLD estimado: días que tomó, en los cruces pasados de ESTE ticker,
                    # llegar a este mismo objetivo % (mediana de esos casos)
                    dias_previos = [
                        dias_hasta_objetivo(df, idx_c, idx_limite_por_cruce[idx_c], objetivo_pct)
                        for idx_c in dorados_confirmados_ordenados if idx_c < i
                    ]
                    hold_estimado_dias = int(np.median(dias_previos)) if dias_previos else None

                    señal_completa = {
                        "ticker": ticker,
                        "industria": industria if industria else "Sin dato",
                        "market_cap_texto": formatear_market_cap(market_cap),
                        "precio": hoy["cierre"],
                        "score_pct": round(score, 1),
                        "objetivo_pct": round(objetivo_pct, 1),
                        "hold_estimado_dias": hold_estimado_dias,
                        "num_cruces_previos_objetivo": len(mfes_previos),
                    }
                    return señal_completa, candidato_hoy
        elif not es_señal_fuerte:
            en_señal = False

    return None, candidato_hoy


def obtener_universo_filtrado(conn):
    filas = conn.execute("""
        SELECT ticker, market_cap, industria FROM tickers
        WHERE activo = 1 AND tipo = 'stock'
    """).fetchall()
    resultado = []
    for r in filas:
        if r["market_cap"] is not None and r["market_cap"] < MARKET_CAP_MINIMO:
            continue
        if r["industria"] in INDUSTRIAS_EXCLUIDAS:
            continue
        resultado.append((r["ticker"], r["market_cap"], r["industria"]))
    return resultado


def armar_mensajes(señales, candidatos, total_universo):
    from datetime import datetime, timezone

    bloques = []
    for s in señales:
        hold_texto = f"{s['hold_estimado_dias']} días" if s['hold_estimado_dias'] is not None else "N/A"
        link_tradingview = f"https://www.tradingview.com/symbols/{s['ticker']}/"
        bloques.append(
            f"TRADE - ${s['ticker']}\n"
            f"LONG - ${s['precio']:.2f}\n"
            f"% EST - {s['objetivo_pct']:.1f}%\n"
            f"HOLD - {hold_texto}\n"
            f"LINK - 🔗 {link_tradingview}"
        )

    # candidatos: los que están más cerca (n_min más chico) primero
    candidatos_ordenados = sorted(candidatos, key=lambda c: c["n_min_dias"])[:5]
    lineas_candidatos = []
    for c in candidatos_ordenados:
        score_texto = f"{c['score_pct']:.0f}%" if c["score_pct"] is not None else "SIN HISTORIAL"
        lineas_candidatos.append(f"  {c['ticker']} — proyecta cruce en ~{c['n_min_dias']}d — SCORE {score_texto}")

    fecha_texto = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    resumen = (
        f"GOLDEN CROSS DAILY\n"
        f"Universo: `{total_universo} stocks`\n"
        f"Señales hoy: `{len(señales)}`\n"
        f"Candidatos vigilando: `{len(candidatos)}`\n"
    )
    if lineas_candidatos:
        resumen += "TOP CERCANOS:\n" + "\n".join(lineas_candidatos) + "\n"
    resumen += fecha_texto

    encabezado = f"📊 *{len(señales)} señal(es) de Golden Cross anticipado hoy:*\n" if señales else None

    mensajes = [resumen]
    if bloques:
        grupo = [encabezado]
        for i, bloque in enumerate(bloques, 1):
            grupo.append(bloque)
            if i % 10 == 0:
                mensajes.append("\n\n".join(grupo))
                grupo = []
        if len(grupo) > 1:
            mensajes.append("\n\n".join(grupo))
    return mensajes


def enviar_telegram(texto):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "parse_mode": "Markdown"})
    return r.status_code == 200


def correr():
    conn = get_connection()
    universo = obtener_universo_filtrado(conn)
    print(f"Universo filtrado: {len(universo)} tickers. Evaluando señales de hoy...")

    señales = []
    candidatos = []
    for i, (ticker, market_cap, industria) in enumerate(universo):
        señal, candidato = evaluar_ticker_hoy(conn, ticker, market_cap, industria)
        if señal:
            señales.append(señal)
            print(f"  ✅ Señal: {ticker}")
        if candidato:
            candidatos.append(candidato)
        if i % 500 == 0:
            print(f"  [{i}/{len(universo)}] procesados...")

    conn.close()

    print(f"\nTotal de señales hoy: {len(señales)}")
    print(f"Total de candidatos cercanos: {len(candidatos)}")
    mensajes = armar_mensajes(señales, candidatos, len(universo))
    for m in mensajes:
        ok = enviar_telegram(m)
        print("  ✅ Enviado a Telegram" if ok else "  ⚠️ Falló el envío a Telegram")


if __name__ == "__main__":
    correr()