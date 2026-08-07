"""
generar_senales_diarias.py (v2)

Corre TODOS LOS DÍAS (vía Railway cron). Dos conceptos separados:

  1. TRADE (accionable, se manda como mensaje separado por cada uno):
     Solo se dispara cuando el EMA50 CRUZÓ REALMENTE por encima del EMA200
     EL DÍA DE HOY (cruce confirmado, no proyección). El objetivo % y el
     HOLD estimado se calibran con el historial de cruces confirmados
     PASADOS de ese mismo ticker (walk-forward, sin ver el futuro).

  2. CANDIDATOS (informativo, solo en el resumen diario):
     Usa el modelo de proyección anticipada (igual que antes) para avisar
     qué tickers están "a punto de cruzar" en los próximos días — no
     genera ninguna alerta accionable, es solo para que sepas qué vigilar.

Universo: NASDAQ+NYSE, market cap >= $300M, sin las industrias excluidas.

Corre: python generar_senales_diarias.py
       python generar_senales_diarias.py --solo-resumen   (para el aviso previo, 2h antes del cron principal)
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
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

SUFIJOS_CORPORATIVOS = [
    " INCORPORATED", " CORPORATION", " COMPANY", " HOLDINGS", " GROUP",
    " LIMITED", " INC.", " CORP.", " CO.", " LTD.", " PLC", " INC", " CORP",
    " CO", " LTD",
]


def formatear_market_cap(mc):
    if mc is None:
        return "Sin dato"
    if mc >= 10_000_000_000:
        return f"${mc/1e9:.1f}B (Large-cap)"
    if mc >= 2_000_000_000:
        return f"${mc/1e9:.1f}B (Mid-cap)"
    return f"${mc/1e6:.0f}M (Small-cap)"


def acortar_nombre(nombre):
    """MAYÚSCULAS, sin sufijos corporativos comunes, truncado — para diferenciar
    tickers parecidos sin depender de ISIN/CUSIP/SEDOL (que Polygon no nos da)."""
    if not nombre:
        return "SIN NOMBRE"
    n = nombre.upper()
    for suf in SUFIJOS_CORPORATIVOS:
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = n.strip(" .,")
    return n[:30]


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
    precio_entrada = df["cierre"].iloc[idx_entrada]
    for idx in range(idx_entrada, idx_fin + 1):
        ganancia = (df["cierre"].iloc[idx] - precio_entrada) / precio_entrada * 100
        if ganancia >= objetivo_pct:
            return idx - idx_entrada
    return idx_fin - idx_entrada


def evaluar_ticker_hoy(conn, ticker, market_cap, industria, nombre):
    """
    Regresa (señal_completa, candidato_hoy):
      - señal_completa: dict SOLO si hoy hubo un cruce CONFIRMADO real (o None)
      - candidato_hoy: dict informativo si hay proyección anticipada fuerte
        hoy (independiente de si señal_completa existe), o None
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
    limite_validable = ultimo_idx - VENTANA_VALIDACION_DIAS

    # --- 1. SEÑAL COMPLETA (TRADE): cruce CONFIRMADO exactamente hoy ---
    señal_completa = None
    if cruces and cruces[-1][0] == ultimo_idx and cruces[-1][1] == "dorado" \
            and not bool(df["salto_sospechoso"].iloc[ultimo_idx]):
        mfes_previos = [mfe_por_cruce[idx] for idx in dorados_confirmados_ordenados if idx < ultimo_idx]
        if len(mfes_previos) >= MIN_CRUCES_PREVIOS_ADAPTATIVO:
            objetivo_pct = max(float(np.median(mfes_previos)) * CAPTURA_RATIO_ADAPTATIVO, 2.0)
        else:
            objetivo_pct = OBJETIVO_FIJO_DEFAULT_PCT

        dias_previos = [
            dias_hasta_objetivo(df, idx, idx_limite_por_cruce[idx], objetivo_pct)
            for idx in dorados_confirmados_ordenados if idx < ultimo_idx
        ]
        hold_estimado_dias = int(np.median(dias_previos)) if dias_previos else None

        señal_completa = {
            "ticker": ticker,
            "nombre_corto": acortar_nombre(nombre),
            "industria": industria if industria else "Sin dato",
            "market_cap_texto": formatear_market_cap(market_cap),
            "precio": df["cierre"].iloc[ultimo_idx],
            "objetivo_pct": round(objetivo_pct, 1),
            "hold_estimado_dias": hold_estimado_dias,
            "num_cruces_previos_objetivo": len(mfes_previos),
        }

    # --- 2. CANDIDATO (informativo): proyección anticipada, walk-forward ---
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
                ventana = list(range(i + 1, min(i + 1 + VENTANA_VALIDACION_DIAS, len(df))))
                idx_cruce_real = next((idx for idx in ventana if idx in dias_cruce_dorado_real), None)
                eventos_prediccion_previos.append(idx_cruce_real is not None)
            elif i == ultimo_idx:
                n_prev = len(eventos_prediccion_previos)
                score = (sum(eventos_prediccion_previos) / n_prev * 100) if n_prev >= MIN_PREDICCIONES_PREVIAS_SCORE else None
                candidato_hoy = {
                    "ticker": ticker,
                    "n_min_dias": n_min,
                    "score_pct": round(score, 1) if score is not None else None,
                }
        elif not es_señal_fuerte:
            en_señal = False

    return señal_completa, candidato_hoy


def obtener_universo_filtrado(conn):
    filas = conn.execute("""
        SELECT ticker, market_cap, industria, nombre FROM tickers
        WHERE activo = 1 AND tipo = 'stock'
    """).fetchall()
    resultado = []
    for r in filas:
        if r["market_cap"] is not None and r["market_cap"] < MARKET_CAP_MINIMO:
            continue
        if r["industria"] in INDUSTRIAS_EXCLUIDAS:
            continue
        resultado.append((r["ticker"], r["market_cap"], r["industria"], r["nombre"]))
    return resultado


def armar_mensaje_resumen(candidatos, total_universo, total_señales, es_previo=False):
    candidatos_ordenados = sorted(candidatos, key=lambda c: c["n_min_dias"])[:5]
    lineas_candidatos = []
    for c in candidatos_ordenados:
        score_texto = f"{c['score_pct']:.0f}%" if c["score_pct"] is not None else "SIN HISTORIAL"
        lineas_candidatos.append(f"  {c['ticker']} — proyecta cruce en ~{c['n_min_dias']}d — SCORE {score_texto}")

    fecha_texto = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    titulo = "GOLDEN CROSS PREVIEW (2h antes del cierre)" if es_previo else "GOLDEN CROSS DAILY"

    resumen = (
        f"{titulo}\n"
        f"Universo: `{total_universo} stocks`\n"
        f"Cruces confirmados hoy: `{total_señales}`\n"
        f"Candidatos vigilando: `{len(candidatos)}`\n"
    )
    if lineas_candidatos:
        resumen += "TOP CERCANOS:\n" + "\n".join(lineas_candidatos) + "\n"
    resumen += fecha_texto
    return resumen


def armar_mensaje_trade(s):
    link_tradingview = f"https://www.tradingview.com/symbols/{s['ticker']}/"
    hold_texto = f"{s['hold_estimado_dias']} días" if s['hold_estimado_dias'] is not None else "N/A"
    return (
        f"TRADE - ${s['ticker']}\n"
        f"NAME - {s['nombre_corto']}\n"
        f"INDUSTRY - {s['industria']}\n"
        f"LONG - ${s['precio']:.2f}\n"
        f"% EST - {s['objetivo_pct']:.1f}%\n"
        f"HOLD - {hold_texto}\n"
        f"LINK - 🔗 {link_tradingview}"
    )


def enviar_telegram(texto):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "parse_mode": "Markdown"})
    return r.status_code == 200


def correr(solo_resumen=False):
    conn = get_connection()
    universo = obtener_universo_filtrado(conn)
    print(f"Universo filtrado: {len(universo)} tickers. Evaluando {'(modo preview)' if solo_resumen else ''} señales de hoy...")

    señales = []
    candidatos = []
    for i, (ticker, market_cap, industria, nombre) in enumerate(universo):
        señal, candidato = evaluar_ticker_hoy(conn, ticker, market_cap, industria, nombre)
        if señal:
            señales.append(señal)
            print(f"  ✅ TRADE confirmado: {ticker}")
        if candidato:
            candidatos.append(candidato)
        if i % 500 == 0:
            print(f"  [{i}/{len(universo)}] procesados...")

    conn.close()

    print(f"\nTotal de TRADEs (cruce confirmado) hoy: {len(señales)}")
    print(f"Total de candidatos cercanos (informativo): {len(candidatos)}")

    resumen = armar_mensaje_resumen(candidatos, len(universo), len(señales), es_previo=solo_resumen)
    ok = enviar_telegram(resumen)
    print("  ✅ Resumen enviado a Telegram" if ok else "  ⚠️ Falló el envío del resumen")

    if not solo_resumen:
        for s in señales:
            texto = armar_mensaje_trade(s)
            ok = enviar_telegram(texto)
            print(f"  ✅ TRADE {s['ticker']} enviado" if ok else f"  ⚠️ Falló el envío de {s['ticker']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-resumen", action="store_true",
                         help="Solo manda el resumen/preview, sin disparar TRADEs (para la corrida 2h antes del cron principal)")
    args = parser.parse_args()
    correr(solo_resumen=args.solo_resumen)