"""
calcular_emas.py — Fase 6
Calcula EMA50/EMA200 Y SMA50/SMA200 (A.4: se corren ambas en paralelo,
no hay que elegir de antemano) para cada ticker, guarda en ema_historico,
y detecta cruces dorado/muerte (A.2: ambas direcciones desde el día uno).

Corre: python calcular_emas.py
"""

import pandas as pd
from config import get_connection


def calcular_para_ticker(conn, ticker):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < 200:
        return None  # no hay suficiente historia todavía para EMA200/SMA200

    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["sma50"] = df["cierre"].rolling(window=50).mean()
    df["sma200"] = df["cierre"].rolling(window=200).mean()

    return df


def guardar_ema_historico(conn, ticker, df):
    filas = [
        (ticker, row.fecha, row.ema50, row.ema200, row.sma50, row.sma200)
        for row in df.itertuples()
        if pd.notna(row.ema50) and pd.notna(row.sma50)
    ]
    conn.executemany("""
        INSERT OR REPLACE INTO ema_historico (ticker, fecha, ema50, ema200, sma50, sma200)
        VALUES (?, ?, ?, ?, ?, ?)
    """, filas)


def detectar_cruce(df, col_rapida, col_lenta):
    """
    Revisa las últimas 2 filas: si la rápida cruzó de abajo hacia arriba
    de la lenta -> 'dorado'; de arriba hacia abajo -> 'muerte'; si no, None.
    Regresa (tipo, fecha) o (None, None).
    """
    ultimas = df.dropna(subset=[col_rapida, col_lenta]).tail(2)
    if len(ultimas) < 2:
        return None, None

    ayer, hoy = ultimas.iloc[0], ultimas.iloc[1]

    cruzo_arriba = ayer[col_rapida] <= ayer[col_lenta] and hoy[col_rapida] > hoy[col_lenta]
    cruzo_abajo = ayer[col_rapida] >= ayer[col_lenta] and hoy[col_rapida] < hoy[col_lenta]

    if cruzo_arriba:
        return "dorado", hoy["fecha"]
    if cruzo_abajo:
        return "muerte", hoy["fecha"]
    return None, None


def registrar_cruce(conn, ticker, fecha, tipo, promedio_tipo, metodo="confirmado"):
    conn.execute("""
        INSERT OR IGNORE INTO cruces_detectados
            (ticker, fecha_cruce, tipo, metodo, promedio_tipo, notificado)
        VALUES (?, ?, ?, ?, ?, 0)
    """, (ticker, fecha, tipo, metodo, promedio_tipo))


def procesar_todos(solo_ticker=None):
    conn = get_connection()
    cur = conn.cursor()

    if solo_ticker:
        tickers = [solo_ticker]
    else:
        tickers = [r["ticker"] for r in cur.execute(
            "SELECT ticker FROM tickers WHERE activo = 1"
        ).fetchall()]

    print(f"Procesando {len(tickers)} tickers...")
    cruces_encontrados = []

    for i, ticker in enumerate(tickers):
        df = calcular_para_ticker(conn, ticker)
        if df is None:
            continue

        guardar_ema_historico(conn, ticker, df)

        tipo_ema, fecha_ema = detectar_cruce(df, "ema50", "ema200")
        if tipo_ema:
            registrar_cruce(conn, ticker, fecha_ema, tipo_ema, "EMA")
            cruces_encontrados.append((ticker, "EMA", tipo_ema, fecha_ema))

        tipo_sma, fecha_sma = detectar_cruce(df, "sma50", "sma200")
        if tipo_sma:
            registrar_cruce(conn, ticker, fecha_sma, tipo_sma, "SMA")
            cruces_encontrados.append((ticker, "SMA", tipo_sma, fecha_sma))

        if i % 200 == 0:
            conn.commit()
            print(f"  [{i}/{len(tickers)}] procesados...")

    conn.commit()
    conn.close()

    print(f"✅ Listo. Cruces nuevos detectados: {len(cruces_encontrados)}")
    for c in cruces_encontrados:
        print("  ", c)

    return cruces_encontrados


if __name__ == "__main__":
    procesar_todos()
