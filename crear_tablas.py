"""
crear_tablas.py — Fase 2
Se corre UNA sola vez. Crea las tablas de database.db según el diseño del plan.

Nota: agregué las columnas "metodo_promedio" (EMA/SMA) donde aplica,
para poder comparar ambos enfoques desde el día uno, tal como pide A.4.

Corre: python crear_tablas.py
"""

from config import get_connection

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
    ticker TEXT PRIMARY KEY,
    tipo TEXT NOT NULL CHECK (tipo IN ('stock', 'crypto')),
    exchange TEXT,
    activo INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS precios_diarios (
    ticker TEXT NOT NULL,
    fecha TEXT NOT NULL,
    cierre REAL NOT NULL,
    PRIMARY KEY (ticker, fecha)
);

CREATE TABLE IF NOT EXISTS ema_historico (
    ticker TEXT NOT NULL,
    fecha TEXT NOT NULL,
    ema50 REAL,
    ema200 REAL,
    sma50 REAL,
    sma200 REAL,
    PRIMARY KEY (ticker, fecha)
);

CREATE TABLE IF NOT EXISTS cruces_detectados (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    fecha_cruce TEXT NOT NULL,
    tipo TEXT NOT NULL CHECK (tipo IN ('dorado', 'muerte')),
    metodo TEXT NOT NULL CHECK (metodo IN ('confirmado', 'anticipado')),
    promedio_tipo TEXT NOT NULL CHECK (promedio_tipo IN ('EMA', 'SMA')),
    notificado INTEGER NOT NULL DEFAULT 0,
    UNIQUE(ticker, fecha_cruce, tipo, metodo, promedio_tipo)
);

CREATE TABLE IF NOT EXISTS backtest_resultados (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    estrategia TEXT NOT NULL CHECK (estrategia IN ('confirmado', 'anticipado')),
    direccion TEXT NOT NULL CHECK (direccion IN ('long', 'short')),
    promedio_tipo TEXT NOT NULL CHECK (promedio_tipo IN ('EMA', 'SMA')),
    estrategia_salida TEXT NOT NULL DEFAULT 'cruce_contrario',
    fecha_entrada TEXT NOT NULL,
    fecha_salida TEXT,
    dias_en_operacion INTEGER,
    retorno_pct REAL,
    retorno_anualizado_pct REAL,
    fue_falso_positivo INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_precios_ticker_fecha ON precios_diarios(ticker, fecha);
CREATE INDEX IF NOT EXISTS idx_ema_ticker_fecha ON ema_historico(ticker, fecha);
"""

if __name__ == "__main__":
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()

    tablas = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print("✅ Tablas creadas en database.db:")
    for t in tablas:
        print(" -", t["name"])
    conn.close()