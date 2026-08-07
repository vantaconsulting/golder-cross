"""
config.py
Configuración centralizada: carga las variables de entorno y expone
una función para conectarse a la base de datos SQLite.
Todos los demás scripts importan de aquí.
"""

import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

# Carga el archivo .env (debe estar en la raíz del proyecto)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

DB_PATH = BASE_DIR / "database.db"

POLYGON_BASE_URL = "https://api.polygon.io"

# Las 10 criptomonedas seleccionadas (ajusta a tu gusto).
# Formato Polygon para crypto: "X:BTCUSD"
CRYPTO_TICKERS = [
    "X:BTCUSD",
    "X:ETHUSD",
    "X:SOLUSD",
    "X:BNBUSD",
    "X:XRPUSD",
    "X:ADAUSD",
    "X:DOGEUSD",
    "X:AVAXUSD",
    "X:LINKUSD",
    "X:LTCUSD",
]


def get_connection():
    """Regresa una conexión a database.db con row_factory tipo dict."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def check_env():
    """Valida que las variables de entorno mínimas existan."""
    faltantes = []
    if not POLYGON_API_KEY:
        faltantes.append("POLYGON_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        faltantes.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        faltantes.append("TELEGRAM_CHAT_ID")
    if faltantes:
        print(f"⚠️  Faltan variables de entorno en .env: {', '.join(faltantes)}")
        return False
    return True


if __name__ == "__main__":
    ok = check_env()
    print("✅ .env cargado correctamente" if ok else "❌ Revisa tu archivo .env")
