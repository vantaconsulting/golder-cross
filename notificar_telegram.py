"""
notificar_telegram.py — Fase 8
Revisa cruces_detectados donde notificado=0, arma el mensaje y lo manda
por Telegram, diferenciando confirmado/anticipado, dorado/muerte, EMA/SMA.

Corre: python notificar_telegram.py
"""

import requests
from config import get_connection, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

EMOJI_TIPO = {"dorado": "🟢", "muerte": "🔴"}
EMOJI_METODO = {"confirmado": "✅", "anticipado": "🔮"}


def armar_mensaje(cruce):
    emoji_tipo = EMOJI_TIPO.get(cruce["tipo"], "")
    emoji_metodo = EMOJI_METODO.get(cruce["metodo"], "")
    nombre_tipo = "Golden Cross" if cruce["tipo"] == "dorado" else "Death Cross"

    texto = (
        f"{emoji_tipo} *{nombre_tipo}* {emoji_metodo}\n"
        f"Ticker: `{cruce['ticker']}`\n"
        f"Método: {cruce['metodo'].capitalize()} ({cruce['promedio_tipo']})\n"
        f"Fecha: {cruce['fecha_cruce']}"
    )
    return texto


def enviar_mensaje(texto):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": texto,
        "parse_mode": "Markdown",
    })
    return r.status_code == 200


def procesar_pendientes():
    conn = get_connection()
    pendientes = conn.execute("""
        SELECT id, ticker, fecha_cruce, tipo, metodo, promedio_tipo
        FROM cruces_detectados
        WHERE notificado = 0
        ORDER BY fecha_cruce
    """).fetchall()

    enviados = 0
    for cruce in pendientes:
        texto = armar_mensaje(dict(cruce))
        if enviar_mensaje(texto):
            conn.execute("UPDATE cruces_detectados SET notificado = 1 WHERE id = ?", (cruce["id"],))
            enviados += 1

    conn.commit()
    conn.close()
    print(f"✅ Notificaciones enviadas: {enviados}/{len(pendientes)}")
    return enviados


if __name__ == "__main__":
    procesar_pendientes()
