"""
main.py — producción, un solo comando para el ciclo diario completo

1. Descarga los precios de HOY (acciones NASDAQ+NYSE)
2. Corre el detector de señales con el criterio validado en el prospecto
   (universo filtrado por market cap/industria, entrada anticipada con
   score >= 70%, objetivo adaptativo por ticker) y manda el mensaje a
   Telegram — todo dentro de generar_senales_diarias.py

Si algo falla a la mitad, se intenta avisar por Telegram en vez de
quedarse callado.

Corre: python main.py
Pensado para correrse vía cron (Railway) después del cierre de mercado
de EEUU (~18:00-19:00 hora del este, ajustado a UTC).

Requiere haber corrido ANTES, al menos una vez:
  - crear_tablas.py, obtener_universo.py, backfill_historico.py (historial)
  - obtener_fundamentales.py (market cap / industria, para el filtro)
"""

import sys
import traceback
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, check_env
import descargar_precios
import generar_senales_diarias


def avisar_error(mensaje):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"⚠️ Error en el sistema Golden Cross:\n{mensaje[:500]}"
        })
    except Exception:
        pass  # si hasta el aviso de error falla, ya no hay mucho más que hacer


def run():
    if not check_env():
        print("❌ Revisa tu archivo .env (o las variables de entorno en Railway) antes de correr main.py")
        sys.exit(1)

    try:
        print("1/2 — Descargando precios de hoy...")
        descargar_precios.descargar_acciones_hoy()

        print("2/2 — Evaluando señales del día (criterio validado) y notificando por Telegram...")
        generar_senales_diarias.correr()

        print("✅ Corrida completa.")

    except Exception as e:
        error_texto = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print("❌", error_texto)
        avisar_error(error_texto)
        sys.exit(1)


if __name__ == "__main__":
    run()