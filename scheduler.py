"""
scheduler.py

Corre como PROCESO PERSISTENTE (no como cron job de Railway) — la lógica
de horarios vive en el código, no en la configuración de Railway. Evita
necesitar un segundo servicio/volumen en Railway solo para el resumen
previo.

Dispara, una sola vez por día hábil (lunes a viernes):
  - 19:30 UTC: solo el resumen/preview (sin TRADEs, solo candidatos)
  - 21:30 UTC: la corrida completa (descarga precios + evalúa + TRADEs reales)

⚠️ IMPLICACIÓN DE COSTO: al ser un proceso persistente (no un job que se
apaga entre corridas), Railway cobra por el tiempo que el contenedor está
VIVO, no solo por los minutos que tarda en ejecutar. Esto consume el
crédito más rápido que el modelo de Cron Schedule anterior. Revisa tu
uso en Railway después de un par de días para confirmar que el costo
sigue siendo razonable.

Corre: python scheduler.py
(pensado como Start Command PERMANENTE en Railway — y sin Cron Schedule
configurado ahí, porque este script YA hace ese trabajo internamente)
"""

import time
from datetime import datetime, timezone

import descargar_precios
import generar_senales_diarias

HORA_RESUMEN = (19, 30)   # UTC
HORA_COMPLETA = (21, 30)  # UTC
DIAS_HABILES = {0, 1, 2, 3, 4}  # lunes=0 ... viernes=4
SEGUNDOS_ENTRE_CHECKS = 30


def ya_disparo_hoy(ultimo_registro, ahora, hora_objetivo):
    """Evita disparar 2 veces si el loop revisa varias veces dentro del mismo minuto objetivo."""
    if ultimo_registro is None:
        return False
    return (ultimo_registro.date() == ahora.date()
            and ultimo_registro.hour == hora_objetivo[0]
            and ultimo_registro.minute == hora_objetivo[1])


def correr_resumen():
    print(f"[{datetime.now(timezone.utc)}] Disparando RESUMEN (preview, 19:30 UTC)...")
    generar_senales_diarias.correr(solo_resumen=True)


def correr_completa():
    print(f"[{datetime.now(timezone.utc)}] Disparando CORRIDA COMPLETA (21:30 UTC)...")
    descargar_precios.descargar_acciones_hoy()
    generar_senales_diarias.correr(solo_resumen=False)


def debe_disparar(ahora, hora_objetivo, ultimo_registro):
    es_dia_habil = ahora.weekday() in DIAS_HABILES
    es_la_hora = (ahora.hour, ahora.minute) == hora_objetivo
    return es_dia_habil and es_la_hora and not ya_disparo_hoy(ultimo_registro, ahora, hora_objetivo)


def loop():
    print("Scheduler iniciado. Resumen: 19:30 UTC | Completa: 21:30 UTC | Lun-Vie")
    ultimo_resumen = None
    ultima_completa = None

    while True:
        ahora = datetime.now(timezone.utc)

        if debe_disparar(ahora, HORA_RESUMEN, ultimo_resumen):
            try:
                correr_resumen()
            except Exception as e:
                print(f"⚠️ Error en resumen: {e}")
            ultimo_resumen = ahora

        if debe_disparar(ahora, HORA_COMPLETA, ultima_completa):
            try:
                correr_completa()
            except Exception as e:
                print(f"⚠️ Error en corrida completa: {e}")
            ultima_completa = ahora

        time.sleep(SEGUNDOS_ENTRE_CHECKS)


if __name__ == "__main__":
    loop()
