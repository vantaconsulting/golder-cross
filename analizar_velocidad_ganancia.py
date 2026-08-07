"""
analizar_velocidad_ganancia.py

R&D puro: sobre TODOS los golden cross CONFIRMADOS del universo (sin ningún
filtro, sin ninguna estrategia de salida), mide qué % alcanzan distintos
umbrales de ganancia (2%, 5%, 10%, 15%, 20%) dentro de distintas ventanas
de tiempo (20, 40, 60, 90 días) después del cruce.

Responde: "de ~8,000 cruces, ¿cuántos llegaron a subir 5%? ¿10%? ¿15%? en
los primeros 20/40/60/90 días" — sirve como referencia de qué tan realista
es cada objetivo de toma de ganancia, independiente de cualquier regla de
salida que ya hayamos probado.

Corre: python analizar_velocidad_ganancia.py
       python analizar_velocidad_ganancia.py AAPL MSFT NVDA
"""

import numpy as np
import pandas as pd
from config import get_connection

MIN_HISTORIA = 250
UMBRAL_SALTO_SOSPECHOSO = 0.50
VENTANAS_DIAS = [20, 40, 60, 90]
UMBRALES_PCT = [2, 5, 10, 15, 20]


def encontrar_cruces_dorados(df):
    diff = df["ema50"] - df["ema200"]
    signo = np.sign(diff)
    cambio = signo.diff()
    return list(df.index[cambio == 2])


def hay_salto_en_tramo(df, idx_inicio, idx_fin):
    return bool(df["salto_sospechoso"].iloc[idx_inicio:idx_fin + 1].any())


def procesar_ticker(conn, ticker):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA:
        return []

    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO

    cruces = encontrar_cruces_dorados(df)
    max_ventana = max(VENTANAS_DIAS)
    eventos = []

    for idx_c in cruces:
        idx_fin_disponible = min(idx_c + max_ventana, len(df) - 1)
        dias_disponibles = idx_fin_disponible - idx_c

        if hay_salto_en_tramo(df, idx_c, idx_fin_disponible):
            continue

        precio_entrada = df["cierre"].iloc[idx_c]
        if precio_entrada == 0:
            continue

        tramo = df["cierre"].iloc[idx_c:idx_fin_disponible + 1]
        ganancia_acumulada = ((tramo.cummax() - precio_entrada) / precio_entrada * 100).reset_index(drop=True)

        dias_hasta_umbral = {}
        for u in UMBRALES_PCT:
            alcanzados = ganancia_acumulada[ganancia_acumulada >= u]
            dias_hasta_umbral[u] = int(alcanzados.index[0]) if not alcanzados.empty else None

        eventos.append({
            "ticker": ticker,
            "dias_disponibles": dias_disponibles,
            "ganancia_acumulada": ganancia_acumulada,
            "dias_hasta_umbral": dias_hasta_umbral,
        })

    return eventos


def correr(tickers=None):
    conn = get_connection()
    if tickers is None:
        tickers = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM tickers WHERE activo = 1 AND tipo = 'stock'"
        ).fetchall()]

    print(f"Analizando velocidad de ganancia post-cruce sobre {len(tickers)} tickers...")
    todos_eventos = []
    for i, ticker in enumerate(tickers):
        eventos = procesar_ticker(conn, ticker)
        todos_eventos.extend(eventos)
        if i % 200 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(eventos)} cruces dorados")
    conn.close()

    if not todos_eventos:
        print("⚠️  No hay suficientes cruces.")
        return

    print(f"\nTotal de cruces dorados confirmados analizados: {len(todos_eventos)}")

    print("\n" + "=" * 95)
    print("% DE CRUCES QUE ALCANZARON CADA UMBRAL DE GANANCIA, DENTRO DE CADA VENTANA DE DÍAS")
    print("=" * 95)
    header = f"{'Ventana':<12}" + "".join([f">={u}%".ljust(10) for u in UMBRALES_PCT]) + "N elegibles"
    print(header)

    for ventana in VENTANAS_DIAS:
        elegibles = [e for e in todos_eventos if e["dias_disponibles"] >= ventana]
        n = len(elegibles)
        fila = f"{ventana} días{'':<6}"
        for u in UMBRALES_PCT:
            if n == 0:
                fila += f"{'N/A':<10}"
                continue
            alcanzaron = sum(1 for e in elegibles if e["ganancia_acumulada"].iloc[:ventana + 1].max() >= u)
            fila += f"{alcanzaron/n*100:.1f}%".ljust(10)
        fila += f"{n}"
        print(fila)

    print("=" * 95)
    print("\n--- DÍAS PROMEDIO/MEDIANA PARA ALCANZAR CADA UMBRAL (ventana máxima de 90 días) ---")
    total = len(todos_eventos)
    for u in UMBRALES_PCT:
        dias_lista = [e["dias_hasta_umbral"][u] for e in todos_eventos if e["dias_hasta_umbral"][u] is not None]
        if dias_lista:
            print(f">= {u}%: alcanzado por {len(dias_lista)}/{total} cruces ({len(dias_lista)/total*100:.1f}%) "
                  f"— días promedio: {np.mean(dias_lista):.1f}, mediana: {np.median(dias_lista):.0f}")
        else:
            print(f">= {u}%: ningún cruce lo alcanzó en 90 días")

    print("\nLectura rápida:")
    print("- Esto NO usa ninguna estrategia de salida ni filtro — es el comportamiento crudo del precio")
    print("  después de CUALQUIER golden cross confirmado, en todo el universo.")
    print("- Sirve de referencia de 'techo realista': si el 15% se alcanza pocas veces, un objetivo_fijo")
    print("  de 15% puede estar pidiendo demasiado; si se alcanza casi siempre y rápido, puede pedir más.")


if __name__ == "__main__":
    import sys
    correr(tickers=sys.argv[1:] if len(sys.argv) > 1 else None)
