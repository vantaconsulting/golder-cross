"""
ema_analisis_variables.py

Analisis exploratorio del sistema EMA50/EMA200 (golden cross), con el
mismo tratamiento riguroso que le dimos a los otros 4 sistemas.

Detecta TODOS los cruces dorados (EMA50 cruza arriba de EMA200) de tus
5 anios de historia, y para cada uno calcula 15 variables, probandolas
una por una.

IMPORTANTE: aqui NO se aplica el filtro de drawdown que ya tiene tu
sistema en produccion -- se calcula como UNA VARIABLE MAS, para poder
evaluar si ese filtro realmente ayuda o si hay mejores.

Universo: Market Cap >= $300M (igual que produccion).

Corre: python ema_analisis_variables.py
"""

import math
import numpy as np
import pandas as pd
from config import get_connection

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
MIN_HISTORIA_DIAS = 400
UMBRAL_SALTO_SOSPECHOSO = 0.50
RSI_PERIODO = 14
MIN_N_CONFIABLE = 30

HORIZONTES = [30, 60, 90, 120, 180]
HORIZONTE_PRINCIPAL = "retorno_90d"


def calcular_rsi(precios, periodo=14):
    delta = precios.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    avg_g = ganancia.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()
    avg_p = perdida.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()
    rs = avg_g / avg_p.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def encontrar_cruces(df):
    diff = df["ema50"] - df["ema200"]
    signo = np.sign(diff)
    cambio = signo.diff()
    cruces = []
    for i in range(1, len(df)):
        if pd.isna(cambio.iloc[i]):
            continue
        if cambio.iloc[i] == 2:
            cruces.append((i, "dorado"))
        elif cambio.iloc[i] == -2:
            cruces.append((i, "muerte"))
    return cruces


def obtener_universo_base(conn):
    filas = conn.execute("""
        SELECT ticker, market_cap, industria FROM tickers
        WHERE activo = 1 AND tipo = 'stock'
    """).fetchall()
    resultado = []
    for r in filas:
        if r["market_cap"] is None or r["market_cap"] < MARKET_CAP_MINIMO:
            continue
        if r["industria"] in INDUSTRIAS_EXCLUIDAS:
            continue
        resultado.append((r["ticker"], r["market_cap"]))
    return resultado


def obtener_serie_regimen(conn, ticker_mercado):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker_mercado,)
    )
    if len(df) < 200:
        return None
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha")
    df["sma200"] = df["cierre"].rolling(window=200, min_periods=200).mean()
    df["alcista"] = df["cierre"] > df["sma200"]
    return df


def procesar_ticker(ticker, market_cap, conn, series_regimen):
    df = pd.read_sql_query(
        "SELECT fecha, cierre FROM precios_diarios WHERE ticker = ? ORDER BY fecha",
        conn, params=(ticker,)
    )
    if len(df) < MIN_HISTORIA_DIAS:
        return []

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ema50"] = df["cierre"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["cierre"].ewm(span=200, adjust=False).mean()
    df["sma200"] = df["cierre"].rolling(window=200, min_periods=200).mean()
    df["rsi14"] = calcular_rsi(df["cierre"], RSI_PERIODO)
    df["cambio_pct"] = df["cierre"].pct_change().abs()
    df["salto_sospechoso"] = df["cambio_pct"] > UMBRAL_SALTO_SOSPECHOSO
    df["max_52w"] = df["cierre"].rolling(window=252, min_periods=100).max()
    df["min_52w"] = df["cierre"].rolling(window=252, min_periods=100).min()
    df["retorno_diario"] = df["cierre"].pct_change()
    df["vol_20"] = df["retorno_diario"].rolling(20, min_periods=15).std()
    df["vol_60"] = df["retorno_diario"].rolling(60, min_periods=40).std()

    cruces = encontrar_cruces(df)
    resultados = []

    for j, (idx_c, tipo_c) in enumerate(cruces):
        if tipo_c != "dorado":
            continue
        if idx_c < 220 or idx_c >= len(df) - 1:
            continue
        if bool(df["salto_sospechoso"].iloc[idx_c]):
            continue

        precio_c = df["cierre"].iloc[idx_c]
        if precio_c <= 0:
            continue
        fecha_c = df["fecha"].iloc[idx_c]

        fila = {"ticker": ticker, "market_cap": market_cap, "fecha_cruce": fecha_c, "precio": precio_c}

        max52 = df["max_52w"].iloc[idx_c]
        fila["drawdown_52w_%"] = round((precio_c / max52 - 1) * 100, 2) if pd.notna(max52) and max52 > 0 else None
        min52 = df["min_52w"].iloc[idx_c]
        fila["distancia_min52w_%"] = round((precio_c / min52 - 1) * 100, 2) if pd.notna(min52) and min52 > 0 else None

        rsi = df["rsi14"].iloc[idx_c]
        fila["rsi"] = round(rsi, 1) if pd.notna(rsi) else None

        sma200_c = df["sma200"].iloc[idx_c]
        fila["distancia_sma200_%"] = round((precio_c / sma200_c - 1) * 100, 2) if pd.notna(sma200_c) and sma200_c > 0 else None

        ema50_c, ema200_c = df["ema50"].iloc[idx_c], df["ema200"].iloc[idx_c]
        fila["fuerza_cruce_%"] = round((ema50_c / ema200_c - 1) * 100, 3) if ema200_c > 0 else None

        if idx_c >= 20:
            ema200_antes = df["ema200"].iloc[idx_c - 20]
            ema50_antes = df["ema50"].iloc[idx_c - 20]
            fila["pendiente_ema200_%"] = round((ema200_c / ema200_antes - 1) * 100, 3) if ema200_antes > 0 else None
            fila["pendiente_ema50_%"] = round((ema50_c / ema50_antes - 1) * 100, 3) if ema50_antes > 0 else None
        else:
            fila["pendiente_ema200_%"] = None
            fila["pendiente_ema50_%"] = None

        idx_death = None
        for k in range(j - 1, -1, -1):
            if cruces[k][1] == "muerte":
                idx_death = cruces[k][0]
                break
        if idx_death is not None:
            fila["dias_desde_death_cross"] = (fecha_c - df["fecha"].iloc[idx_death]).days
            precio_death = df["cierre"].iloc[idx_death]
            tramo = df["cierre"].iloc[idx_death:idx_c + 1]
            fila["caida_desde_death_%"] = round((precio_death - tramo.min()) / precio_death * 100, 2) if precio_death > 0 else None
        else:
            fila["dias_desde_death_cross"] = None
            fila["caida_desde_death_%"] = None

        v20, v60 = df["vol_20"].iloc[idx_c], df["vol_60"].iloc[idx_c]
        fila["ratio_volatilidad"] = round(v20 / v60, 3) if pd.notna(v20) and pd.notna(v60) and v60 > 0 else None

        if idx_c >= 60:
            precio_hace_60 = df["cierre"].iloc[idx_c - 60]
            fila["momentum_60d_%"] = round((precio_c / precio_hace_60 - 1) * 100, 2) if precio_hace_60 > 0 else None
        else:
            fila["momentum_60d_%"] = None

        fecha_solo = fecha_c.date()
        for tm, serie in series_regimen.items():
            if serie is None:
                fila[f"{tm}_alcista"] = None
                continue
            try:
                recorte = serie.loc[:pd.Timestamp(fecha_solo)]
                fm = recorte.iloc[-1] if len(recorte) > 0 else None
            except Exception:
                fm = None
            fila[f"{tm}_alcista"] = bool(fm["alcista"]) if fm is not None and pd.notna(fm.get("alcista")) else None

        for h in HORIZONTES:
            idx_h = idx_c + h
            if idx_h <= len(df) - 1:
                precio_h = df["cierre"].iloc[idx_h]
                fila[f"retorno_{h}d"] = round((precio_h - precio_c) / precio_c * 100, 2)
            else:
                fila[f"retorno_{h}d"] = None

        resultados.append(fila)

    return resultados


def wilson_ci(exitos, n):
    if n == 0:
        return (None, None)
    z = 1.959963985
    p = exitos / n
    denom = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / denom
    margen = (z * math.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denom
    return (round((centro - margen) * 100, 1), round((centro + margen) * 100, 1))


def stats_de_serie(serie):
    n = len(serie)
    if n == 0:
        return {"n": 0}
    ganadoras = serie[serie > 0]
    win_rate = len(ganadoras) / n * 100
    ci_lo, ci_hi = wilson_ci(len(ganadoras), n)
    return {
        "n": n, "win_%": round(win_rate, 1),
        "CI95": f"[{ci_lo}, {ci_hi}]" if ci_lo is not None else None,
        "mediana_%": round(serie.median(), 2),
    }


def imprimir_bucket(df, columna, buckets, titulo):
    print("\n" + "-" * 120)
    print(titulo)
    print("-" * 120)
    sub_total = df.dropna(subset=[columna, HORIZONTE_PRINCIPAL])
    filas = []
    for lo, hi, nombre in buckets:
        sub = sub_total[(sub_total[columna] >= lo) & (sub_total[columna] < hi)]
        if len(sub) == 0:
            continue
        s = stats_de_serie(sub[HORIZONTE_PRINCIPAL])
        s["bucket"] = nombre + (" ⚠️" if s["n"] < MIN_N_CONFIABLE else "")
        filas.append(s)
    if filas:
        print(pd.DataFrame(filas)[["bucket", "n", "win_%", "CI95", "mediana_%"]].set_index("bucket").to_string())


def imprimir_bucket_booleano(df, columna, titulo):
    print("\n" + "-" * 120)
    print(titulo)
    print("-" * 120)
    for val, nombre in [(True, "SI"), (False, "NO")]:
        sub = df[df[columna] == val].dropna(subset=[HORIZONTE_PRINCIPAL])
        s = stats_de_serie(sub[HORIZONTE_PRINCIPAL])
        marca = " ⚠️" if s.get("n", 0) < MIN_N_CONFIABLE else ""
        print(f"  {nombre}: N={s.get('n')}{marca}  win={s.get('win_%')}%  mediana={s.get('mediana_%')}%")


def correr():
    conn = get_connection()
    universo = obtener_universo_base(conn)
    series_regimen = {tm: obtener_serie_regimen(conn, tm) for tm in ["SPY", "QQQ"]}
    print(f"Universo base (>=$300M): {len(universo)} tickers. Buscando cruces EMA50/EMA200...")

    todos = []
    for i, (ticker, market_cap) in enumerate(universo):
        todos.extend(procesar_ticker(ticker, market_cap, conn, series_regimen))
        if i % 300 == 0:
            print(f"  [{i}/{len(universo)}] procesados...")
    conn.close()

    if not todos:
        print("No hay resultados.")
        return

    df = pd.DataFrame(todos)
    df.to_csv("eventos_ema_analisis.csv", index=False)
    print(f"\nDataset guardado: eventos_ema_analisis.csv ({len(df)} cruces, {len(df.columns)} columnas)")

    print("\n" + "=" * 130)
    print("BASELINE (todos los cruces dorados, SIN ningun filtro)")
    print("=" * 130)
    for h in HORIZONTES:
        s = stats_de_serie(df[f"retorno_{h}d"].dropna())
        print(f"  {h}d: {s}")

    imprimir_bucket(df, "market_cap", [
        (300_000_000, 1_000_000_000, "$300M-$1B"), (1_000_000_000, 5_000_000_000, "$1B-$5B"),
        (5_000_000_000, 20_000_000_000, "$5B-$20B"), (20_000_000_000, float("inf"), ">$20B"),
    ], "A) MARKET CAP")

    imprimir_bucket(df, "drawdown_52w_%", [
        (-100000, -50, "50%+ abajo"), (-50, -30, "30-50% abajo"), (-30, -20, "20-30% abajo"),
        (-20, -10, "10-20% abajo"), (-10, -5, "5-10% abajo"), (-5, 0.01, "0-5% abajo"),
    ], "B) DRAWDOWN 52 SEMANAS (el filtro actual de produccion: -20% a -50%)")

    imprimir_bucket(df, "rsi", [
        (0, 40, "RSI<40"), (40, 50, "RSI 40-50"), (50, 60, "RSI 50-60"),
        (60, 70, "RSI 60-70"), (70, 100, "RSI>70"),
    ], "C) RSI AL MOMENTO DEL CRUCE")

    imprimir_bucket(df, "distancia_sma200_%", [
        (-100000, -10, "10%+ abajo del SMA200"), (-10, 0, "0-10% abajo"),
        (0, 10, "0-10% arriba"), (10, 100000, "10%+ arriba"),
    ], "D) DISTANCIA DEL PRECIO AL SMA200")

    imprimir_bucket(df, "fuerza_cruce_%", [
        (-100000, 0.1, "<0.1% (cruce apenas)"), (0.1, 0.3, "0.1-0.3%"),
        (0.3, 0.6, "0.3-0.6%"), (0.6, 100000, ">0.6% (cruce fuerte)"),
    ], "E) FUERZA DEL CRUCE (separacion EMA50 vs EMA200)")

    imprimir_bucket(df, "pendiente_ema200_%", [
        (-100000, -0.5, "EMA200 bajando fuerte"), (-0.5, 0, "EMA200 casi plano, bajando"),
        (0, 0.5, "EMA200 casi plano, subiendo"), (0.5, 100000, "EMA200 subiendo fuerte"),
    ], "F) PENDIENTE DEL EMA200")

    imprimir_bucket(df, "pendiente_ema50_%", [
        (-100000, 0, "EMA50 bajando"), (0, 2, "EMA50 subiendo poco"),
        (2, 5, "EMA50 subiendo medio"), (5, 100000, "EMA50 subiendo fuerte"),
    ], "G) PENDIENTE DEL EMA50")

    imprimir_bucket(df, "dias_desde_death_cross", [
        (0, 60, "0-60 dias"), (60, 120, "60-120 dias"), (120, 250, "120-250 dias"),
        (250, 500, "250-500 dias"), (500, 100000, "500+ dias"),
    ], "H) DIAS DESDE EL ULTIMO DEATH CROSS")

    imprimir_bucket(df, "caida_desde_death_%", [
        (0, 15, "0-15%"), (15, 30, "15-30%"), (30, 50, "30-50%"), (50, 100000, "50%+"),
    ], "I) CAIDA MAXIMA DESDE EL DEATH CROSS")

    imprimir_bucket(df, "distancia_min52w_%", [
        (0, 20, "0-20% arriba del minimo"), (20, 50, "20-50% arriba"),
        (50, 100, "50-100% arriba"), (100, 100000, "100%+ arriba"),
    ], "K) DISTANCIA AL MINIMO DE 52 SEMANAS")

    imprimir_bucket(df, "ratio_volatilidad", [
        (0, 0.7, "Contrajo fuerte"), (0.7, 1.0, "Contrajo algo"),
        (1.0, 1.5, "Normal/expandio"), (1.5, 100000, "Muy caotico"),
    ], "L) CONTRACCION DE VOLATILIDAD")

    imprimir_bucket(df, "momentum_60d_%", [
        (-100000, 0, "Venia bajando"), (0, 15, "Subio 0-15%"),
        (15, 40, "Subio 15-40%"), (40, 100000, "Subio 40%+"),
    ], "M) MOMENTUM 60 DIAS ANTES DEL CRUCE")

    imprimir_bucket_booleano(df, "SPY_alcista", "N) REGIMEN SPY ALCISTA")
    imprimir_bucket_booleano(df, "QQQ_alcista", "O) REGIMEN QQQ ALCISTA")

    print("\nLectura rapida:")
    print("- Busca las variables donde la diferencia entre el mejor y peor bucket sea MAS GRANDE,")
    print("  con buena muestra (sin advertencia) -- esas son las que valdria la pena combinar.")
    print("- Fijate especialmente en la seccion B: te dice si el filtro de drawdown que YA tienes")
    print("  en produccion (-20% a -50%) es realmente el mejor rango, o si hay uno mejor.")


if __name__ == "__main__":
    correr()
