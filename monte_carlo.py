"""
monte_carlo.py
Toma las operaciones YA SIMULADAS en backtest_resultados para una combinación
específica (ej. EMA / anticipado / objetivo_adaptativo_ticker) y las remuestrea
con reemplazo (bootstrap) miles de veces, en órdenes distintos, componiendo
capital trade tras trade dentro de cada trayectoria simulada.

Esto responde: "¿el retorno que vimos depende del orden particular en que
llegaron los trades históricamente, o es robusto sin importar el orden?"

IMPORTANTE — qué SÍ y qué NO mide esto:
  - SÍ mide sensibilidad al orden de los trades ya observados.
  - NO genera escenarios de mercado nuevos ni futuros — sigue atado a la
    distribución de retornos que ya ocurrió en tus 2 años de historia.
    Si el edge real depende de condiciones de mercado que no se repitan
    (ver pendiente "dependencia del edge con bull rally"), este Monte Carlo
    no lo va a detectar por sí solo.

Corre: python monte_carlo.py
       python monte_carlo.py --promedio-tipo EMA --estrategia anticipado --salida objetivo_adaptativo_ticker
       python monte_carlo.py --simulaciones 10000 --capital 5000
"""

import numpy as np
import pandas as pd
from config import get_connection

N_SIMULACIONES_DEFAULT = 5000
CAPITAL_INICIAL_DEFAULT = 1000


def cargar_retornos(conn, promedio_tipo, estrategia, estrategia_salida):
    df = pd.read_sql_query("""
        SELECT retorno_pct, fecha_entrada
        FROM backtest_resultados
        WHERE promedio_tipo = ? AND estrategia = ? AND estrategia_salida = ?
        ORDER BY fecha_entrada
    """, conn, params=(promedio_tipo, estrategia, estrategia_salida))
    df["fecha_entrada"] = pd.to_datetime(df["fecha_entrada"])
    return df


def estimar_trades_por_anio(df):
    """Usa la densidad histórica real de señales para estimar cuántas operaciones
    caben típicamente en 365 días, y así simular trayectorias de ~1 año."""
    rango_dias = (df["fecha_entrada"].max() - df["fecha_entrada"].min()).days
    if rango_dias <= 0:
        return len(df)
    return max(1, int(len(df) * 365 / rango_dias))


def simular_una_trayectoria(retornos_pct, n_trades, capital_por_trade):
    """
    Modelo SIN COMPONER: cada operación recibe capital_por_trade fresco e
    independiente (igual que el backtest original), no reinversión secuencial.
    Esto es consistente con la realidad de que las señales llegan de tickers
    distintos en paralelo, no una sola cuenta ciclando un trade a la vez.
    """
    muestra = np.random.choice(retornos_pct, size=n_trades, replace=True)
    pnl_por_trade = capital_por_trade * (muestra / 100)
    pnl_acumulado = np.cumsum(pnl_por_trade)
    capital_desplegado = capital_por_trade * np.arange(1, n_trades + 1)

    pico = np.maximum.accumulate(pnl_acumulado)
    drawdown_usd = pico - pnl_acumulado
    drawdown_pct = np.where(capital_desplegado > 0, drawdown_usd / capital_desplegado * 100, 0)
    max_drawdown_pct = drawdown_pct.max()

    ganancia_total = pnl_acumulado[-1]
    capital_total = capital_por_trade * n_trades
    retorno_total_pct = ganancia_total / capital_total * 100 if capital_total > 0 else 0
    return retorno_total_pct, max_drawdown_pct


def correr_monte_carlo(promedio_tipo, estrategia, estrategia_salida,
                        n_simulaciones, capital_por_trade, n_trades_override=None):
    conn = get_connection()
    df = cargar_retornos(conn, promedio_tipo, estrategia, estrategia_salida)
    conn.close()

    if df.empty:
        print(f"⚠️  No hay operaciones en backtest_resultados para "
              f"{promedio_tipo}/{estrategia}/{estrategia_salida}. Corre backtest.py primero.")
        return

    retornos_pct = df["retorno_pct"].values
    n_trades = n_trades_override or estimar_trades_por_anio(df)

    print(f"Combinación: {promedio_tipo} / {estrategia} / {estrategia_salida}")
    print(f"Operaciones históricas disponibles para remuestrear: {len(retornos_pct)}")
    print(f"Simulando trayectorias de {n_trades} operaciones (~1 año de densidad histórica de señales)")
    print(f"x {n_simulaciones:,} corridas, ${capital_por_trade:,.0f} por operación (sin componer)...")

    resultados_retorno = np.empty(n_simulaciones)
    resultados_drawdown = np.empty(n_simulaciones)

    for i in range(n_simulaciones):
        ret_total, dd = simular_una_trayectoria(retornos_pct, n_trades, capital_por_trade)
        resultados_retorno[i] = ret_total
        resultados_drawdown[i] = dd

    prob_perdida = (resultados_retorno < 0).mean() * 100

    print("\n" + "=" * 72)
    print(f"RESULTADOS MONTE CARLO ({n_simulaciones:,} simulaciones, ${capital_por_trade:,.0f} por operación)")
    print("=" * 72)
    print(f"{'Percentil':<12}{'Retorno total %':<20}{'Max drawdown %':<20}")
    for p in [5, 25, 50, 75, 95]:
        print(f"P{p:<11}{np.percentile(resultados_retorno, p):<20.2f}{np.percentile(resultados_drawdown, p):<20.2f}")

    peor_idx = int(np.argmin(resultados_retorno))
    mejor_idx = int(np.argmax(resultados_retorno))

    print(f"\nProbabilidad de terminar con pérdida (retorno < 0%): {prob_perdida:.1f}%")
    print(f"Peor trayectoria simulada:  {resultados_retorno[peor_idx]:>10.2f}%  (drawdown {resultados_drawdown[peor_idx]:.2f}%)")
    print(f"Mediana (P50):              {np.percentile(resultados_retorno, 50):>10.2f}%")
    print(f"Mejor trayectoria simulada: {resultados_retorno[mejor_idx]:>10.2f}%")
    print("=" * 72)
    print("\nLectura rápida:")
    print("- Si P5 (el escenario del 5% peor de los casos) sigue siendo positivo, el edge")
    print("  es robusto al orden de los trades — buena señal.")
    print("- Si P5 es muy negativo o 'probabilidad de pérdida' es alta, el resultado del")
    print("  backtest original pudo depender de tener suerte con el orden real de los trades.")
    print("- Esto NO reemplaza la comparación contra buy-and-hold (SPY) — sigue pendiente.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--promedio-tipo", default="EMA", choices=["EMA", "SMA"])
    parser.add_argument("--estrategia", default="anticipado", choices=["confirmado", "anticipado"])
    parser.add_argument("--salida", default="objetivo_adaptativo_ticker",
                         choices=["cruce_contrario", "objetivo_fijo", "objetivo_adaptativo_ticker"])
    parser.add_argument("--simulaciones", type=int, default=N_SIMULACIONES_DEFAULT)
    parser.add_argument("--capital", type=float, default=CAPITAL_INICIAL_DEFAULT)
    parser.add_argument("--trades", type=int, default=None,
                         help="Número de operaciones por trayectoria simulada (default: estima ~1 año de densidad histórica)")
    args = parser.parse_args()

    correr_monte_carlo(
        promedio_tipo=args.promedio_tipo,
        estrategia=args.estrategia,
        estrategia_salida=args.salida,
        n_simulaciones=args.simulaciones,
        capital_por_trade=args.capital,
        n_trades_override=args.trades,
    )