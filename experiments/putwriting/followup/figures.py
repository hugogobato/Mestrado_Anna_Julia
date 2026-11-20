"""Plot the HAR specification comparison and close-marked portfolio paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd

from .panel import RESULTS


def plot(results):
    data = pd.read_csv(results / "har_crisis_h22.csv", parse_dates=["origin_date"])
    labels = {
        "HAR": ("HAR, Rogers–Satchell", "#377eb8"),
        "HAR-VIX": ("HAR–VIX, Rogers–Satchell + log(VIX²)", "#00857a"),
        "RS-HAR-VIX-level": ("HAR–VIX, Rogers–Satchell + VIX²", "#b83c2f"),
        "YZsmooth-HAR-VIX-level": ("HAR–VIX, Yang–Zhang suavizado + VIX²", "#ae7241"),
    }
    fig, ax = plt.subplots(figsize=(9.1, 4.6), layout="constrained")
    for model, (label, color) in labels.items():
        df = data[data.model.eq(model)]
        ax.plot(df.origin_date, df.y_hat, label=label, color=color, lw=1.7)
    ax.plot(df.origin_date, df.y, label="Yang–Zhang realizada", color="#272727", lw=1.6, linestyle="--")
    ax.set(title="Previsão H = 22 durante o choque de março de 2020",
           ylabel="Variância anualizada (escala log)", xlabel="Origem da previsão")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.legend(frameon=False)
    ax.grid(alpha=.22)
    fig.savefig(results / "har_crise_2020.png", dpi=155)
    plt.close(fig)

    curves = pd.read_parquet(results / "daily_curves.parquet")
    curves = curves.query("iv_source=='LIVE' and execution=='bid' and mark_policy=='observed_mid'")
    har = curves[curves.model.eq("HAR")].set_index("date")
    har_vix = curves[curves.model.eq("HAR-VIX")].set_index("date")
    series = [
        (har.strategy_nav, "HAR", "#377eb8"),
        (har_vix.strategy_nav, "HAR–VIX", "#00857a"),
        (har.put_nav, "Venda integral, mesmos contratos", "#a6653b"),
        (har.official_put_nav, "PUT oficial", "#292929"),
        (har.cash_nav, "Caixa", "#888888"),
    ]
    fig, (nav, dd) = plt.subplots(2, 1, sharex=True, figsize=(9.1, 6.0), layout="constrained")
    for values, label, color in series:
        nav.plot(values, label=label, color=color, lw=1.4)
        dd.plot(values / values.cummax() - 1, color=color, lw=1.1)
    nav.set(title="Put writing com sinal LIVE e execução no bid", ylabel="Patrimônio (base 1)")
    nav.legend(ncol=2, frameon=False)
    dd.set(ylabel="Drawdown nos fechamentos", xlabel="Fechamento do pregão")
    dd.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    for ax in (nav, dd):
        ax.grid(alpha=.22)
    fig.savefig(results / "patrimonio_diario_live_bid.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    plot(parser.parse_args().results)
