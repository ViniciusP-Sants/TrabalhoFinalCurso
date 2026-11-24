import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, kruskal

from antenna_model import calc_antenna_metrics, evaluate_antenna_fitness
from optimizers import run_ga, run_pso, run_gwo

# Configuração de estilo acadêmico para os gráficos (padrão IEEE / Monografia)
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.autolayout": True,
})

N_RUNS = 30
MAX_ITER = 100

def main():
    print(f"Iniciando bateria experimental: {N_RUNS} execuções independentes por algoritmo...")

    # 1. Avaliação do Baseline Teórico Inicial (Tabela VI: L=32, W=38, Xf=10)
    base_params = [32.0, 38.0, 10.0]
    base_metrics = calc_antenna_metrics(base_params)
    base_fit = evaluate_antenna_fitness(base_params)[0]

    records = []
    curves = {"GA": [], "PSO": [], "GWO": []}
    best_overall = {
        "GA": {"fit": float("inf"), "params": None},
        "PSO": {"fit": float("inf"), "params": None},
        "GWO": {"fit": float("inf"), "params": None},
    }

    algos = [
        ("GA", run_ga),
        ("PSO", run_pso),
        ("GWO", run_gwo),
    ]

    for run_idx in range(1, N_RUNS + 1):
        seed = 100 + run_idx
        for name, func in algos:
            t0 = time.perf_counter()
            # No PSO, usamos inércia ligeiramente maior para refletir o comportamento clássico da Tabela VII
            if name == "PSO":
                params, fit, curve = func(seed=seed, pop_size=30, max_iter=MAX_ITER)
            elif name == "GA":
                params, fit, curve = func(seed=seed, pop_size=30, max_iter=MAX_ITER, cxpb=0.75, mutpb=0.15)
            else:
                params, fit, curve = func(seed=seed, pop_size=30, max_iter=MAX_ITER)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            m = calc_antenna_metrics(params)
            curves[name].append(curve)

            if fit < best_overall[name]["fit"]:
                best_overall[name]["fit"] = fit
                best_overall[name]["params"] = params

            records.append({
                "Run": run_idx,
                "Algoritmo": name,
                "L_mm": m["L_mm"],
                "W_mm": m["W_mm"],
                "Xf_mm": m["Xf_mm"],
                "fr_GHz": m["f_r_GHz"],
                "S11_dB": m["S11_24GHz_dB"],
                "VSWR": m["VSWR_24GHz"],
                "Gain_dBi": m["Gain_24GHz_dBi"],
                "BW_MHz": m["Bandwidth_MHz"],
                "Fitness": fit,
                "Tempo_ms": elapsed_ms,
            })

    df = pd.DataFrame(records)
    df.to_csv("resultados_30_execucoes.csv", index=False)

    # ==========================================================================
    # TABELA 1: Melhores Parâmetros Otimizados e Desempenho Eletromagnético
    # ==========================================================================
    print("\n" + "=" * 118)
    print("TABELA 1: COMPARAÇÃO DOS PARÂMETROS GEOMÉTRICOS E DESEMPENHO ELETROMAGNÉTICO (MELHOR SOLUÇÃO / MEDIANA)")
    print("=" * 118)
    print(f"{'Modelo':<18} | {'L (mm)':<7} | {'W (mm)':<7} | {'Xf (mm)':<7} | {'fr (GHz)':<8} | {'S11 (dB)':<9} | {'VSWR':<6} | {'Ganho (dBi)':<11} | {'BW (MHz)':<8} | {'Fitness'}")
    print("-" * 118)

    # Linha Baseline
    print(
        f"{'Baseline Inicial':<18} | {base_metrics['L_mm']:<7.2f} | {base_metrics['W_mm']:<7.2f} | "
        f"{base_metrics['Xf_mm']:<7.2f} | {base_metrics['f_r_GHz']:<8.4f} | {base_metrics['S11_24GHz_dB']:<9.2f} | "
        f"{base_metrics['VSWR_24GHz']:<6.2f} | {base_metrics['Gain_24GHz_dBi']:<11.2f} | "
        f"{base_metrics['Bandwidth_MHz']:<8.1f} | {base_fit:.5f}"
    )

    summary_rows = []
    for name in ["PSO", "GA", "GWO"]:
        sub = df[df["Algoritmo"] == name]
        print(
            f"{name + ' (Média 30x)':<18} | {sub['L_mm'].mean():<7.2f} | {sub['W_mm'].mean():<7.2f} | "
            f"{sub['Xf_mm'].mean():<7.2f} | {sub['fr_GHz'].mean():<8.4f} | {sub['S11_dB'].mean():<9.2f} | "
            f"{sub['VSWR'].mean():<6.2f} | {sub['Gain_dBi'].mean():<11.2f} | "
            f"{sub['BW_MHz'].mean():<8.1f} | {sub['Fitness'].mean():.5f}"
        )
        summary_rows.append({
            "Algoritmo": name,
            "L_mean": sub["L_mm"].mean(),
            "W_mean": sub["W_mm"].mean(),
            "Xf_mean": sub["Xf_mm"].mean(),
            "fr_mean": sub["fr_GHz"].mean(),
            "S11_mean": sub["S11_dB"].mean(),
            "S11_std": sub["S11_dB"].std(),
            "VSWR_mean": sub["VSWR"].mean(),
            "Gain_mean": sub["Gain_dBi"].mean(),
            "BW_mean": sub["BW_MHz"].mean(),
            "Fitness_median": sub["Fitness"].median(),
            "Fitness_mean": sub["Fitness"].mean(),
            "Fitness_std": sub["Fitness"].std(),
            "Tempo_ms_mean": sub["Tempo_ms"].mean(),
        })

    pd.DataFrame(summary_rows).to_csv("tabela_resumo_estatistico.csv", index=False)

    # Extrai os vetores das 30 execuções de cada algoritmo primeiro
    fit_pso = df[df["Algoritmo"] == "PSO"]["Fitness"].values
    fit_ga = df[df["Algoritmo"] == "GA"]["Fitness"].values
    fit_gwo = df[df["Algoritmo"] == "GWO"]["Fitness"].values

    # ==========================================================================
    # TESTE GLOBAL MULTIGRUPO (Friedman e Kruskal-Wallis)
    # ==========================================================================
    stat_friedman, p_friedman = friedmanchisquare(fit_pso, fit_ga, fit_gwo)
    stat_kruskal, p_kruskal = kruskal(fit_pso, fit_ga, fit_gwo)
    print("\n" + "=" * 95)
    print("TESTE GLOBAL MULTIGRUPO (GA vs. PSO vs. GWO - 30 EXECUÇÕES)")
    print("=" * 95)
    print(f"Teste de Friedman (Pareado por População Inicial) : Q = {stat_friedman:.4f} | p-valor = {p_friedman:.4e}")
    print(f"Teste de Kruskal-Wallis (Amostras Independentes)  : H = {stat_kruskal:.4f} | p-valor = {p_kruskal:.4e}")

    # ==========================================================================
    # TABELA 2: Teste Pareado de Wilcoxon (Post-Hoc com Bonferroni)
    # ==========================================================================
    print("\n" + "=" * 95)
    print("TABELA 2: TESTE ESTATÍSTICO DE WILCOXON (30 EXECUÇÕES PAREADAS - FITNESS)")
    print("=" * 95)

    comparacoes = [
        ("GWO vs. PSO", fit_gwo, fit_pso),
        ("GWO vs. GA", fit_gwo, fit_ga),
        ("PSO vs. GA", fit_pso, fit_ga),
    ]
    alpha_bonf = 0.05 / 3.0  # Correção de Bonferroni (0.0167)
    print(f"{'Comparação Pareada':<22} | {'Estatística W':<15} | {'p-valor':<18} | {'Conclusão (Bonferroni = 0.0167)'}")
    print("-" * 95)
    for label, a, b in comparacoes:
        stat, p_val = wilcoxon(a, b, alternative="less")
        concl = "Significativo (p < 0.0167)" if p_val < alpha_bonf else "Não significativo"
        print(f"{label:<22} | {stat:<15.1f} | {p_val:<18.4e} | {concl}")

    # ==========================================================================
    # GERAÇÃO DOS GRÁFICOS (300 DPI)
    # ==========================================================================
    iters = np.arange(1, MAX_ITER + 1)

    # FIGURA 1: Curva de Convergência Média (100 Iterações)
    plt.figure(figsize=(8, 5))
    colors = {"PSO": "#1f77b4", "GA": "#ff7f0e", "GWO": "#d62728"}
    styles = {"PSO": "--", "GA": "-.", "GWO": "-"}
    for name in ["PSO", "GA", "GWO"]:
        mean_curve = np.mean(curves[name], axis=0)
        plt.plot(iters, mean_curve, label=f"{name} (Média de {N_RUNS} execuções)", color=colors[name], linestyle=styles[name], linewidth=2.2)
    plt.xlabel("Iterações / Gerações")
    plt.ylabel("Valor da Função Custo (Fitness)")
    plt.title("Curva de Convergência Comparativa: GA, PSO e GWO (2.4 GHz)")
    plt.legend()
    plt.savefig("fig1_convergencia.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Dados das curvas eletromagnéticas usando os parâmetros médios de cada algoritmo
    freqs = np.linspace(2.0, 2.8, 801)
    m_base = calc_antenna_metrics(base_params, freqs)
    m_algos = {}
    for name in ["PSO", "GA", "GWO"]:
        sub = df[df["Algoritmo"] == name]
        mean_p = [sub["L_mm"].mean(), sub["W_mm"].mean(), sub["Xf_mm"].mean()]
        m_algos[name] = calc_antenna_metrics(mean_p, freqs)

    # FIGURA 2: Perda de Retorno (S11 vs Frequência)
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, m_base["S11_curve_dB"], label="Baseline Inicial (Não Otimizado)", color="gray", linestyle=":", linewidth=1.8)
    for name in ["PSO", "GA", "GWO"]:
        s11_min = m_algos[name]["S11_24GHz_dB"]
        plt.plot(freqs, m_algos[name]["S11_curve_dB"], label=f"{name} ($S_{{11}}$ = {s11_min:.2f} dB @ 2.4 GHz)", color=colors[name], linestyle=styles[name], linewidth=2.2)
    plt.axhline(-10.0, color="black", linestyle="--", linewidth=1.0, label="Limiar de Casamento (-10 dB)")
    plt.axvline(2.4, color="green", linestyle=":", linewidth=1.2, label="Frequência Alvo (2.40 GHz)")
    plt.xlabel("Frequência (GHz)")
    plt.ylabel("Perda de Retorno $S_{11}$ (dB)")
    plt.title("Perda de Retorno ($S_{11}$) vs. Frequência para a Antena Patch em 2.4 GHz")
    plt.ylim(-26, 0)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.savefig("fig2_s11_comparativo.png", dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 3: VSWR vs Frequência
    plt.figure(figsize=(8, 5))
    for name in ["PSO", "GA", "GWO"]:
        vswr_val = m_algos[name]["VSWR_24GHz"]
        plt.plot(freqs, m_algos[name]["VSWR_curve"], label=f"{name} (VSWR = {vswr_val:.2f} @ 2.4 GHz)", color=colors[name], linestyle=styles[name], linewidth=2.2)
    plt.axhline(2.0, color="black", linestyle="--", linewidth=1.0, label="Limite Operacional (VSWR = 2.0)")
    plt.axvline(2.4, color="green", linestyle=":", linewidth=1.2, label="Frequência Alvo (2.40 GHz)")
    plt.xlabel("Frequência (GHz)")
    plt.ylabel("VSWR")
    plt.title("Razão de Onda Estacionária (VSWR) vs. Frequência")
    plt.ylim(1.0, 3.5)
    plt.xlim(2.1, 2.7)
    plt.legend(loc="upper right")
    plt.savefig("fig3_vswr_comparativo.png", dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 4: Boxplot de Estabilidade (30 Execuções)
    plt.figure(figsize=(7.5, 5))
    data_box = [fit_pso, fit_ga, fit_gwo]
    bp = plt.boxplot(data_box, tick_labels=["PSO (DEAP)", "GA (DEAP)", "GWO (Proposto)"], patch_artist=True, showmeans=True)
    box_colors = ["#a6c8e0", "#ffcb99", "#f4a6a6"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
    plt.ylabel("Valor Final da Função Custo (Fitness)")
    plt.title(f"Distribuição e Estabilidade de Convergência ({N_RUNS} Execuções Independentes)")
    plt.savefig("fig4_boxplot_fitness.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSucesso! Arquivos CSV e as 4 figuras (300 DPI) foram salvos na pasta do projeto.")

if __name__ == "__main__":
    main()