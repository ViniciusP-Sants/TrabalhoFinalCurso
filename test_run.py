from antenna_model import calc_antenna_metrics, evaluate_antenna_fitness
from optimizers import run_ga, run_pso, run_gwo

# 1. Avaliar Baseline Inicial (Tabela VI do artigo: L=32 mm, W=38 mm, Xf=10 mm)
baseline_params = [32.0, 38.0, 10.0]
base_res = calc_antenna_metrics(baseline_params)
base_fit = evaluate_antenna_fitness(baseline_params)[0]

# 2. Executar 1 rodada controlada (seed=42) de cada metaheurística
ga_params, ga_fit, ga_curve = run_ga(seed=42)
pso_params, pso_fit, pso_curve = run_pso(seed=42)
gwo_params, gwo_fit, gwo_curve = run_gwo(seed=42)

modelos = [
    ("Baseline (Inicial)", baseline_params, base_fit),
    ("GA (DEAP)", ga_params, ga_fit),
    ("PSO (DEAP)", pso_params, pso_fit),
    ("GWO (DEAP/NumPy)", gwo_params, gwo_fit),
]

print(f"{'Modelo':<20} | {'L (mm)':<7} | {'W (mm)':<7} | {'Xf (mm)':<7} | {'fr (GHz)':<8} | {'S11 (dB)':<9} | {'VSWR':<6} | {'Ganho (dBi)':<11} | {'BW (MHz)':<8} | {'Fitness'}")
print("-" * 115)
for nome, params, fit in modelos:
    m = calc_antenna_metrics(params)
    print(
        f"{nome:<20} | {m['L_mm']:<7.2f} | {m['W_mm']:<7.2f} | {m['Xf_mm']:<7.2f} | "
        f"{m['f_r_GHz']:<8.4f} | {m['S11_24GHz_dB']:<9.2f} | {m['VSWR_24GHz']:<6.2f} | "
        f"{m['Gain_24GHz_dBi']:<11.2f} | {m['Bandwidth_MHz']:<8.1f} | {fit:.5f}"
    )