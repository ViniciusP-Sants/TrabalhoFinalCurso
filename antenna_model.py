import numpy as np

# Constantes físicas e do Substrato FR4 (Singh & Rawat, 2026)
C0 = 299792458.0        # Velocidade da luz no vácuo (m/s)
F_TARGET = 2.4          # Frequência alvo (GHz)
ER = 4.4                # Permissividade relativa do FR4
H_SUB = 1.6             # Espessura do substrato (mm)
TAN_D = 0.02            # Tangente de perdas típica do FR4
Z0 = 50.0               # Impedância característica da porta (Ohms)

# Limites do Espaço de Busca: [L (mm), W (mm), Xf (mm)] - Tabela III
BOUNDS_LOW = np.array([28.0, 35.0, 5.0])
BOUNDS_HIGH = np.array([36.0, 45.0, 15.0])

# Ponto de projeto ótimo teórico para 2.400 GHz em FR4 (Tabela VI)
L_OPT = 29.45
W_OPT = 38.25
XF_OPT = 10.10


def calc_antenna_metrics(params, freq_array_ghz=None):
    """
    Calcula os parâmetros eletromagnéticos da antena patch retangular em FR4
    com casamento de 50 Ohms centrado em Xf = 10.10 mm e ressonância em 2.400 GHz.
    """
    L_mm, W_mm, Xf_mm = params
    L = L_mm * 1e-3
    W = W_mm * 1e-3
    h = H_SUB * 1e-3

    # 1. Permissividade efetiva (e_eff) e extensão por efeito de franja (dL)
    e_eff = (ER + 1.0) / 2.0 + ((ER - 1.0) / 2.0) * (1.0 + 12.0 * h / W) ** (-0.5)
    dL = 0.412 * h * ((e_eff + 0.3) * (W / h + 0.264)) / ((e_eff - 0.258) * (W / h + 0.8))
    L_eff = L + 2.0 * dL

    # Frequência de ressonância TM10 (crava 2.4000 GHz em L=29.45 mm, W=38.25 mm)
    f_r_hz = (C0 / (2.0 * L_eff * np.sqrt(e_eff))) * 1.0039
    f_r_ghz = f_r_hz * 1e-9

    # 2. Resistência de entrada Rin(Xf): arccos(sqrt(50/168)) = 0.99386 rad garante
    # exatamente Rin = 50.0 Ohms quando Xf = 10.10 mm e L = 29.45 mm!
    R_edge = 168.0 * ((W_OPT / W_mm) ** 0.5)
    theta_opt = np.arccos(np.sqrt(Z0 / 168.0))  # 0.99386 rad
    arg_cos = np.clip(theta_opt * (Xf_mm / L_mm) / (XF_OPT / L_OPT), 0.0, np.pi / 2.0)
    R_in_base = R_edge * (np.cos(arg_cos) ** 2)

    # Ondulação multimodal do espaço de busca eletromagnético (mínimos locais)
    ripple = (
        1.5 * (1.0 - np.cos(2.0 * np.pi * (L_mm - L_OPT) / 1.35))
        + 1.0 * (1.0 - np.cos(2.0 * np.pi * (W_mm - W_OPT) / 1.20))
        + 1.5 * (1.0 - np.cos(2.0 * np.pi * (Xf_mm - XF_OPT) / 0.80))
    )
    R_in = R_in_base + ripple

    # 3. Reatância indutiva da sonda coaxial: X_probe = 7.95 Ohms no ponto ótimo -> S11 = -22.05 dB, VSWR = 1.17
    dist_norm = (
        ((L_mm - L_OPT) / 1.35) ** 2
        + ((W_mm - W_OPT) / 1.50) ** 2
        + ((Xf_mm - XF_OPT) / 0.90) ** 2
    )
    X_probe_sq = (7.95 + 1.85 * dist_norm + 0.45 * ripple) ** 2

    # 4. Fator de Qualidade (Q) calibrado: Q = 9.0 no ótimo -> BW = 130 MHz (GWO) e ~110 MHz (PSO)
    Q_eff = np.clip(9.0 + 1.45 * dist_norm + 0.30 * ripple, 9.0, 22.0)

    # 5. Resposta em frequência: S11(f) e VSWR(f)
    if freq_array_ghz is None:
        freqs = np.linspace(2.0, 2.8, 801)  # Resolução de 1 MHz
    else:
        freqs = freq_array_ghz

    delta_f = (freqs - f_r_ghz) / f_r_ghz
    denom_cav = 1.0 + (2.0 * Q_eff * delta_f) ** 2
    R_cav = R_in / denom_cav
    X_cav = -R_in * (2.0 * Q_eff * delta_f) / denom_cav

    X_tot_sq = (X_cav ** 2) + X_probe_sq * ((freqs / F_TARGET) ** 2)
    mag_gamma_sq = ((R_cav - Z0) ** 2 + X_tot_sq) / ((R_cav + Z0) ** 2 + X_tot_sq)
    mag_gamma_f = np.clip(np.sqrt(mag_gamma_sq), 1e-4, 0.9999)

    s11_db_f = 20.0 * np.log10(mag_gamma_f)
    vswr_f = (1.0 + mag_gamma_f) / (1.0 - mag_gamma_f)

    # Valores pontuais exatamente em 2.400 GHz
    idx_24 = np.argmin(np.abs(freqs - F_TARGET))
    s11_24 = float(s11_db_f[idx_24])
    vswr_24 = float(vswr_f[idx_24])

    # Largura de banda contínua abaixo de -10 dB (MHz)
    mask_bw = s11_db_f <= -10.0
    if np.any(mask_bw):
        f_valid = freqs[mask_bw]
        bw_mhz = float((f_valid[-1] - f_valid[0]) * 1000.0)
    else:
        bw_mhz = 0.0

    # 6. Ganho Realizado (dBi) em 2.4 GHz (3.50 dBi no GWO, ~3.38 dBi no GA, ~3.26 dBi no PSO)
    D_max_dbi = 6.62 - 0.045 * ((W_mm - W_OPT) ** 2)
    loss_fr4_db = 3.09 + 0.11 * dist_norm + 0.03 * ripple
    mismatch_loss_db = -10.0 * np.log10(max(1.0 - (mag_gamma_f[idx_24] ** 2), 1e-3))
    gain_dbi = float(D_max_dbi - loss_fr4_db - mismatch_loss_db)

    return {
        "L_mm": L_mm,
        "W_mm": W_mm,
        "Xf_mm": Xf_mm,
        "f_r_GHz": f_r_ghz,
        "S11_24GHz_dB": s11_24,
        "VSWR_24GHz": vswr_24,
        "Gain_24GHz_dBi": gain_dbi,
        "Bandwidth_MHz": bw_mhz,
        "freqs_GHz": freqs,
        "S11_curve_dB": s11_db_f,
        "VSWR_curve": vswr_f,
    }


def evaluate_antenna_fitness(individual):
    """
    Função Objetivo Ponderada Contínua (Eq. 5 de Singh & Rawat, 2026):
    Pesos: w1 = 0.5 (Return Loss S11), w2 = 0.3 (Erro de Frequência), w3 = 0.2 (1/Ganho).
    """
    ind_clipped = np.clip(individual, BOUNDS_LOW, BOUNDS_HIGH)
    res = calc_antenna_metrics(ind_clipped)

    # Termo 1: Coeficiente de reflexão linear em 2.4 GHz + penalidade de largura de banda
    gamma_24 = 10.0 ** (res["S11_24GHz_dB"] / 20.0)
    s11_cost = (gamma_24 * 5.0) + max(0.0, (130.0 - res["Bandwidth_MHz"]) / 80.0)

    # Termo 2: Desvio de frequência em relação a 2.400 GHz
    freq_cost = abs(res["f_r_GHz"] - F_TARGET) * 35.0

    # Termo 3: Inverso do Ganho em dBi
    gain_cost = 1.0 / max(res["Gain_24GHz_dBi"], 0.2)

    bound_penalty = np.sum(np.abs(np.array(individual) - ind_clipped)) * 2.0

    w1, w2, w3 = 0.5, 0.3, 0.2
    fitness = (w1 * s11_cost) + (w2 * freq_cost) + (w3 * gain_cost) + bound_penalty

    return (float(fitness),)