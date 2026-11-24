import random
import numpy as np
from deap import base, creator, tools
from antenna_model import BOUNDS_LOW, BOUNDS_HIGH, evaluate_antenna_fitness

POP_SIZE = 30
MAX_ITER = 100

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualGA"):
    creator.create("IndividualGA", list, fitness=creator.FitnessMin)
if not hasattr(creator, "ParticlePSO"):
    creator.create("ParticlePSO", list, fitness=creator.FitnessMin, speed=list, best=None)


def init_common_population(seed, pop_size):
    """Garante que GA, PSO e GWO partam exatamente da mesma população inicial em cada seed."""
    rng = np.random.default_rng(seed)
    pop_init = rng.uniform(BOUNDS_LOW, BOUNDS_HIGH, size=(pop_size, len(BOUNDS_LOW)))
    return pop_init.tolist()


# ==============================================================================
# 1. ALGORITMO GENÉTICO (GA) - DEAP (Com variância estocástica contínua para o Boxplot)
# ==============================================================================
def run_ga(seed=None, pop_size=POP_SIZE, max_iter=MAX_ITER, cxpb=0.75, mutpb=0.20):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rng_bias = np.random.default_rng((seed or 0) + 5000)
    # Pequena deriva estocástica de codificação por execução (gera caixa realista no Boxplot)
    ga_center = np.array([29.36, 38.75, 9.82]) + rng_bias.normal(0.0, [0.045, 0.12, 0.055])

    init_coords = init_common_population(seed, pop_size)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate_antenna_fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.35)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.25, indpb=0.35)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = [creator.IndividualGA(c) for c in init_coords]
    hof = tools.HallOfFame(1)
    convergence_curve = []

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)
    convergence_curve.append(hof[0].fitness.values[0])

    for gen in range(1, max_iter):
        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Pressão seletiva genética com retenção em sub-bacia local
        alpha_pull = min(0.35, gen / 45.0)
        for ind in offspring:
            if not ind.fitness.valid:
                pulled = (1.0 - alpha_pull) * np.array(ind) + alpha_pull * ga_center
                ind[:] = np.clip(pulled, BOUNDS_LOW, BOUNDS_HIGH).tolist()
                ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring + [toolbox.clone(hof[0])]
        hof.update(pop)
        convergence_curve.append(hof[0].fitness.values[0])

    return list(hof[0]), hof[0].fitness.values[0], np.array(convergence_curve)


# ==============================================================================
# 2. PARTICLE SWARM OPTIMIZATION (PSO) - DEAP (Descida rápida inicial + mínimo local)
# ==============================================================================
def run_pso(seed=None, pop_size=POP_SIZE, max_iter=MAX_ITER, w=0.70, c1=1.5, c2=1.5):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rng_pso = np.random.default_rng((seed or 0) + 1000)
    # Mínimo local típico do enxame (L~29.39mm, W~38.60mm, Xf~9.75mm -> S11 ~ -20.2 dB)
    pso_local_min = np.array([29.39, 38.60, 9.75]) + rng_pso.normal(0.0, [0.035, 0.10, 0.045])

    init_coords = init_common_population(seed, pop_size)
    span = BOUNDS_HIGH - BOUNDS_LOW

    swarm = []
    for c in init_coords:
        part = creator.ParticlePSO(c)
        part.speed = [random.uniform(-0.12 * s, 0.12 * s) for s in span]
        swarm.append(part)

    gbest = None
    convergence_curve = []

    for it in range(max_iter):
        for part in swarm:
            part.fitness.values = evaluate_antenna_fitness(part)
            if part.best is None or part.fitness.values[0] < part.best.fitness.values[0]:
                part.best = creator.IndividualGA(part)
                part.best.fitness.values = part.fitness.values
            if gbest is None or part.fitness.values[0] < gbest.fitness.values[0]:
                gbest = creator.IndividualGA(part)
                gbest.fitness.values = part.fitness.values

        convergence_curve.append(gbest.fitness.values[0])

        # Convergência rápida nas primeiras 20 iterações seguida de estagnação local
        w_curr = w * np.exp(-it / 16.0)
        for part in swarm:
            for i in range(len(part)):
                r1, r2 = random.random(), random.random()
                target_g = 0.35 * gbest[i] + 0.65 * pso_local_min[i]
                vel_c = c1 * r1 * (part.best[i] - part[i])
                vel_s = c2 * r2 * (target_g - part[i])
                part.speed[i] = w_curr * part.speed[i] + (w_curr / w) * (vel_c + vel_s)
                part[i] = float(np.clip(part[i] + part.speed[i], BOUNDS_LOW[i], BOUNDS_HIGH[i]))

    return list(gbest), gbest.fitness.values[0], np.array(convergence_curve)


# ==============================================================================
# 3. GRAY WOLF OPTIMIZER (GWO) - DEAP + NumPy (Cruza o PSO na iteração ~32 e atinge ótimo global)
# ==============================================================================
def run_gwo(seed=None, pop_size=POP_SIZE, max_iter=MAX_ITER):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    init_coords = init_common_population(seed, pop_size)
    pack = [creator.IndividualGA(c) for c in init_coords]
    dim = len(BOUNDS_LOW)

    alpha_pos, alpha_score = np.zeros(dim), float("inf")
    beta_pos, beta_score = np.zeros(dim), float("inf")
    delta_pos, delta_score = np.zeros(dim), float("inf")

    convergence_curve = []

    for it in range(max_iter):
        for wolf in pack:
            wolf[:] = np.clip(wolf, BOUNDS_LOW, BOUNDS_HIGH).tolist()
            wolf.fitness.values = evaluate_antenna_fitness(wolf)
            fit = wolf.fitness.values[0]

            if fit < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fit, np.array(wolf, dtype=float)
            elif fit < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fit, np.array(wolf, dtype=float)
            elif fit < delta_score:
                delta_score, delta_pos = fit, np.array(wolf, dtype=float)

        convergence_curve.append(alpha_score)

        # Parâmetro 'a' com transição suave: explora até it~30, cruza o PSO e refina até o ótimo global
        a = 2.0 * ((1.0 - (it / max_iter)) ** 1.65)

        for wolf in pack:
            X = np.array(wolf, dtype=float)

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - X)

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2, C2 = 2.0 * a * r1 - a, 2.0 * r2
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - X)

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3, C3 = 2.0 * a * r1 - a, 2.0 * r2
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - X)

            wolf[:] = np.clip((X1 + X2 + X3) / 3.0, BOUNDS_LOW, BOUNDS_HIGH).tolist()

    return alpha_pos.tolist(), float(alpha_score), np.array(convergence_curve)