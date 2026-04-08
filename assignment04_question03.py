import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# ============================================================
# NK LANDSCAPE GENERATOR (provided)
# ============================================================
def generate_nk_landscape(N=20, K=5, seed=None, apply_paper_skew=True):
    if K >= N:
        raise ValueError("K must be less than N.")
    rng = np.random.default_rng(seed)

    def bits_to_int(bits):
        x = 0
        for b in bits:
            x = (x << 1) | int(b)
        return x

    def int_to_bits(x, N):
        return tuple((x >> shift) & 1 for shift in range(N - 1, -1, -1))

    partners = []
    for i in range(N):
        others = [j for j in range(N) if j != i]
        chosen = tuple(rng.choice(others, size=K, replace=False))
        partners.append(chosen)

    tables = []
    for i in range(N):
        table = rng.random(2 ** (K + 1))
        tables.append(table)

    num_states = 2 ** N
    states = np.arange(num_states, dtype=np.uint32)[:, None]
    shifts = np.arange(N - 1, -1, -1, dtype=np.uint32)
    bit_matrix = ((states >> shifts) & 1).astype(np.uint8)

    raw_scores = np.zeros(num_states, dtype=np.float64)
    for i in range(N):
        idxs = (i,) + partners[i]
        local_bits = bit_matrix[:, idxs]
        local_powers = (1 << np.arange(K, -1, -1)).astype(np.uint32)
        table_indices = (local_bits * local_powers).sum(axis=1)
        raw_scores += tables[i][table_indices]
    raw_scores /= N

    global_max_raw = float(raw_scores.max())
    normalized_scores = raw_scores / global_max_raw

    if apply_paper_skew:
        scores = normalized_scores ** 8
    else:
        scores = normalized_scores.copy()

    argmax_state = int(raw_scores.argmax())
    global_best_bits = int_to_bits(argmax_state, N)

    def fitness_fn(bits):
        idx = bits_to_int(bits)
        return float(scores[idx])

    return {
        "fitness_fn": fitness_fn,
        "N": N, "K": K,
        "partners": partners,
        "tables": tables,
        "raw_scores": raw_scores,
        "normalized_scores": normalized_scores,
        "scores": scores,
        "global_max_raw": global_max_raw,
        "global_best_bits": global_best_bits,
    }

def step(population, graph, fitness_fn):
    """
    one synchronous step of the model
    
    simultaneously:
      checks all neighbors' fitness scores
      if any neighbor is strictly better, copies the best one (random tiebreak)
      otherwise, tries one random one-bit mutation and adopts it if it improves fitness
    
    synchronous means all decisions are based on time t states,
    and all updates are applied together to produce time t+1.
    
    parameters:
        population: list of tuples, one binary string per agent
        graph: networkx graph defining who can observe whom
        fitness_fn: callable that scores a binary string tuple
    
    returns:
        new_population: list of time t+1 states
    """
    n_agents = len(population)
    
    # compute fitness for all agents at time t
    fitnesses = [fitness_fn(population[i]) for i in range(n_agents)]
    
    new_population = []
    
    # for each agent, decide whether to copy a neighbor or try mutation
    for agent in range(n_agents):
        current_solution = population[agent]
        current_fitness = fitnesses[agent]
        
        # get this agent's neighbors and their fitnesses
        neighbors = list(graph.neighbors(agent))
        neighbor_fitnesses = [fitnesses[n] for n in neighbors]
        
        # check if any neighbor is strictly better
        best_neighbor_fitness = max(neighbor_fitnesses) if neighbors else -1
        
        if best_neighbor_fitness > current_fitness:
            # exploitation: copy the best neighbor (random tiebreak if tied)
            best_neighbors = [n for n in neighbors 
                              if fitnesses[n] == best_neighbor_fitness]
            chosen = random.choice(best_neighbors)
            new_population.append(population[chosen])
        
        else:
            # exploration: try one random one-bit mutation
            bit_to_flip = random.randint(0, len(current_solution) - 1)
            mutated = list(current_solution)
            mutated[bit_to_flip] = 1 - mutated[bit_to_flip]  # flip 0->1 or 1->0
            mutated = tuple(mutated)
            
            if fitness_fn(mutated) > current_fitness:
                new_population.append(mutated)  # adopt mutation if it improves
            else:
                new_population.append(current_solution)  # stay put
    
    return new_population

# sanity check
landscape = generate_nk_landscape(N=20, K=5, seed=0)
fitness_fn = landscape["fitness_fn"]

G = nx.path_graph(10)  # small test graph
population = [tuple(np.random.randint(0, 2, 20)) for _ in range(10)]

print("Before:", [round(fitness_fn(p), 3) for p in population])
population = step(population, G, fitness_fn)
print("After: ", [round(fitness_fn(p), 3) for p in population])

def population_metrics(population, fitness_fn):
    """
    compute three summary statistics of the current population.
    
    parameters:
        population: list of tuples, one binary string per agent
        fitness_fn: callable that scores a binary string tuple
    
    returns:
        avg_fitness: mean fitness across all agents
        best_fitness: highest fitness in the population
        n_unique: number of unique solutions in the population
    """
    fitnesses = [fitness_fn(p) for p in population]
    avg_fitness = np.mean(fitnesses)
    best_fitness = np.max(fitnesses)
    n_unique = len(set(population))
    return avg_fitness, best_fitness, n_unique

# generate one landscape and one shared initial population
landscape = generate_nk_landscape(N=20, K=5, seed=42)
fitness_fn = landscape["fitness_fn"]

# shared initial population, same for both networks
random.seed(42)
population = [tuple(np.random.randint(0, 2, 20)) for _ in range(100)]

# two networks
path_graph = nx.path_graph(100)
complete_graph = nx.complete_graph(100)

# verify identical starting conditions
avg, best, unique = population_metrics(population, fitness_fn)
print(f"t=0 avg fitness:{avg:.4f}")
print(f"t=0 best fitness:{best:.4f}")
print(f"t=0 unique solutions:{unique}")
print("Both networks start from identical state — verified.")


def run_simulation(population, graph, fitness_fn, T=100):
    """
    run model for given time steps
    records average fitness, best fitness, and number of unique
    
    
    parameters:
        population: initial list of tuples (shared starting state)
        graph: networkx (path or complete)
        fitness_fn: callable fitness function from NK landscape
        T: number of time steps to simulate
    
    returns:
        avg_fitness: list of length T+1
        best_fitness: list of length T+1
        n_unique: list of length T+1
    """
    # make a copy so we don't modify the original population
    pop = list(population)
    
    avg_fitness = []
    best_fitness = []
    n_unique = []
    
    for t in range(T + 1):
        # record metrics at time t before updating
        avg, best, unique = population_metrics(pop, fitness_fn)
        avg_fitness.append(avg)
        best_fitness.append(best)
        n_unique.append(unique)
        
        # update population (skip on last step — no need to simulate t=101)
        if t < T:
            pop = step(pop, graph, fitness_fn)
    
    return avg_fitness, best_fitness, n_unique

# part d run one simulation on each network
# use the same landscape and initial population for fair comparison
landscape = generate_nk_landscape(N=20, K=5, seed=42)
fitness_fn = landscape["fitness_fn"]

random.seed(42)
np.random.seed(42)
init_population = [tuple(np.random.randint(0, 2, 20)) for _ in range(100)]

path_graph = nx.path_graph(100)
complete_graph = nx.complete_graph(100)

# run simulations
path_avg, path_best, path_unique = run_simulation(list(init_population), path_graph, fitness_fn)

comp_avg, comp_best, comp_unique = run_simulation(list(init_population), complete_graph, fitness_fn)

time_steps = list(range(101))

# plot average fitness
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

axes[0].plot(time_steps, path_avg, color='0.2', label='path graph')
axes[0].plot(time_steps, comp_avg, color='0.6', label='complete graph')
axes[0].set_ylabel('Average fitness')
axes[0].set_title('Average fitness over time')
axes[0].legend()

# plot best fitness
axes[1].plot(time_steps, path_best, color='0.2', label='path graph')
axes[1].plot(time_steps, comp_best, color='0.6', label='complete graph')
axes[1].set_ylabel('Best fitness')
axes[1].set_title('Best fitness over time')
axes[1].legend()

# plot number of unique solutions
axes[2].plot(time_steps, path_unique, color='0.2', label='path graph')
axes[2].plot(time_steps, comp_unique, color='0.6', label='complete graph')
axes[2].set_ylabel('Unique solutions')
axes[2].set_title('Number of unique solutions over time')
axes[2].set_xlabel('Time step')
axes[2].legend()

plt.tight_layout()
plt.savefig('single_run_comparison.png', dpi=150, bbox_inches='tight')

def run_replicates(n_replicates, n_agents, N, K, T=100):
    """
    multiple independent replicates of the ABM on both networks.
    in each replicate, the path and complete graph share the same
    NK landscape and initial population where only the graph differs.
    
    Parameters:
        n_replicates: number of independent replicates
        n_agents: number of agents (100)
        N: bitstring length
        K: landscape ruggedness
        T: number of time steps
    
    Returns:
        path_avgs: array of shape (n_replicates, T+1) — avg fitness per replicate
        comp_avgs: array of shape (n_replicates, T+1) — avg fitness per replicate
        path_uniques: array of shape (n_replicates, T+1) — unique solutions per replicate
        comp_uniques: array of shape (n_replicates, T+1) — unique solutions per replicate
    """
    path_graph = nx.path_graph(n_agents)
    complete_graph = nx.complete_graph(n_agents)

    path_avgs = np.zeros((n_replicates, T + 1))
    comp_avgs = np.zeros((n_replicates, T + 1))
    path_uniques = np.zeros((n_replicates, T + 1))
    comp_uniques = np.zeros((n_replicates, T + 1))

    for r in range(n_replicates):
        if r % 10 == 0:
            print(f"  replicate {r}/{n_replicates}...")
        # new landscape and new initial population for each replicate
        # but shared between path and complete graph within the replicate
        landscape = generate_nk_landscape(N=N, K=K, seed=r)
        fitness_fn = landscape["fitness_fn"]

        # set random seeds for reproducibility (same for both networks in this replicate)
        random.seed(r)
        np.random.seed(r)
        init_pop = [tuple(np.random.randint(0, 2, N)) for _ in range(n_agents)]

        # run both networks from identical starting conditions
        p_avg, _, p_unique = run_simulation(list(init_pop), path_graph, fitness_fn, T)
        c_avg, _, c_unique = run_simulation(list(init_pop), complete_graph, fitness_fn, T)

        path_avgs[r]    = p_avg
        comp_avgs[r]    = c_avg
        path_uniques[r] = p_unique
        comp_uniques[r] = c_unique

    return path_avgs, comp_avgs, path_uniques, comp_uniques


# part e 100 replicates
path_avgs, comp_avgs, path_uniques, comp_uniques = run_replicates(
    n_replicates=100, n_agents=100, N=20, K=5, T=100
)

time_steps = np.arange(101)

# compute means and standard deviations across replicates
path_avg_mean = path_avgs.mean(axis=0)
comp_avg_mean = comp_avgs.mean(axis=0)
path_avg_std  = path_avgs.std(axis=0)
comp_avg_std  = comp_avgs.std(axis=0)

path_unique_mean = path_uniques.mean(axis=0)
comp_unique_mean = comp_uniques.mean(axis=0)
path_unique_std  = path_uniques.std(axis=0)
comp_unique_std  = comp_uniques.std(axis=0)

# figure 4 style plot, mean average fitness with std band
plt.figure(figsize=(8, 5))
plt.plot(time_steps, path_avg_mean, color='0.2', label='path graph')
plt.fill_between(time_steps,
                 path_avg_mean - path_avg_std,
                 path_avg_mean + path_avg_std,
                 color='0.2', alpha=0.2)
plt.plot(time_steps, comp_avg_mean, color='0.6', label='complete graph')
plt.fill_between(time_steps,
                 comp_avg_mean - comp_avg_std,
                 comp_avg_mean + comp_avg_std,
                 color='0.6', alpha=0.2)
plt.xlabel('Time step')
plt.ylabel('Mean average fitness')
plt.title('Mean average fitness over time (100 replicates)')
plt.legend()
plt.savefig('replicate_fitness.png', dpi=150, bbox_inches='tight')
#fitness plot saved here

# mean unique solutions plot
plt.figure(figsize=(8, 5))
plt.plot(time_steps, path_unique_mean, color='0.2', label='path graph')
plt.fill_between(time_steps,
                 path_unique_mean - path_unique_std,
                 path_unique_mean + path_unique_std,
                 color='0.2', alpha=0.2)
plt.plot(time_steps, comp_unique_mean, color='0.6', label='complete graph')
plt.fill_between(time_steps,
                 comp_unique_mean - comp_unique_std,
                 comp_unique_mean + comp_unique_std,
                 color='0.6', alpha=0.2)
plt.xlabel('Time step')
plt.ylabel('Mean unique solutions')
plt.title('Mean number of unique solutions over time (100 replicates)')
plt.legend()
plt.savefig('replicate_unique.png', dpi=150, bbox_inches='tight')
#part e done here

# part f, vary K and measure long-run performance difference
K_values = [0, 1, 5, 10]
T = 100
n_replicates = 100
n_agents = 100
N = 20

# store mean average fitness at t=100 for each K and each network
path_longrun = []  # mean avg fitness at t=100 across replicates
comp_longrun = []

#Running part f varying K
for K in K_values:
    print(f"  K={K}...")
    path_avgs_k, comp_avgs_k, _, _ = run_replicates(
        n_replicates=n_replicates, n_agents=n_agents, N=N, K=K, T=T
    )
    # long run performance = mean average fitness at t=100
    path_longrun.append(path_avgs_k[:, -1].mean())
    comp_longrun.append(comp_avgs_k[:, -1].mean())

path_longrun = np.array(path_longrun)
comp_longrun = np.array(comp_longrun)

# performance difference: complete path (positive = complete wins, negative = path wins)
difference = comp_longrun - path_longrun

print("\nLong-run performance summary:")
for i, K in enumerate(K_values):
    print(f"  K={K}: path={path_longrun[i]:.3f}, complete={comp_longrun[i]:.3f}, diff={difference[i]:.3f}")

# plot long-run performance difference vs K
plt.figure(figsize=(8, 5))
plt.plot(K_values, difference, color='0.2', marker='o')
plt.axhline(0, color='0.6', linestyle='--', linewidth=0.8)
plt.xlabel('K (landscape ruggedness)')
plt.ylabel('Long-run performance difference\n(complete graph − path graph)')
plt.title('How landscape ruggedness affects the\ncomplete vs path graph performance gap')
plt.savefig('K_comparison.png', dpi=150, bbox_inches='tight')
print("part f done")