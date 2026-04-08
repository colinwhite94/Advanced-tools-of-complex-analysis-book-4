import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# a: implement a bloom filter for edges in an undirected graph
class BloomFilter:
    def __init__(self, n, epsilon):
        # m = number of bits, k = number of hash functions
        self.m = int(-n * np.log(epsilon) / (np.log(2) ** 2))
        self.k = int((self.m / n) * np.log(2))
        self.bits = np.zeros(self.m, dtype=bool)
    
    def _hashes(self, edge):
        # generate k hash indices for an edge (i, j)
        # normalize edge so (i,j) and (j,i) hash identically
        edge = tuple(sorted(edge))
        indices = []
        for seed in range(self.k):
            h = hash((edge, seed)) % self.m
            indices.append(h)
        return indices
    
    def add(self, edge):
        for h in self._hashes(edge):
            self.bits[h] = True
    
    def check(self, edge):
        # t = "maybe exists", f = "definitely does not exist"
        return all(self.bits[h] for h in self._hashes(edge))

# er graph with 1000 nodes and 5% edge probability
G = nx.erdos_renyi_graph(1000, 0.05, seed=42)
edges = set(G.edges())
n_edges = len(edges)
print(f"no. of edges: {n_edges}")

# all possible edges in a 1000-node undirected graph
n_nodes = 1000
all_possible_edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
n_possible = len(all_possible_edges)
print(f"no. of possible edges: {n_possible}")

# b: compare lookup times with and without bloom filter
time_direct = n_possible  # 1 second per lookup, no filter
print(f"\nDirect lookup time: {time_direct:,} seconds ({time_direct/3600:.1f} hours)")

# only edges that pass the filter need a lookup
for epsilon in [0.01, 0.05, 0.1]:
    # create bloom filter and add edges
    bf = BloomFilter(n=n_edges, epsilon=epsilon)
    for edge in edges:
        bf.add(edge)
    
    # only edges that pass the filter require a full lookup
    filter_positives = sum(1 for e in all_possible_edges if bf.check(e))
    time_with_filter = filter_positives
    speedup = time_direct / time_with_filter
    
    print(f"\nε={epsilon}:")
    print(f"filter positives (true + false): {filter_positives:,}")
    print(f"lookup time with filter: {time_with_filter:,} seconds ({time_with_filter/3600:.1f} hours)")


# part c — measure actual FPR as filter is overfilled beyond designed capacity
non_edges = [(i, j) for i, j in all_possible_edges if (i, j) not in edges]
test_sample = non_edges[:10000]  # 10k non-edges
edge_list = list(edges) 
multipliers = np.linspace(1, 3, 20)  # 1x to 3x capacity

plt.figure()

for epsilon in [0.01, 0.05, 0.1]:
    actual_fpr = []
    n_elements = []
    
    for mult in multipliers:
        bf = BloomFilter(n=n_edges, epsilon=epsilon)
        n_to_add = int(mult * n_edges)
        n_elements.append(n_to_add)
        
        # add real edges first, then synthetic pairs to simulate overfilling
        for edge in edge_list:
            bf.add(edge)
        
        added = n_edges
        i = 0
        while added < n_to_add:
            synthetic = (i, i + 1000)
            if synthetic not in edges:
                bf.add(synthetic)
                added += 1
            i += 1
        
        # measure fpr on held-out non-edges
        fp = sum(1 for e in test_sample if bf.check(e))
        actual_fpr.append(fp / len(test_sample))
    
    plt.plot(n_elements, actual_fpr, label=f'ε={epsilon}',
             color={0.01: '0.1', 0.05: '0.5', 0.1: '0.8'}[epsilon])

plt.axvline(x=n_edges, color='black', linestyle='--', linewidth=0.8, label='actual number of edges')
plt.xlabel('Number of elements added')
plt.ylabel('Actual false positive rate')
plt.title('Bloom Filter degradation as capacity is exceeded')
plt.legend()
plt.savefig('bloom_degradation.png', dpi=150, bbox_inches='tight')
