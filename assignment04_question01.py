import networkx as nx
import netrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ER graph, 64 nodes, 10% probability of edge creation
G = nx.erdos_renyi_graph(64, 0.1, seed=42)

#  kuramoto dynamics 
time_series_list = []
for seed in range(20):
    sim = netrd.dynamics.Kuramoto()
    TS = sim.simulate(G, L=1000)  # shape: (64, 1000)
    time_series_list.append(TS)

# set up reconstructr
reconstructor = netrd.reconstruction.MutualInformationMatrix()

# git average degree of the original graph
avg_k = np.mean([d for _, d in G.degree()])

# reconstruct each time series into a thresholded network (for part c)
# threshold by degree, matching the average degree of g
reconstructed_thresholded = []
for TS in time_series_list:
    R = reconstructor.fit(TS, threshold_type='degree', avg_k=avg_k)
    reconstructed_thresholded.append(R)

# d(g, r) dissimilarity between true graph g and reconstructed graph R
# 0 is best, 1 is bad
def D(G, R):
    A_G = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    A_R = nx.to_numpy_array(R, nodelist=sorted(R.nodes()))
    # binarize 
    A_G_binary = (A_G != 0).astype(int)
    A_R_binary = (A_R != 0).astype(int)
    np.fill_diagonal(A_G_binary, 0)
    np.fill_diagonal(A_R_binary, 0)
    y_true = A_G_binary.flatten()
    y_pred = A_R_binary.flatten()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return 1 - f1

# part d compute D(G, Ri) for reconstructed networks
scores = [D(G, R) for R in reconstructed_thresholded]

plt.figure()
plt.bar(range(20), scores, color='dimgray')
plt.xlabel('Reconstruction index i')
plt.ylabel('D(G, Rᵢ)')
plt.title('Dissimilarity between G and each reconstructed network')
plt.savefig('Q1_dissimilarity_barplot.png', dpi=150, bbox_inches='tight')
print("part d done")

# part e pairwise dissimilarity matrix of 20 reconstructed networks
d = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        d[i, j] = D(reconstructed_thresholded[i], reconstructed_thresholded[j])

plt.figure()
plt.imshow(d, cmap='gray')
plt.colorbar(label='D(Ri, Rj)')
plt.xlabel('Reconstruction index j')
plt.ylabel('Reconstruction index i')
plt.title('Pairwise dissimilarity between reconstructed networks')
plt.savefig('Q1_distance_matrix.png', dpi=150, bbox_inches='tight')
# visualize the last time series
plt.figure()
plt.imshow(TS, aspect='auto', cmap='gray')
plt.xlabel('Time Steps')
plt.ylabel('Nodes')
plt.title('Kuramoto Time Series for Erdos Renyi Graph')
plt.colorbar(label='Node State')
plt.savefig('Q1_timeseries.png', dpi=150, bbox_inches='tight')
print("done")

