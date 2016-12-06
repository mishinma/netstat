import os
import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

dirname = os.path.expanduser("~/docs/netstat/")
fname = os.path.join(dirname, "dist_matrix.h5")

with h5py.File(fname, 'r') as f:
    dist_matrix = f['wiki_vote'][:]

num_nodes = 7115

dist = dist_matrix[np.triu_indices(num_nodes, 1)]
dist = dist[dist > 0]
#
# n, bins, patches = plt.hist(dist, bins='auto', facecolor='green', alpha=0.75)
#
# plt.xlabel('Distance')
# plt.ylabel('Probability')
# plt.grid(True)
#
# plt.show()

median_dist = np.median(dist)
mean_dist = np.mean(dist)
diameter = np.max(dist)
eff_diameter = np.percentile(dist, 90, interpolation='linear')

print 'Median distance: {}'.format(median_dist)
print 'Mean distance: {}'.format(mean_dist)
print 'Diameter: {}'.format(diameter)
print 'Effective diameter: {}'.format(eff_diameter)
