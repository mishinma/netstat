import os

import pandas as pd
import numpy as np

# Nodes: 7115 Edges: 103689
num_nodes = 7115
num_edges = 103689

dirname = os.path.expanduser("~/docs/netstat/data")
fname = os.path.join(dirname, "wiki-Vote.txt")
fname2 = os.path.join(dirname, "wiki-Vote_clean.txt")

df = pd.read_csv(fname, delim_whitespace=True, names=['FromNodeId', 'ToNodeId'],
                 dtype={'FromNodeId': np.uint64, 'ToNodeId': np.uint64}, skiprows=4)

df = df.sort_values(by=['FromNodeId', 'ToNodeId'])
unique_ids = np.sort(pd.unique(df.values.ravel()))
df = df.replace(to_replace=unique_ids, value=range(num_nodes))

with open(fname, 'r') as f:
    with open(fname2, 'w') as f2:
        for _ in range(4):
            line = f.readline()
            f2.write(line)

with open(fname2, 'a') as f2:
    df.to_csv(f2, header=False, index=False)