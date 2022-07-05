import math
import os

import scipy.io as sio
import numpy as np
adjacency_z = np.load("cc_signalz.npy")
for adj in adjacency_z:
    for i in range(200):
        for j in range(200):
            #print(adj[i][j])
            if abs(adj[i][j])>15:
                print(adj[i][j])
                #adj[i][j] = 0
                #print(i)
                #print(j)
#np.save("cc_signalz.npy",np.array(adjacency_z))