import sys
import os
import pandas as pd

current_dir = os.path.dirname(__file__)

import random
import unittest
import numpy as np

import index_hnsw_py as ih
import time

import faiss
import numpy as np

d = 128
n_data = 32000
n_query = 1
normalized = 0
M = 32
# np.random.seed(124)
def normalize(x):
    if normalized:
        return x / np.linalg.norm(x)
    else:
        return x
database_vectors = [normalize(np.random.rand(d)) for _ in range(n_data)]
query_vector = [normalize(np.random.rand(d)) for _ in range(n_query)]

# ----------------------------------------------
# build
# ----------------------------------------------

seed = random.randint(0,100)
np.random.seed(seed)

# Manil HNSW build graph
index_hnsw_manul = ih.IndexHNSW(d,M=32, metric=ih.MetricType.COSINE_SIMILARITY, search_metric = ih.MetricType.INNER_PRODUCT)
index_hnsw_manul.efConstruction = 50
index_hnsw_manul.add(database_vectors)