import sys
import os
import pandas as pd

current_dir = os.path.dirname(__file__)

import random
import unittest
import numpy as np

import index_hnsw_py as ih
import time

# build index instances
np.random.seed(42)
d = 128
index = ih.IndexHNSW(d=d,M=24, metric=ih.MetricType.INNER_PRODUCT)
index.efConstruction = 24

import faiss
import numpy as np

d = 128
n_data = 131072
n_query = 1
normalized = 1
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


# FAISS build graph
index_hnsw = faiss.IndexHNSWFlat(d, 20)
index_hnsw.hnsw.metric_type = faiss.IndexFlatIP(d)
index_hnsw.add(np.array(database_vectors))

# Manil HNSW build graph
index_hnsw_manul = ih.IndexHNSW(d,M=20, metric=ih.MetricType.INNER_PRODUCT)
index_hnsw_manul.add(database_vectors)


# scan k's and efsearch's
k_list = [256, 512, 1024, 2048, 4096 ]
efsearch_list = []

dict_list = []
for k in k_list:
    efsearch = k

    result_dict = dict()

    index_hnsw.hnsw.efSearch = efsearch
    D_hnsw, I_hnsw = index_hnsw.search(np.array(query_vector), k)

    D_hnsw_manul, I_hnsw_manul = index_hnsw_manul.search(query_vector, k, param_efSearch = efsearch)

    # ----------------------------------------------
    # ground truch
    # ----------------------------------------------
    index_flat = faiss.IndexFlatIP(d)
    index_flat.add(np.array(database_vectors))
    D_true, I_true = index_flat.search(np.array(query_vector), k=k)

    # ----------------------------------------------
    # recall
    # ----------------------------------------------
    def calculate_recall(hnsw_results, true_results):
        recall_list = []
        for hnsw_row, true_row in zip(hnsw_results, true_results):
            def recall(list_x:list[int],list_y:list[int])->int:
                set_x = set(list_x)
                set_y = set(list_y)
                # set_cap =  set_x & set_y
                set_cap = set_x.intersection(set_y)
                # print(set_x,set_y,set_cap)
                return len(set_cap)/len(set_y)
            recall_list.append(recall(hnsw_row,true_row))
            print(recall(hnsw_row,true_row))
        return np.mean(recall_list)

    recall = calculate_recall(I_hnsw, I_true)
    recall_manul = calculate_recall(I_hnsw_manul,I_true)
    
    # print("HNSW indices:", I_hnsw)
    # print("ground truth indices:", I_true)

    print(f"FAISS recall (Recall@{k}): {recall * 100:.2f}%")
    print(f"Customized HNSW recall (Recall@{k}): {recall_manul * 100:.2f}%")
    
    result_dict["k"] = k
    result_dict["efsearch"] = efsearch
    result_dict["FAISS_recall"] = recall
    result_dict["Manul_HNSW_recall"] = recall_manul
    dict_list.append(result_dict)


df = pd.DataFrame(dict_list)
df.to_csv(os.path.join(current_dir, f"HNSW_align_dataset{n_data}_dim{d}_nor{normalized}_INNERPRODUCT.csv"))
