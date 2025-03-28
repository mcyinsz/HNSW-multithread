import sys
import os

import index_hnsw_py as ih
import unittest
import numpy as np
import random

import matplotlib.pyplot as plt

import pandas as pd

def normalize(x):
    return x / np.linalg.norm(x)

def recall(list_x:list[int],list_y:list[int])->int:
    set_x = set(list_x)
    set_y = set(list_y)
    # set_cap =  set_x & set_y
    set_cap = set_x.intersection(set_y)
    # print(set_x,set_y,set_cap)
    return len(set_cap)/len(set_y)

class test_IndexHNSW:

    def __init__(self,
                 vector_dim:int,
                 dataset_size:int,
                 normalize_vector:bool,
                 data_format):
       self.vector_dim = vector_dim
       self.dataset_size = dataset_size
       self.normalize_vector = normalize_vector
       self.data_format = data_format
       self.vectors = [normalize(np.random.rand(vector_dim)) for _ in range(dataset_size)] if normalize_vector else\
              [np.random.rand(vector_dim) for _ in range(dataset_size)]
       self.vectors = np.array(self.vectors)

       self.index_hnsw = ih.IndexHNSW(d = vector_dim, metric=ih.MetricType.INNER_PRODUCT) # attention! TODO
       self.index_flat = ih.IndexFlat(d = vector_dim, metric=ih.MetricType.INNER_PRODUCT)
    
    def build(self):
        self.index_hnsw.add(self.vectors)
        self.index_flat.add(self.vectors)

    def test_recall_HNSW(self,
                         k,
                         efSearch,
                         random_seed):
        np.random.seed(random_seed)
        query = normalize(np.random.rand(self.vector_dim)) if self.normalize_vector else np.random.rand(self.vector_dim)
        query = query[np.newaxis, :]
        # HNSW搜索
        hnsw_distances, hnsw_labels = self.index_hnsw.search(query, k, efSearch)
        
        # 精确搜索
        flat_dist, flat_labels = self.index_flat.search(query, k)

        return recall(hnsw_labels[0],flat_labels[0])

def test_on_a_certain_IndexHNSW(vector_dim,
                                dataset_size,
                                normalize_vector,
                                k_list = [32,64,128,256],
                                efSearch_list = [2,4,8,16,32,64,128,256,512,1024],
                                total_test = 10,
                                data_format = np.float32):
    test_instance = test_IndexHNSW(vector_dim,
                   dataset_size,
                   normalize_vector,
                   data_format)
    test_instance.build()

    list_of_dict:list[dict] = []

    for k in k_list:
        for efSearch in efSearch_list:
            rc_list = []
            for test_num in range(total_test):
                seed = random.randint(0,100)
                rc_list.append(
                 test_instance.test_recall_HNSW(
                     k,
                     efSearch,
                     seed
                 )   
                )
            mean_rc = np.mean(rc_list)
            list_of_dict.append({"vector_dim":vector_dim,
                                 "dataset_size":dataset_size,
                                 "normalized": 1 if test_instance.normalize_vector else 0,
                                 "k":k,
                                 "efSearch":efSearch,
                                 "mean_recall": mean_rc,
                                 "data_format":data_format})
    
    df = pd.DataFrame(list_of_dict)
    df.to_csv(os.path.join(os.path.dirname(__file__),f"test_IndexHNSW_recall_dim{vector_dim}_size{dataset_size}_normal{1 if test_instance.normalize_vector else 0}.csv"))


    

if __name__ == "__main__":
    for ds in [1000,5000, 10000, 50000, 100000, 500000]:
        for dim in [128]:
            for normalize_vector in [True]:
                test_on_a_certain_IndexHNSW(vector_dim=dim,
                                            dataset_size = ds,
                                            normalize_vector=normalize_vector,
                                            k_list=[128,256,512,1024,2048],
                                            efSearch_list = [128,256,512,1024,2048,4096],
                                            data_format=np.float32
                                            )