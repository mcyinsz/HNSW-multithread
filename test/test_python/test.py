import index_hnsw_py as ih
import numpy as np
import time

# build index
np.random.seed(42)
d = 128
index = ih.IndexHNSW(d=d,M=24, metric=ih.MetricType.INNER_PRODUCT)
index.efConstruction = 24

data = np.random.rand(131072, d)
print(data.shape)

start = time.time()
index.add(data)
end = time.time()
print(f"total adding time {(end - start):.2f}")

query = np.random.rand(3, d)
distances, labels = index.search(query, k=5, param_efSearch = 2048)

print(f"共有 {len(distances)} 个查询结果")
for i, (dists, ids) in enumerate(zip(distances, labels)):
    print(f"第{i}个查询结果:")
    print("距离:", dists)
    print("标签:", ids)