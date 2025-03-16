import index_hnsw_py as ih
import numpy as np
import time

# 创建索引
np.random.seed(42)
d = 128
index = ih.IndexHNSW(d=d,M=24, metric=ih.MetricType.INNER_PRODUCT)
index.efConstruction = 24
# 添加数据

data = np.random.rand(131072, d)  # 100个d维向量
print(data.shape)

start = time.time()
# print(f"current time {time.time()}")
index.add(data)
end = time.time()
# print(f"end add {end}")
print(f"total adding time {(end - start):.2f}")

# 执行搜索
query = np.random.rand(3, d)  # 3个查询
distances, labels = index.search(query, k=5, param_efSearch = 2048)

# 输出结果
print(f"共有 {len(distances)} 个查询结果")
for i, (dists, ids) in enumerate(zip(distances, labels)):
    print(f"第{i}个查询结果:")
    print("距离:", dists)
    print("标签:", ids)