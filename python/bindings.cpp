#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utils/Index.h"
#include "utils/constants.h"
#include "impl/IndexHNSW.h"

namespace py = pybind11;

// 通用维度校验函数
inline void check_dimensions(const py::array_t<float>& arr, int expected_dim, const char* msg) {
    if (arr.ndim() != 2 || arr.shape(1) != expected_dim) {
        throw std::runtime_error(msg);
    }
}


PYBIND11_MODULE(index_hnsw_py, m) {
    m.doc() = "HNSW index Python binding (Optimized with Zero-Copy)";

    // 暴露枚举类型
    py::enum_<MetricType>(m, "MetricType")
        .value("INNER_PRODUCT", MetricType::INNER_PRODUCT)
        .value("L2_DISTANCE", MetricType::L2_DISTANCE)
        .export_values();

    // Index基类绑定
    py::class_<Index>(m, "Index")
        .def_readwrite("d", &Index::d)
        .def_readwrite("ntotal", &Index::ntotal)
        .def_readwrite("metric_type", &Index::metric_type);

    // IndexFlat绑定
    py::class_<IndexFlat, Index>(m, "IndexFlat")
        .def(py::init<int, MetricType>(),
            py::arg("d") = 0,
            py::arg("metric") = MetricType::INNER_PRODUCT)
        .def("add", [](IndexFlat& self, py::array_t<float, py::array::c_style | py::array::forcecast> data) {
            check_dimensions(data, self.d, "Input data must have shape [n, d]");
            
            // 获取数组信息
            py::buffer_info buf = data.request();
            const float* ptr = static_cast<float*>(buf.ptr);
            const size_t n = buf.shape[0];
            
            // 零拷贝传递数据
            {
                py::gil_scoped_release release;
                self.add(n, std::vector<float>(ptr, ptr + n * self.d));
            }
        }, py::arg("data"))
        .def("search", [](IndexFlat& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                        int k,
                        int param_efSearch) {
            // 校验查询数据维度
            check_dimensions(queries, self.d, "Queries must have shape [n, d]");
            
            // 准备输出容器
            std::vector<std::vector<float>> distances;
            std::vector<std::vector<int>> labels;
            
            // 获取查询数据
            py::buffer_info buf = queries.request();
            const float* q_ptr = static_cast<float*>(buf.ptr);
            const size_t n = buf.shape[0];
            
            // 执行搜索
            {
                py::gil_scoped_release release;
                self.search(n, std::vector<float>(q_ptr, q_ptr + n * self.d), k, distances, labels, param_efSearch);
            }
            
            // 转换为NumPy数组返回
            py::array_t<float> dist_array({n, static_cast<size_t>(k)});
            py::array_t<int> labels_array({n, static_cast<size_t>(k)});
            
            auto dist_buf = dist_array.mutable_unchecked<2>();
            auto labels_buf = labels_array.mutable_unchecked<2>();
            
            for (size_t i = 0; i < n; ++i) {
                for (int j = 0; j < k; ++j) {
                    dist_buf(i, j) = distances[i][j];
                    labels_buf(i, j) = labels[i][j];
                }
            }
            
            return std::make_pair(dist_array, labels_array);
        }, py::arg("queries"), py::arg("k"), py::arg("param_efSearch") = 0);

    // IndexHNSW绑定
    py::class_<IndexHNSW, Index>(m, "IndexHNSW")
        .def_property("efConstruction",
            [](IndexHNSW& self) { return self.hnsw.efConstruction; },  // Getter
            [](IndexHNSW& self, int value) { self.hnsw.efConstruction = value; }  // Setter
        )
        .def(py::init<int, int, MetricType>(),
            py::arg("d") = 0,
            py::arg("M") = 32,
            py::arg("metric") = MetricType::INNER_PRODUCT)
        .def("add", [](IndexHNSW& self, py::array_t<float, py::array::c_style | py::array::forcecast> data) {
            check_dimensions(data, self.d, "Input data must have shape [n, d]");
            
            py::buffer_info buf = data.request();
            const float* ptr = static_cast<float*>(buf.ptr);
            const size_t n = buf.shape[0];
            
            {
                py::gil_scoped_release release;
                self.add(n, std::vector<float>(ptr, ptr + n * self.d));
            }
        }, py::arg("data"))
        .def("search", [](IndexHNSW& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                        int k,
                        int param_efSearch) {
            check_dimensions(queries, self.d, "Queries must have shape [n, d]");
            
            std::vector<std::vector<float>> distances;
            std::vector<std::vector<int>> labels;
            
            py::buffer_info buf = queries.request();
            const float* q_ptr = static_cast<float*>(buf.ptr);
            const size_t n = buf.shape[0];
            
            {
                py::gil_scoped_release release;
                self.search(n, std::vector<float>(q_ptr, q_ptr + n * self.d), k, distances, labels, param_efSearch);
            }
            
            // 优化后的结果转换
            const size_t num_results = distances.size();
            py::array_t<float> dist_array({num_results, static_cast<size_t>(k)});
            py::array_t<int> labels_array({num_results, static_cast<size_t>(k)});
            
            auto dist_buf = dist_array.mutable_unchecked<2>();
            auto labels_buf = labels_array.mutable_unchecked<2>();
            
            #pragma omp parallel for
            for (size_t i = 0; i < num_results; ++i) {
                std::copy(distances[i].begin(), distances[i].end(), &dist_buf(i, 0));
                std::copy(labels[i].begin(), labels[i].end(), &labels_buf(i, 0));
            }
            
            return std::make_pair(dist_array, labels_array);
        }, py::arg("queries"), py::arg("k"), py::arg("param_efSearch") = 0);
}