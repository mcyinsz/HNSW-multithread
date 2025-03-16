#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// 包含所有项目头文件
#include "include/impl/HNSW.h"
#include "include/impl/IndexHNSW.h"
#include "include/utils/Index.h"
#include "include/utils/DistanceComputer.h"
#include "include/utils/RandomGenerator.h"

namespace py = pybind11;

// 处理纯虚函数的Trampoline类
class PyIndex : public Index {
public:
    using Index::Index;
    
    void add(int n, const std::vector<float>& x) override {
        PYBIND11_OVERRIDE_PURE(void, Index, add, n, x);
    }
    
    void search(int n, const std::vector<float>& x, int k,
                std::vector<std::vector<float>>& distances,
                std::vector<std::vector<int>>& labels,
                int Param_efSearch) override {
        PYBIND11_OVERRIDE_PURE(void, Index, search, 
            n, x, k, distances, labels, Param_efSearch);
    }
};

// 核心绑定代码
PYBIND11_MODULE(_core, m) {
    // 导出距离类型
    py::enum_<MetricType>(m, "MetricType")
        .value("INNER_PRODUCT", INNER_PRODUCT)
        .value("L2_DISTANCE", L2_DISTANCE);

    // 基类Index
    py::class_<Index, PyIndex>(m, "Index")
        .def(py::init<int, MetricType>(), 
             py::arg("d") = 0, 
             py::arg("metric") = INNER_PRODUCT)
        .def_readwrite("d", &Index::d)
        .def_readwrite("ntotal", &Index::ntotal);

    // IndexFlat
    py::class_<IndexFlat, Index>(m, "IndexFlat")
        .def(py::init<int, MetricType>(), 
             py::arg("d") = 0, 
             py::arg("metric") = INNER_PRODUCT)
        .def("add", [](IndexFlat& self, py::array_t<float> x) {
            py::buffer_info buf = x.request();
            const int n = buf.shape[0];
            self.add(n, std::vector<float>(x.data(), x.data() + x.size()));
        }, py::arg("x"))
        .def("search", [](IndexFlat& self, py::array_t<float> x, int k) {
            std::vector<std::vector<float>> dist;
            std::vector<std::vector<int>> labels;
            const int n = x.shape()[0];
            self.search(n, 
                std::vector<float>(x.data(), x.data() + x.size()),
                k, dist, labels, 0);
            return py::make_pair(dist, labels);
        }, py::arg("x"), py::arg("k"));

    // HNSW
    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int>(), py::arg("M") = 32)
        .def("set_default_probas", &HNSW::set_default_probas)
        .def("reset", &HNSW::reset)
        .def_readwrite("efConstruction", &HNSW::efConstruction)
        .def_readwrite("efSearch", &HNSW::efSearch);

    // IndexHNSW
    py::class_<IndexHNSW, Index>(m, "IndexHNSW")
        .def(py::init<int, int, MetricType>(), 
             py::arg("d") = 0, 
             py::arg("M") = 32, 
             py::arg("metric") = INNER_PRODUCT)
        .def(py::init<Index*, int>(), 
             py::arg("storage"), 
             py::arg("M") = 32,
             py::keep_alive<1, 2>())
        .def("add", [](IndexHNSW& self, py::array_t<float> x) {
            const int n = x.shape()[0];
            self.add(n, std::vector<float>(x.data(), x.data() + x.size()));
        }, py::arg("x"))
        .def("search", [](IndexHNSW& self, py::array_t<float> x, int k, int ef) {
            std::vector<std::vector<float>> dist;
            std::vector<std::vector<int>> labels;
            const int n = x.shape()[0];
            self.search(n, 
                std::vector<float>(x.data(), x.data() + x.size()),
                k, dist, labels, ef);
            return py::make_pair(dist, labels);
        }, py::arg("x"), py::arg("k"), py::arg("ef"))
        .def_readwrite("init_level0", &IndexHNSW::init_level0)
        .def_readonly("hnsw", &IndexHNSW::hnsw);
}
