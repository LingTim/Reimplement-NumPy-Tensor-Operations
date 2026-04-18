#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // 讓 C++ 的 std::vector 可以和 Python 的 List 自動轉換
#include <pybind11/operators.h> // 支援運算子多載綁定
#include "NDarray.h"

namespace py = pybind11;

PYBIND11_MODULE(my_tensor_module, m) {
    m.doc() = "C++ Tensor library built with pybind11";

    // 綁定 NDarray<float> 型態，在 Python 裡面我們叫它 "Tensor"
    py::class_<NDarray<float>>(m, "Tensor")
        // 綁定建構子
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, float>()) // 支援傳入 initial_value
        
        // 綁定屬性 (Getters)
        .def_property_readonly("shape", &NDarray<float>::shape)
        .def_property_readonly("strides", &NDarray<float>::strides)
        .def_property_readonly("ndim", &NDarray<float>::ndim)

        // 綁定 Array vs Array 運算
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)

        // 綁定 Array vs Scalar 運算
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())

        // 綁定 Scalar vs Array 運算 (反向運算)
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)

        // 綁定索引存取 (讓 Python 可以用 tensor[[0, 1]] 讀寫資料)
        .def("__getitem__", [](const NDarray<float>& t, const std::vector<size_t>& indices) {
            return t(indices);
        })
        .def("__setitem__", [](NDarray<float>& t, const std::vector<size_t>& indices, float val) {
            t(indices) = val;
        });
}