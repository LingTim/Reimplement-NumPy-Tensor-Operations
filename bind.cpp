#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "NDarray.h"

namespace py = pybind11;

PYBIND11_MODULE(my_NDarray_module, m)
{
    m.doc() = "C++ Tensor library built with pybind11";

    // 綁定NDarray<float>並在Python中稱他為tensor
    py::class_<NDarray<float>>(m, "Tensor")

        // 綁定建構子
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, float>())
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&>())
        
        // 綁定屬性
        .def_property_readonly("shape", &NDarray<float>::shape)
        .def_property_readonly("strides", &NDarray<float>::strides)
        .def_property_readonly("ndim", &NDarray<float>::ndim)

        // 綁定 NDarray by NDarray 運算
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)

        // 綁定 NDarray by Scalar 運算
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())

        // 綁定 Scalar vs NDarray 運算
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)

        // 綁定 Reduction
        .def("sum", &NDarray<float>::sum, py::arg("axis") = -1)

        // 綁定索引存取
        .def("__getitem__", [](const NDarray<float>& t, const std::vector<size_t>& indices) {
            return t(indices);
        })
        .def("__setitem__", [](NDarray<float>& t, const std::vector<size_t>& indices, float val) {
            t(indices) = val;
        });
}