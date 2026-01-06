#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "point_matcher.hpp"
#include "icp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(helicopter_cpp, m) {
    m.doc() = "Module for computing point cloud orientation using triangle matching and ICP.";

    py::class_<TrianglePointMatcher>(m, "TrianglePointMatcher")
        .def(py::init<const MatrixX3fRow&, int>(),
             py::arg("reference_points"),
             py::arg("n") = 5,
             R"doc(
Create a TrianglePointMatcher with reference points.

Args:
    reference_points: Nx3 numpy array of reference point coordinates
    n: Number of triangle candidates to try when matching (lower is faster)
)doc")
    .def("get_alignment", [](TrianglePointMatcher &self, const py::EigenDRef<const MatrixX3fRow>& sample_points) {
            auto [q, t] = self.get_alignment(sample_points);
            Eigen::Vector4f q_vec(q.w(), q.x(), q.y(), q.z());
            return std::make_tuple(q_vec, t);
    }, py::arg("sample_points"),
             R"doc(
Find rotation + translation that best explains a rigid transform from reference to sample.

Args:
    sample_points: Nx3 numpy array of measured point coordinates

Returns:
    tuple[numpy.ndarray, numpy.ndarray] : (quaternion_xyzw, translation_xyz)
)doc");

    py::class_<ICP>(m, "ICP")
    .def(py::init<int, float, float>(),
         py::arg("max_iter"),
         py::arg("distance_threshold"),
         py::arg("etol"),
         R"doc(
Create an ICP solver.

Args:
    max_iter: Maximum number of iterations
    distance_threshold: Distance beyond which correspondence will not be calculated.
    etol: Alignment error at which to stop iterating early
)doc")
    .def("get_correspondences",
        &ICP::get_correspondences,
        py::arg("ref_points"),
        py::arg("sample_points"))
    .def("iterate", [](const ICP &self, const MatrixX3fRow &ref, const MatrixX3fRow &sample) {
        auto [q, t] = self.iterate(ref, sample);
        Eigen::Vector4f q_vec(q.w(), q.x(), q.y(), q.z());

        return std::make_tuple(q_vec, t);
    },
        py::arg("ref_points"),
        py::arg("sample_points"),
        R"doc(
Performs ICP.

Args:
    ref_points: Points to be transformed (Nx3 Eigen/Numpy)
    sample_points: Measured points (Nx3 Eigen/Numpy)

Returns:
    tuple[numpy.ndarray, numpy.ndarray] : (quaternion_xyzw, translation_xyz)
)doc");
}