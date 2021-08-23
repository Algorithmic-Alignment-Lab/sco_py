#include "numpy_utils.hpp"
#include <boost/python.hpp>
#include <iostream>

namespace py = boost::python;

void pyTestReset(py::object x, int N){
  double* p = getPointer<double>(x);
  for (int i=0; i < N; i++){
    p[i] = 0;
  }
}


BOOST_PYTHON_MODULE(csco) {
  py::def("test_reset", &pyTestReset, (py::arg("x"), py::arg("N")));
}
