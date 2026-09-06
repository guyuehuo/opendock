#include <pybind11/pybind11.h>

int add_integers(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(mylib, m) {
    m.def("add_integers", &add_integers, "Add two integers");
}