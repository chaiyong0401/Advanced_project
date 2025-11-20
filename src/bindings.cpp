#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>
#include <Eigen/Dense>

#include "controller.h"

namespace bp = boost::python;
BOOST_PYTHON_MODULE(cRoboticsController_wrapper_cpp) 
{
    eigenpy::enableEigenPy();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, Eigen::Dynamic, 1>>();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, 4, 4>>();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, 7, 1>>();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>();

    // Bind cRoboticsController
    bp::class_<cRoboticsController, boost::noncopyable>("cRoboticsController", bp::init<std::string, std::string, double>())
    .def("keyMapping", &cRoboticsController::keyMapping)
    .def("compute", &cRoboticsController::compute)
    .def("printState", &cRoboticsController::printState)
    .def("updateModel", &cRoboticsController::updateModel)
    .def("getCtrlInput", &cRoboticsController::getCtrlInput);
}
