cmake_minimum_required(VERSION 2.8)

find_package(Boost COMPONENTS system python thread program_options REQUIRED)
include("boost-python.cmake")

boost_python_module(csco csco.cpp)
