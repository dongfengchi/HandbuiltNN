cmake_minimum_required(VERSION 3.8)

project(HandbuiltNN)
add_executable(Test HandbuiltNN.cpp CNN.cpp CNN.h)
add_definitions(-std=c++11)