#!/bin/bash
set -x

gfortran constants.f90 modules.f90 problem.f90 && 
./a.out && 
python plot_escape.py
