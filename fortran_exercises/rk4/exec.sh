#!/bin/bash
set -x

gfortran constants.f90 modules.f90 main.f90 && 
./a.out && 
python plotsol.py
