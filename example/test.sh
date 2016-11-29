#!/bin/bash

python2.7 ../src/estimate.py -h > estimate.help
python2.7 ../src/estimate.py 'Estimation Tool.mm'
python2.7 ../src/estimate.py --sort 'Estimation Tool.mm' -o 'Estimation Tool.mm.sort.xlsx'
python2.7 ../src/estimate.py --no-roles 'Estimation Tool.mm' -o 'Estimation Tool.mm.nr.xlsx'
python2.7 ../src/estimate.py --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99.xlsx'
python2.7 ../src/estimate.py --p99 --no-roles 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99nr.xlsx'
