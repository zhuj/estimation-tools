#!/bin/bash -e

if true; then
 [ -L themes ] || ln -s ../src/themes themes

 python2.7 ../src/estimate.py -h > estimate.help
 python2.7 ../src/estimate.py 'Estimation Tool.mm'
 python2.7 ../src/estimate.py --sort 'Estimation Tool.mm' -o 'Estimation Tool.mm.sort.xlsx'
 python2.7 ../src/estimate.py --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99.xlsx'
 python2.7 ../src/estimate.py --p99 --no-roles 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99nr.xlsx'
 python2.7 ../src/estimate.py --formulas 'Estimation Tool.mm' -o 'Estimation Tool.mm.f.xlsx'
 python2.7 ../src/estimate.py --filter-visibility --formulas --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.filter.xlsx'
 python2.7 ../src/estimate.py --theme themes.light --formulas 'Estimation Tool.mm' -o 'Estimation Tool.mm.light.xlsx'
 python2.7 ../src/estimate.py --formulas 'Estimation Tool.deep.mm' -o 'Estimation Tool.mm.deep.xlsx'
fi


