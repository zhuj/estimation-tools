#!/bin/bash -e

E="../estimation_tools/estimate.py"
if true; then
 [ -L themes ] || ln -s ../estimation_tools/themes themes

 python $E -h > estimate.help
 python $E 'Estimation Tool.mm'
 python $E --sort 'Estimation Tool.mm' -o 'Estimation Tool.mm.sort.xlsx'
 python $E --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99.xlsx'
 python $E --p99 --no-roles 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99nr.xlsx'
 python $E --formulas 'Estimation Tool.mm' -o 'Estimation Tool.mm.f.xlsx'
 python $E --filter-visibility --formulas --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.filter.xlsx'
 python $E --theme themes.light --formulas 'Estimation Tool.mm' -o 'Estimation Tool.mm.light.xlsx'
 python $E --formulas --no-mvp 'Estimation Tool.mm' -o 'Estimation Tool.mm.nomvp.xlsx'
 python $E --formulas 'Estimation Tool.deep.mm' -o 'Estimation Tool.mm.deep.xlsx'
fi


