#!/bin/bash -e

cd $(dirname ${0})

E="../estimation_tools/estimate.py"
if true; then
 [ -L themes ] || ln -s ../estimation_tools/themes themes

 python $E -h > estimate.help
 python $E 'Estimation Tool.mm' -o 'Estimation Tool.mm.xlsx'
 python $E --sort 'Estimation Tool.mm' -o 'Estimation Tool.mm.sort.xlsx'
 python $E --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99.xlsx'
 python $E --p99 --no-roles 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99nr.xlsx'
 python $E --formulas --stages 'Estimation Tool.mm' -o 'Estimation Tool.mm.f.xlsx'
 python $E --filter-visibility --formulas --p99 --stages 'Estimation Tool.mm' -o 'Estimation Tool.mm.filter.xlsx'
 python $E --theme themes.light --formulas 'Estimation Tool.mm' -o 'Estimation Tool.mm.light.xlsx'
 python $E --filter-visibility --mvp 'Estimation Tool.mm' -o 'Estimation Tool.mm.mvp.xlsx'
 python $E --filter-visibility --formulas --mvp 'Estimation Tool.mm' -o 'Estimation Tool.mm.mvp.f.xlsx'
 python $E --filter-visibility --formulas --mvp --no-data-validation 'Estimation Tool.mm' -o 'Estimation Tool.mm.mvp.nv.xlsx'
 python $E --filter-visibility --formulas --mvp --corrections bogfix:0.05,codecheck:0.05,tests:0.25 'Estimation Tool.mm' -o 'Estimation Tool.mm.mvp.ff.xlsx'
 python $E --formulas 'Estimation Tool.deep.mm' -o 'Estimation Tool.mm.deep.xlsx'
fi


