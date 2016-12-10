#!/bin/bash -e

if true; then
 python2.7 ../src/estimate.py -h > estimate.help
 python2.7 ../src/estimate.py 'Estimation Tool.mm'
 python2.7 ../src/estimate.py --sort 'Estimation Tool.mm' -o 'Estimation Tool.mm.sort.xlsx'
 python2.7 ../src/estimate.py --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99.xlsx'
 python2.7 ../src/estimate.py --p99 --no-roles 'Estimation Tool.mm' -o 'Estimation Tool.mm.p99nr.xlsx'
 python2.7 ../src/estimate.py --formulas 'Estimation Tool.mm' -o 'Estimation Tool.mm.f.xlsx'
 python2.7 ../src/estimate.py --filter-visibility --formulas --p99 'Estimation Tool.mm' -o 'Estimation Tool.mm.filter.xlsx'
 python2.7 ../src/estimate.py --theme themes.light --formulas 'Estimation Tool.mm' -o 'Estimation Tool.mm.light.xlsx'
fi

# you have to add the script from test.xba to Standard.Test module manually
# ! uncomment the following code only after that !
# ! Unfortunately, there is a bug in LibreOffice with filtering 'by blank' in xlsx (it works with ods) !
if false; then
 ls *.xlsx | while read f; do
  libreoffice --headless "$f" "macro:///Standard.Test.RecalculateAll"
 done
fi



