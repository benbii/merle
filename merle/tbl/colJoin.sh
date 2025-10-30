#!/bin/bash
set -e
cd "$(dirname "$0")"
B="../build/wahConv"
G="../build/wahProfileGPU"
mkdir -p cols bitset zipf ../wahData/st ../wahData/zipf

python extractCols.py --inPath dbgen/part.tbl --outDir cols/tPart \
  --useCols 3 6 --pad0 >/dev/null &
python extractCols.py --inPath dbgen/orders.tbl --outDir cols/tOrder \
  --useCols 0 1 3 4 5 --needFill0 >/dev/null &
python extractCols.py --inPath dbgen/customer.tbl --outDir cols/tCustomer \
  --useCols 6 --pad0 >/dev/null &
python extractCols.py --inPath dbgen/lineitem.tbl --outDir cols/tLineitem \
  --useCols 0 1 4 6 10 12 14 >/dev/null &
python extractCols.py --inPath ssb-dbgen/customer.tbl --outDir cols/sCustomer \
  --useCols 3 5 --pad0 >/dev/null &
python extractCols.py --inPath ssb-dbgen/part.tbl --outDir cols/sPart \
  --useCols 2 4 --pad0 >/dev/null &
python extractCols.py --inPath ssb-dbgen/supplier.tbl --outDir cols/sSupplier \
  --useCols 3 5 --pad0 >/dev/null &
python extractCols.py --inPath ssb-dbgen/lineorder.tbl --outDir cols/sLineorder \
  --useCols 0 2 3 4 5 8 11 >/dev/null &

(cd ..; python tbl/zipfLstGen.py >/dev/null)
"$B" arrToWah --inFmt zipf/12l%zu.txt --outFmt ../wahData/zipf/12l%zu.wah \
  --nFile 32 >/dev/null &
"$B" arrToWah --inFmt zipf/14l%zu.txt --outFmt ../wahData/zipf/14l%zu.wah \
  --nFile 32 >/dev/null &
"$B" arrToWah --inFmt zipf/16l%zu.txt --outFmt ../wahData/zipf/16l%zu.wah \
  --nFile 32 >/dev/null &
"$B" arrToWah --inFmt zipf/18l%zu.txt --outFmt ../wahData/zipf/18l%zu.wah \
  --nFile 32 >/dev/null &
"$B" arrToWah --inFmt zipf/20l%zu.txt --outFmt ../wahData/zipf/20l%zu.wah \
  --nFile 32 >/dev/null
wait
rm cols/tOrder/col0.bin

# S34
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col2.bin" \
  --dim1ColFile "cols/sCustomer/col3.bin" \
  --scanMin 140 --scanMax 141 --scanOut "bitset/s34l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col3.bin" \
  --scanMin 95 --scanMax 96 --scanOut "bitset/s34l2.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col5.bin" \
  --scanMin 19971200 --scanMax 19971300 --scanOut "bitset/s34l4.bin"
wait
echo SSB Q3.4 took $(($(date +%s) - stt)) msecs to populate index on CPU
"$B" scanIntoBitset --nDup 1 --factColFile "cols/sLineorder/col2.bin" \
  --dim1ColFile "cols/sCustomer/col3.bin" \
  --scanMin 55 --scanMax 56 --scanOut "bitset/s34l1.bin"
"$B" scanIntoBitset --nDup 1 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col3.bin" \
  --scanMin 140 --scanMax 141 --scanOut "bitset/s34l3.bin"
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 5 --inFmt bitset/s34l%zu.bin \
  --outFmt ../wahData/st/-34l%zu%s >/dev/null
echo SSB Q3.4 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col2.bin" \
  --dim1ColFile "cols/sCustomer/col3.bin" \
  --scanMin 140 --scanMax 141
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col3.bin" \
  --scanMin 95 --scanMax 96
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col5.bin" \
  --scanMin 19971200 --scanMax 19971300
echo SSB Q3.4 took $(($(date +%s) - stt)) msecs to populate index on GPU
"$G" benchJoin --nDup 10 --factColFile "cols/sLineorder/col2.bin" \
  --dim1ColFile "cols/sCustomer/col3.bin" \
  --scanMin 55 --scanMax 56
"$G" benchJoin --nDup 10 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col3.bin" \
  --scanMin 140 --scanMax 141
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 5 --inFmt bitset/s34l%zu.bin
echo SSB Q3.4 took $(($(date +%s) - stt)) msecs to compress index on GPU

# S12
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col11.bin" \
  --scanMin 4 --scanMax 7 --scanOut "bitset/s12l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col8.bin" \
  --scanMin 26 --scanMax 36 --scanOut "bitset/s12l1.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col5.bin" \
  --scanMin 19940100 --scanMax 19940200 --scanOut "bitset/s12l2.bin"
wait
echo SSB Q1.2 took $(($(date +%s) - stt)) msecs to populate index on CPU
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 3 --inFmt bitset/s12l%zu.bin \
  --outFmt ../wahData/st/-12l%zu%s >/dev/null
echo SSB Q1.2 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col11.bin" \
  --scanMin 4 --scanMax 7
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col8.bin" \
  --scanMin 26 --scanMax 36
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col5.bin" \
  --scanMin 19940100 --scanMax 19940200
echo SSB Q1.2 took $(($(date +%s) - stt)) msecs to populate index on GPU
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 3 --inFmt bitset/s12l%zu.bin
echo SSB Q1.2 took $(($(date +%s) - stt)) msecs to compress index on GPU
# S13
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col11.bin" \
  --scanMin 5 --scanMax 8 --scanOut "bitset/s13l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col8.bin" \
  --scanMin 26 --scanMax 36 --scanOut "bitset/s13l1.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col5.bin" \
  --scanMin 19940201 --scanMax 19940208 --scanOut "bitset/s13l2.bin"
wait
echo SSB Q1.3 took $(($(date +%s) - stt)) msecs to populate index on CPU
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 3 --inFmt bitset/s13l%zu.bin \
  --outFmt ../wahData/st/-13l%zu%s >/dev/null
echo SSB Q1.3 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col11.bin" \
  --scanMin 5 --scanMax 8
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col8.bin" \
  --scanMin 26 --scanMax 36
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col5.bin" \
  --scanMin 19940201 --scanMax 19940208
echo SSB Q1.3 took $(($(date +%s) - stt)) msecs to populate index on GPU
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 3 --inFmt bitset/s13l%zu.bin
echo SSB Q1.3 took $(($(date +%s) - stt)) msecs to compress index on GPU

# S23
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col3.bin" \
  --dim1ColFile "cols/sPart/col4.bin" \
  --scanMin 676 --scanMax 677 --scanOut "bitset/s23l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col5.bin" \
  --scanMin 4 --scanMax 5 --scanOut "bitset/s23l1.bin"
wait
echo SSB Q2.3 took $(($(date +%s) - stt)) msecs to populate index on CPU
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 2 --inFmt bitset/s23l%zu.bin \
  --outFmt ../wahData/st/-23l%zu%s >/dev/null
echo SSB Q2.3 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col3.bin" \
  --dim1ColFile "cols/sPart/col4.bin" \
  --scanMin 676 --scanMax 677
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col5.bin" \
  --scanMin 4 --scanMax 5
echo SSB Q2.3 took $(($(date +%s) - stt)) msecs to populate index on GPU
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 2 --inFmt bitset/s23l%zu.bin
echo SSB Q2.3 took $(($(date +%s) - stt)) msecs to compress index on GPU

# S41
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col3.bin" \
  --dim1ColFile "cols/sPart/col2.bin" \
  --scanMin 1 --scanMax 2 --scanOut "bitset/s41l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col5.bin" \
  --scanMin 1 --scanMax 2 --scanOut "bitset/s41l2.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/sLineorder/col2.bin" \
  --dim1ColFile "cols/sCustomer/col5.bin" \
  --scanMin 3 --scanMax 4 --scanOut "bitset/s41l3.bin"
wait
echo SSB Q4.1 took $(($(date +%s) - stt)) msecs to populate index on CPU
"$B" scanIntoBitset --nDup 1 --factColFile "cols/sLineorder/col3.bin" \
  --dim1ColFile "cols/sPart/col2.bin" \
  --scanMin 4 --scanMax 5 --scanOut "bitset/s41l1.bin"
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 4 --inFmt bitset/s41l%zu.bin \
  --outFmt ../wahData/st/-41l%zu%s >/dev/null
echo SSB Q4.1 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col3.bin" \
  --dim1ColFile "cols/sPart/col2.bin" \
  --scanMin 1 --scanMax 2
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col4.bin" \
  --dim1ColFile "cols/sSupplier/col5.bin" \
  --scanMin 1 --scanMax 2
"$G" benchJoin --nDup 1000 --factColFile "cols/sLineorder/col2.bin" \
  --dim1ColFile "cols/sCustomer/col5.bin" \
  --scanMin 3 --scanMax 4
echo SSB Q4.1 took $(($(date +%s) - stt)) msecs to populate index on GPU
"$G" benchJoin --nDup 10 --factColFile "cols/sLineorder/col3.bin" \
  --dim1ColFile "cols/sPart/col2.bin" \
  --scanMin 4 --scanMax 5
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 4 --inFmt bitset/s41l%zu.bin
echo SSB Q4.1 took $(($(date +%s) - stt)) msecs to compress index on GPU

# T3
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col0.bin" \
  --dim1ColFile "cols/tOrder/col4.bin" \
  --scanMin 19950400 --scanMax 99999999 --scanOut "bitset/t3l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col10.bin" \
  --scanMin 0 --scanMax 19950300 --scanOut "bitset/t3l2.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col0.bin" \
  --dim1ColFile "cols/tOrder/col1.bin" \
  --dim2ColFile "cols/tCustomer/col6.bin" \
  --scanMin 1 --scanMax 2 --scanOut "bitset/t3l4.bin"
wait
echo TPCH Q3 took $(($(date +%s) - stt)) msecs to populate index on CPU
"$B" scanIntoBitset --nDup 1 --factColFile "cols/tLineitem/col0.bin" \
  --dim1ColFile "cols/tOrder/col4.bin" \
  --scanMin 19950315 --scanMax 19950400 --scanOut "bitset/t3l1.bin"
"$B" scanIntoBitset --nDup 1 --factColFile "cols/tLineitem/col10.bin" \
  --scanMin 19950300 --scanMax 19950315 --scanOut "bitset/t3l3.bin"
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 5 --inFmt bitset/t3l%zu.bin \
  --outFmt ../wahData/st/3l%zu%s >/dev/null
echo TPCH Q3 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col0.bin" \
  --dim1ColFile "cols/tOrder/col4.bin" \
  --scanMin 19950400 --scanMax 99999999
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col10.bin" \
  --scanMin 0 --scanMax 19950300
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col0.bin" \
  --dim1ColFile "cols/tOrder/col1.bin" \
  --dim2ColFile "cols/tCustomer/col6.bin" \
  --scanMin 1 --scanMax 2
echo TPCH Q3 took $(($(date +%s) - stt)) msecs to populate index on GPU
"$G" benchJoin --nDup 10 --factColFile "cols/tLineitem/col0.bin" \
  --dim1ColFile "cols/tOrder/col4.bin" \
  --scanMin 19950315 --scanMax 19950400
"$G" benchJoin --nDup 10 --factColFile "cols/tLineitem/col10.bin" \
  --scanMin 19950300 --scanMax 19950315
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 5 --inFmt bitset/t3l%zu.bin
echo TPCH Q3 took $(($(date +%s) - stt)) msecs to compress index on GPU

# T6
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col4.bin" \
  --scanMin 24 --scanMax 25 --scanOut "bitset/t6l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col6.bin" \
  --scanMin 5 --scanMax 7 --scanOut "bitset/t6l1.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col10.bin" \
  --scanMin 19930000 --scanMax 19950000 --scanOut "bitset/t6l2.bin"
wait
echo TPCH Q6 took $(($(date +%s) - stt)) msecs to populate index on CPU
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 3 --inFmt bitset/t6l%zu.bin \
  --outFmt ../wahData/st/6l%zu%s >/dev/null
echo TPCH Q6 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col4.bin" \
  --scanMin 24 --scanMax 25
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col6.bin" \
  --scanMin 5 --scanMax 7
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col10.bin" \
  --scanMin 19930000 --scanMax 19950000
echo TPCH Q6 took $(($(date +%s) - stt)) msecs to populate index on GPU
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 3 --inFmt bitset/t6l%zu.bin
echo TPCH Q6 took $(($(date +%s) - stt)) msecs to compress index on GPU

# T12
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col14.bin" \
  --scanMin 2 --scanMax 3 --scanOut "bitset/t12l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col12.bin" \
  --scanMin 19940100 --scanMax 19940108 --scanOut "bitset/t12l2.bin"
wait
echo TPCH Q12 took $(($(date +%s) - stt)) msecs to populate index on CPU
"$B" scanIntoBitset --nDup 1 --factColFile "cols/tLineitem/col14.bin" \
  --scanMin 7 --scanMax 8 --scanOut "bitset/t12l1.bin"
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 3 --inFmt bitset/t12l%zu.bin \
  --outFmt ../wahData/st/12l%zu%s >/dev/null
echo TPCH Q12 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col14.bin" \
  --scanMin 2 --scanMax 3
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col12.bin" \
  --scanMin 19940100 --scanMax 19940108
echo TPCH Q12 took $(($(date +%s) - stt)) msecs to populate index on GPU
"$G" benchJoin --nDup 10 --factColFile "cols/tLineitem/col14.bin" \
  --scanMin 7 --scanMax 8
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 3 --inFmt bitset/t12l%zu.bin
echo TPCH Q12 took $(($(date +%s) - stt)) msecs to compress index on GPU

# T17
stt=$(date +%s)
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col1.bin" \
  --dim1ColFile "cols/tPart/col6.bin" \
  --scanMin 39 --scanMax 40 --scanOut "bitset/t17l0.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col1.bin" \
  --dim1ColFile "cols/tPart/col3.bin" \
  --scanMin 14 --scanMax 15 --scanOut "bitset/t17l1.bin" &
"$B" scanIntoBitset --nDup 1000 --factColFile "cols/tLineitem/col4.bin" \
  --scanMin 0 --scanMax 6 --scanOut "bitset/t17l2.bin"
wait
echo TPCH Q17 took $(($(date +%s) - stt)) msecs to populate index on CPU
stt=$(date +%s)
"$B" bitsetToWah --nDup 1000 --nFile 3 --inFmt bitset/t17l%zu.bin \
  --outFmt ../wahData/st/17l%zu%s >/dev/null
echo TPCH Q17 took $(($(date +%s) - stt)) msecs to compress index on CPU
stt=$(date +%s)
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col1.bin" \
  --dim1ColFile "cols/tPart/col6.bin" \
  --scanMin 39 --scanMax 40
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col1.bin" \
  --dim1ColFile "cols/tPart/col3.bin" \
  --scanMin 14 --scanMax 15
"$G" benchJoin --nDup 1000 --factColFile "cols/tLineitem/col4.bin" \
  --scanMin 0 --scanMax 6
echo TPCH Q17 took $(($(date +%s) - stt)) msecs to populate index on GPU
stt=$(date +%s)
"$G" benchCompression --nDup 1000 --nFile 3 --inFmt bitset/t17l%zu.bin
echo TPCH Q17 took $(($(date +%s) - stt)) msecs to compress index on GPU

rm -r cols bitset zipf
