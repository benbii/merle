#!/bin/bash
set -e
if ! which cmake ninja clang ionice; then
  exit 2
fi
if [ $# -ne 1 ]; then
    echo "Usage: bash batchgen.sh <SF>"
    exit 1
fi
cd "$(dirname "$0")"

mkdir -p "synth/${1}zfcols"/{04,08,12,16,20}_{16,32}
nr_elem=$((6000000 * $1))
for skew in 04 08 12 16 20; do
  max_dimval=$((nr_elem / 100))
  o="synth/${1}zfcols/${skew}_16"
  # Generate attribute columns
  ionice -c 3 python synth/onegen_zipf.py "1.${skew}" 65536 "$nr_elem" "$o/a1" &
  ionice -c 3 python synth/onegen_zipf.py "1.${skew}" 65536 "$nr_elem" "$o/a2" &
  # Generate foreign keys
  ionice -c 3 python synth/onegen_zipf.py "1.${skew}" "$max_dimval" "$nr_elem" "$o/fk" &
  # Repeat with 32b
  o="synth/${1}zfcols/${skew}_32"
  ionice -c 3 python synth/onegen_zipf.py "1.${skew}" 4294967296 "$nr_elem" "$o/a1" &
  ionice -c 3 python synth/onegen_zipf.py "1.${skew}" 4294967296 "$nr_elem" "$o/a2" &
  ionice -c 3 python synth/onegen_zipf.py "1.${skew}" "$max_dimval" "$nr_elem" "$o/fk" &
done

cmake -S. -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -C build
clang ssb-dbgen-lite.c -O3 -march=native -o build/dbgen-lite -lm -fopenmp
if [ "$1" -eq 20 ]; then
  bash rtscan/script/run.sh &
  make -C crystal/test/loader -j32
  make -C crystal -j32
  mkdir -p crystal/test/ssb/data
  cp date.tbl.xz ssb/20ssbCols
  ionice -c 2 build/dbgen-lite -s 20 -d ssb/20ssbCols -b -t
  ln -s "$(realpath ssb/20ssbCols)" crystal/test/ssb/data/s20
  echo -e "Case\tCrystal" >drawFigs/crystal.txt
  pushd crystal/test
  ionice -c 2 python util.py ssb "$1" transform
  cd ..
  find bin/ssb -type f -executable -exec '{}' -t 300 \; | tee -a ../drawFigs/crystal.txt
  popd
else
  build/dbgen-lite -s "$1" -d "ssb/${1}ssbCols" -b
fi
wait
build/mydemo "ssb/${1}ssbCols" "synth/${1}zfcols" 32 | tee "drawFigs/sf${1}.txt"
