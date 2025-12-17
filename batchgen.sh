#!/bin/bash
set -e
if ! which cmake ninja pidwait clang; then
  exit 2
fi
if [ $# -ne 1 ]; then
    echo "Usage: bash batchgen.sh <SF>"
    exit 1
fi
cd "$(dirname "$0")"

pushd crystal/test/ssb/dbgen
make -j32
cp -r . ../dbgen-$1
cd ../dbgen-$1
yes | ./dbgen -v -s "$1" -T c >/dev/null 2>&1 &
yes | ./dbgen -v -s "$1" -T p >/dev/null 2>&1 &
yes | ./dbgen -v -s "$1" -T s >/dev/null 2>&1 &
yes | ./dbgen -v -s "$1" -T d >/dev/null 2>&1 &
yes | ./dbgen -v -s "$1" -T l >/dev/null 2>&1 &
make -C ../loader -j32
make -C ../../.. -j32
popd
if [ "$1" -eq 20 ]; then
  bash rtscan/script/run.sh &
fi
cmake -S. -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -C build

mkdir -p "synth/${1}zfcols"/{04,08,12,16,20}_{16,32}
nr_elem=$((6000000 * $1))
for skew in 04 08 12 16 20; do
  max_dimval=$((nr_elem / 100))
  o="synth/${1}zfcols/${skew}_16"
  # Generate attribute columns
  python synth/onegen_zipf.py "1.${skew}" 65536 "$nr_elem" "$o/a1" &
  python synth/onegen_zipf.py "1.${skew}" 65536 "$nr_elem" "$o/a2" &
  # Generate foreign keys
  python synth/onegen_zipf.py "1.${skew}" "$max_dimval" "$nr_elem" "$o/fk" &
  sleep 1 && pidwait -f "$o"
  # Repeat with 32b
  o="synth/${1}zfcols/${skew}_32"
  python synth/onegen_zipf.py "1.${skew}" 4294967296 "$nr_elem" "$o/a1" &
  python synth/onegen_zipf.py "1.${skew}" 4294967296 "$nr_elem" "$o/a2" &
  python synth/onegen_zipf.py "1.${skew}" "$max_dimval" "$nr_elem" "$o/fk" &
  sleep 1 && pidwait -f "$o"
done

pushd crystal/test
mkdir -p "ssb/data/s$1"
wait
mv ssb/dbgen-$1/*.tbl "ssb/data/s$1"
rm -r ssb/dbgen-$1
if [ "$1" -eq 20 ]; then
  python util.py ssb "$1" transform &
fi
python ../../ssb/extCols.py "ssb/data/s$1"
rm -r "../../ssb/${1}ssbCols" || true
mv "ssb/data/s$1/ssbCols" "../../ssb/${1}ssbCols"
popd
build/mydemo ssb/${1}ssbCols synth/${1}zfcols 32 | tee drawFigs/sf${1}.txt
if [ "$1" -eq 20 ]; then
  cd crystal
  echo -e "Case\tCrystal" >../drawFigs/crystal.txt
  wait
  find bin/ssb -type f -executable -exec '{}' -t 300 \; | tee -a ../drawFigs/crystal.txt
fi
