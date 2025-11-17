#!/bin/bash
set -xe
cd "$(dirname $0)/.."
mkdir -p log

make clean
make -j16 rtscan DATA_N=1e8 VAREA_N=96 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 ENCODE=0 BUILD_TYPE=Release
bin/rtscan -b 3 -w 1200 -m 1200 -a 1111111 -z 1 -q 11 -p test/scan_cmd_1e8-3c.txt >log/1e8

# SF20 contains 1.2e8 rows
make clean
make -j16 rtscan DATA_N=12e7 VAREA_N=96 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 ENCODE=0 BUILD_TYPE=Release
bin/rtscan -b 3 -w 1000 -m 1000 -a 1333333 -z 1 -q 11 -p test/scan_cmd_1e8-3c.txt >log/1.2e8