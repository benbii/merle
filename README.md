# Reproducing results

First you need a 16GiB CUDA GPU. Choose a scale factor:
- 8/16GB - SF=I don't know
- 24GB - SF=100
- 32GB - SF=140
- 80GB - SF=300
(note the main method require less VRAM than this; but having less VRAM will
cause the program skip WAH baseline or even crash if too little)

Prepare (20 + SF)GB disk space. SF=20 must be run since Crystal and RTScan
baseline will NOT work at other scale factors.

Then some really basic dependencies like CMake, ninja-build, numpy, nvcc, and
finally:

```bash
bash batchgen.sh YOUR-SF
bash batchgen.sh 20
python drawDigs/drawFigs.py YOUR-SF drawFigs/
```

`batchgen.sh` not only generates data but also compiles and runs programs.  
AI Slop Warning: Generated Teaser Figure is a Gobbled Mess Other Than SF=140.

# Other Contents

All tests done under CUDA 13.0.88, which is an extremely recent, bleeding edge
version as of Nov. 25. It is unlikely any lower version would work given the
substantial changes CUDA 13 introduces.

## Fused Bitmap

```sh
cmake -GNinja -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build
build/mydemo ssb/20ssbCols synth/20zfcols $(nproc) >build/20Out
build/mydemo ssb/100ssbCols synth/100zfcols $(nproc) >build/100Out
```

SSB SF=100 requires <20G VRAM. Skip SSB SF=100 if lacking VRAM.
Synthetic schema has fewer columns, so not too much VRAM usage even at SF=100.  
For synthetic data SF=1 segfaults; must run at least SF=2 due to encoding tricks.

## Crystal

Only SF=20 supported.

```sh
make -C crystal -j$(nproc)
echo -e "Case\tCrystal" >build/crsOut
# Repeat 300 times and report average
(cd crystal; find bin/ssb -type f -executable -exec '{}' -t 300 \;) >>build/crsOut
```

## RTScan

`bash rtscan/script/run.sh`  
It produces 2 log files, with 1e8 (SF=18) and 1.2e8 (SF=20) rows respectively.
RTScan is extremely VRAM hungry and support conjunctive scans only (not joins).
\>20GB VRAM is required.

The log file `rtscan/log/1.2e8` lists VRAM usage in its various steps, and
running time for each selectivity.

## WAH

Bundled in fused bitmap (see `*/wah_roast.cu`).
