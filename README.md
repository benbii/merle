# Data Population

TBD. If `du -sh .` gives ~100GB then data is probably already populated
properly.

# Result Reproduction

All tests done under CUDA 13.0.88, which is an extremely recent, bleeding edge
version as of Nov. 25. It is unlikely any lower version would work given the
substantial changes CUDA 13 introduces.

All tests are hard-coded and tuned towards CUDA architecture to sm_86 and are
run on RTX 3090 24GB. Modify related fields in CMakeLists and Makefile for other
archs.

## Fused Bitmap

```sh
cmake -GNinja -S. -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build
build/mydemo ssb/20ssbCols synth/20zfcols $(nproc) >build/20Out
build/mydemo ssb/100ssbCols synth/100zfcols $(nproc) >build/100Out
```

SSB SF=100 requires <20G VRAM. Skip SSB SF=100 if lacking VRAM.
Synthetic schema has fewer columns, so not too much VRAM usage even at SF=100.

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

## MeRLE

Standalone MeRLE results (i.e. those produced in `merle/`) are NOT used in the
paper. Rather `*/wah_roast.cu` integrates MeRLE with fused bitmap codes, and
running fused bitmap benchmarks also produces MeRLE (WAH) results.  
See `merle/` for more.
