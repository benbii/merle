# Instruction on Reproducing Results

Initialize submodules. This repo pulls `teb` (Tree Encoded Bitmap) as submodule
exclusively for the testing data (~500MB). We do not depend on teb.

Compile with basic CMake command (we use CUDA 13 with sm_86; other setups may
require code changes):
```sh
cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
make -j16
```

## Bitmap Collection Figures

easy: `bash benchCollection.sh`  
AFAIK the `rename` package in different distributions have various types of
syntaxs so the following may likely fail. These commands rename produced
results into those accepted in res/wahDrawFigs.ipynb

# SSB / TPC-H

Manually download SSB and TPC-H database population tool to `tbl/ssb-dbgen/`
and `tbl/dbgen/` respectively.
> TPC-H licensing prevents us from including it.  
> SSB can be fetched from GitHub, but may require manual fix to compile.

Generate data (SF=20) for SSB and TPC-H:
```sh
make -C tbl/ssb-dbgen -j16
make -C tbl/dbgen dbgen -j16
(cd tbl/ssb-dbgen/; ./dbgen -v -s 20 -T a)
(cd tbl/dbgen/; ./dbgen -vf -s 20)
```

Run `tbl/extractCols.sh` to turn the generated text representation into binary
columnar representation.

## Index Creation Figure

Run `tbl/colJoin.sh` to *both* populate index (WAH and Roaring) *and* obtain
timing for index creation.  
This script also generates bitmaps for all other experiments, like Zipf bitmaps.
It takes ~96GB memory at peak. If lacking memory, remove some `&`s in its first
lines (`extractCols.py`) and reduce some parallelism.

Manually type its output (index creation timing) and index file size (`ls -l
wahData/st`) to `res/drawFigs.ipynb`.

## Main Performance Figure

Run `build/wahProfileGPU benchSTZ >res/benchSTZ.csv`

## Multi-GPU Figure

If >=2 GPU were present when running, then everything should work; otherwise
plotting script errors out and no image will generate. To fix this, copy all
contents of `benchSTZ.csv`, paste them, and change `-0` to `-1`. The generated
multi GPU figure will be meaningless though.

```csv
case,merle,xfer,dnq,xfer,roaring
S12-0,0.084921,0.107838,0.199884,0.937939,0.606687
S13-0,0.060744,0.045804,0.206703,0.952850,0.421432
S12-1,0.084921,0.107838,0.199884,0.937939,0.606687
S13-1,0.060744,0.045804,0.206703,0.952850,0.421432
...
```

## Nsy

Execution breakdown figures require manually profiling with NVIDIA Nsight
systems and filling obtained kernel percentages manually.

1. Open nsy and select target (usually localhost connection)
2. Working directory: directory containing this source
3. Command line: `build/wahProfileGPU benchSTZ`
4. **Edit `runTestCases.cu`!!!** Mask all but `t = s34(ctx)`
5. In function `s34`, mask the decode-and-query codes!!
6. Compile, then start profiling in nsy
7. Unmask `s34`, then mask all but `profZipf(ctx, xx, false)` for Zipf conjunctions
8. Compile, then start profiling in nsy again
9. Repeat with Zipf disjunctions
10. You should now have 3 reports now.
11. In nsy report ui choose 'timeline view', then CUDA HW - kernels - launch_box
12. Do not care about Memory or CPU -- these are for separate benchmarks
13. Take note of the percentages of each function. Right click function - Show
    in events view to see full function name.

Mapping from garbled function names to pie chart components:
- function contains scan_event -> scans
- merge_path_partitioning -> partitioning
- transform_lbs -> resAssoc
- sorted_search{,_flg} -> finalTarget
- pleaseWork (yes it is really named this) -> fused scan / finalTarget
- wahOr -> or operations
- wahAnd* -> and operations

## GENERATE THE FIGURES!!!
Finally, open `res/drawFigs.ipynb` and run all cells.

