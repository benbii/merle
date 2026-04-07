# Contributor Guide

## Overview

This repository collects several GPU database and bitmap-index experiments:

- The top-level CMake project builds `mydemo`, the fused bitmap benchmark used for SSB and synthetic Zipf workloads.
- `crystal/` vendors the Crystal baseline and its SSB query implementations.
- `rtscan/` contains the RTScan baseline and OptiX-based scan variants.
- `merle/` contains WAH and related bitmap benchmarking code.

Most workflows are benchmark-oriented rather than library-oriented. Expect generated datasets, large logs, and hardware-specific build settings.

## Repository Layout

- `CMakeLists.txt`, `main.c`, `primitive*.cuh`: top-level fused bitmap build.
- `ssb/`: SSB query kernels, CPU reference code, and index demos.
- `synth/`: synthetic Zipf data generation and benchmark kernels.
- `crystal/`: Crystal baseline, test data tooling, and SSB binaries.
- `rtscan/`: RTScan baseline, scripts, Makefile targets, and bundled OptiX source.
- `merle/`: WAH-related experiments and figure-generation helpers.
- `drawFigs/`: checked-in benchmark outputs and plotting scripts.
- `batchgen.sh`: top-level end-to-end script for data generation, builds, and benchmark runs.

## Setup

Initialize submodules before touching `merle/`:

```sh
git submodule update --init --recursive
```

The codebase assumes an NVIDIA CUDA environment and substantial GPU memory. The top-level README was tested with CUDA 13, while `rtscan/` documents an OptiX 7.5 setup. Scale factor `20` is the most important reproducibility target because Crystal and RTScan flows depend on it.

Common tools used across the repo:

- `cmake` and `ninja`
- `nvcc`
- `clang`
- `make`
- Python 3 with `numpy`
- `ionice` for long data-generation runs

## Build Commands

Build the main fused bitmap benchmark from the repository root:

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -C build
```

Build Crystal:

```sh
make -C crystal -j"$(nproc)"
```

Build RTScan from its own tree when needed:

```sh
make -C rtscan -j"$(nproc)" rtscan
```

For a full reproduction run, prefer the checked-in automation:

```sh
bash batchgen.sh 20
```

`batchgen.sh` generates synthetic columns, builds the top-level project, prepares SSB data, runs Crystal for `SF=20`, and writes benchmark outputs into `drawFigs/`.

## Validation

There is no unified `ctest` or Python test suite in this repository. Validate changes by running the smallest workflow that exercises the code you touched.

Typical checks:

- Top-level benchmark smoke test:

```sh
build/mydemo ssb/20ssbCols synth/20zfcols "$(nproc)"
```

- Full top-level data/build/run path:

```sh
bash batchgen.sh 20
```

- Crystal baseline:

```sh
make -C crystal -j"$(nproc)"
```

- RTScan experiment path:

```sh
mkdir -p rtscan/optix-scan/build rtscan/bin rtscan/log rtscan/data
(cd rtscan && bash script/gen_data.sh && python script/run.py)
```

If you change build flags, kernels, or data formats, include the exact command you used to validate the change in your commit message or PR description.

## Coding Conventions

Match the surrounding file instead of reformatting aggressively.

- C/CUDA code generally uses 2-space indentation and same-line braces.
- Keep includes local-first when that is already the file’s pattern.
- Prefer descriptive helper names in `snake_case`; preserve existing public names and benchmark labels.
- Avoid broad cleanup edits in benchmark code. Small, behavior-focused diffs are easier to validate on GPU-heavy codepaths.
- Comment only when the kernel logic, indexing math, or benchmark setup is not immediately obvious.

## Data and Generated Artifacts

This repository mixes source with generated benchmark outputs. Be deliberate about what you commit.

- `build/`, `rtscan/bin/`, `rtscan/log/`, generated `ssb/*Cols`, and generated `synth/*zfcols` are build or run artifacts.
- `drawFigs/*.txt` files are benchmark outputs and may be intentionally versioned when updating paper figures.
- Large local logs such as `llama.log` should not be treated as source inputs.

Do not commit regenerated outputs unless the change is specifically about benchmark results, plotting, or reproducibility data.

## Contributor Expectations

- Keep changes scoped to one subsystem when possible; top-level, `crystal/`, `rtscan/`, and `merle/` have distinct build flows.
- Call out hardware assumptions in reviews and PRs, especially VRAM requirements and CUDA/OptiX version dependencies.
- Do not silently change benchmark parameters, scale factors, or output file names; those feed plotting and paper-writing workflows.
- When modifying scripts, preserve existing relative paths so the repository-root workflows continue to work.
