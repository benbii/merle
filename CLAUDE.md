# Project Progress Log

## Current Phase: Paper Writing - Evaluation Section

### Completed Work (2025-11-03)

All experimental baselines and benchmarks are complete! The code runs successfully on scale factors 20 and 100, generating comprehensive performance data.

### Evaluation Section Structure (COMPLETED)

Created a structured outline for the evaluation section in `drawFigs/evaluation section draft.md`, although its contents are just a draft and require significant revision. The section is organized into 5 main subsections:

#### 1. Overall Performance vs. External Baselines
- Figure 1: Bar chart comparing WAH, RTScan, Crystal, MyJoin with our approach
(ManyOrs scenarios) across all SSB queries at SF=20
- A potential Figure 1b at SF=100, excluding RTScan (SF>20 not supported) and Crystal (too VRAM hungry)
- In the figure(s) add a geometric mean column
- Analysis: Would be quite invloved. Waiting for further instructions.

#### 2. Performance Across Query-Index Alignment Scenarios
- **Key clarification**: Perfect/ManyOrs/CandChk are NOT competing methods - they represent different ways query predicates align with pre-built bitmap index bins:
  - **Perfect**: Query boundaries exactly match bin boundaries (best case)
  - **ManyOrs**: Query spans 3 bins but boundaries align (requires OR operations)
  - **CandChk**: Query boundaries fall inside bins (requires candidate verification)
  - Real scenarios are a mix of three.  ,a
- Figure 2: Box plot showing overhead distribution for each scenario across SSB queries
- Analysis: Quantifies overhead percentages; explains that real workloads see a mix depending on binning granularity

#### 3. Index Creation and Memory Overhead
- Figure 3: Dual subplot comparing creation time and memory usage (our approach vs WAH compression)
- Table 2: Total overhead summary showing 8-10x faster creation, 13% more memory than WAH
- Analysis: Tradeoff justified - creation happens once, queries benefit continuously
- TODO: add proper comparison with RTScan

#### 4. Ablation Study: Kernel Fusion Benefits
- **Key insight**: The nofuse breakdown serves dual purposes:
  1. Reveals workload composition (what proportion is bitmap access vs computation)
  2. Shows what overheads fusion eliminates
- Figure 4: Stacked bar chart of nofuse Stage 1/2/3 breakdown across all SSB queries and scenarios
  - Stage 1: Bitmap index access
  - Stage 2: Materialization/intermediate data movement
  - Stage 3: Query processing (joins, aggregation, candidate checking)
- Figure 5: Grouped bar comparing nofuse vs fused vs hardcode for diverse query patterns
- Analysis:
  - Stage 1 varies based on predicate count and scenario (ManyOrs higher due to ORs)
  - Stage 3 varies even more based on query complexity (Q4.x >> Q1.x)
  - Stage 2 relatively constant (materialization overhead)
  - Fusion eliminates Stage 2 entirely and keeps data in registers
  - Fusion benefit correlates with Stage 3 proportion (more computation = more benefit)
  - CandChk shows largest absolute improvement (verification fused with processing)
  - Dynamic virtual program achieves X% of hand-coded performance

#### 5. Sensitivity Analysis: Synthetic Workloads
- Figure 6: Line plots showing performance under varying selectivity and skew
- Optional Figure 7: Column width impact (16-bit vs 32-bit)
- Analysis:
  - CandChk overhead more visible at low selectivity
  - Performance stable across skew range
  - Fusion magnifies narrow-type benefits

### Key Design Decisions

1. **No Discussion section**: All discussion points integrated as inline analysis in respective sections. Strong papers don't need defensive Discussion sections.

2. **Scenarios as input conditions**: Properly frame Perfect/ManyOrs/CandChk as query-index alignment scenarios, not competing approaches. Moved explanation to beginning of Section 2.

3. **Nofuse as analytical tool**: Use nofuse breakdown to both characterize workload composition AND demonstrate what fusion eliminates. This is more valuable than just showing "before vs after."
Of course "before vs after" itself is still valuable though.

4. **Emphasis strategy for fusion**: Since fused version is single kernel (no breakdown possible), use nofuse breakdown to reveal the problem, then show fused total time as solution.

### Data Files Available

- `drawFigs/sf20.txt` - Full output for scale factor 20
- `drawFigs/sf100.txt` - Full output for scale factor 100
- `drawFigs/crystal.txt` - Crystal baseline output (SF=20 only)
- `drawFigs/drawFigs.py` - Soon-to-be Plotting script, now only with dataframe parsing

### Next Steps

0. User complains about missing ablation on the "virtual program" technique and incomprehensive analysis on skewed data, saying not all columns are used
1. Implement the plotting code in `drawFigs/drawFigs.py` to generate figures
2. Review and refine figure designs based on what patterns emerge from data
3. Write detailed analysis text once visualizations reveal specific insights
