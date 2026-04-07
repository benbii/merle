#!/usr/bin/python3
import sys
import os
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load result files into dataframes
if len(sys.argv) > 2:
  os.chdir(sys.argv[2])
a = open('sf20.txt').read().split('\n\n')
idxcreate20 = pd.read_csv(StringIO(a[0]), sep='\t')
synth20 = pd.read_csv(StringIO(a[5]), sep='\t')
ssb20 = pd.concat([
  pd.read_csv(StringIO(a[1]), sep='\t'),
  pd.read_csv(StringIO(a[2]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[3]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[4]), sep='\t').iloc[:, 1:],
  pd.read_csv('crystal.txt', sep='\t').iloc[:, 1:],
], axis=1)
# IMPORTANT: no longer SF=100; rather the SF is specified in CLI arg
a = open(f'sf{sys.argv[1]}.txt').read().split('\n\n')
idxcreate100 = pd.read_csv(StringIO(a[0]), sep='\t')
method12 = pd.read_csv(StringIO(a[5]), sep=r'\s+')
synth100 = pd.read_csv(StringIO(a[6]), sep='\t')
ssb100 = pd.concat([
  pd.read_csv(StringIO(a[1]), sep='\t'),
  pd.read_csv(StringIO(a[2]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[3]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[4]), sep='\t').iloc[:, 1:]
], axis=1)

# print('SF20:', idxcreate20, ssb20, synth20, 'SF100:', idxcreate100,
#   ssb100, synth100, sep='\n\n')

plt.rcParams.update({
  'text.usetex': True,
  'font.size': 14,          # Default font size for text
  'axes.titlesize': 15,     # Font size for axes title
  'axes.labelsize': 15,     # Font size for x and y labels
  'xtick.labelsize': 14,    # Font size for x tick labels
  'ytick.labelsize': 14,    # Font size for y tick labels
  'legend.fontsize': 14,    # Font size for legend
  'figure.titlesize': 15    # Font size for figure title
})
def gmean(arr):
  """Compute geometric mean, handling zeros gracefully."""
  arr = np.array(arr)
  return np.exp(np.mean(np.log(arr[arr > 0])))

# =============================================================================
# Figure 0: TEASER - Most Important Plot (Q3.2)
# =============================================================================
q32 = ssb100[ssb100['Case'] == 'S32'].iloc[0]
# Data for three stages
# 1. Previous Methods (WAH): index + materialize + process (aligned only)
wah_idx = q32['WAH']
wah_mat = q32['PftStg2']
wah_proc = q32['PftStg3']
# 2. Virtual Query Program (nofuse): our index + materialize + process
pft_idx = q32['PftStg1']
pft_mat = q32['PftStg2']
pft_proc = q32['PftStg3']
chk_idx = q32['ChkStg1']
chk_mat = q32['ChkStg2']
chk_proc = q32['ChkStg3']
# 3. Fused: single kernel
pft_fused = q32['Perfect']
chk_fused = q32['Candchk']
fig, ax = plt.subplots(figsize=(4, 8/3))
# Bar positions and width
positions = [0, 1.4, 1.8, 3.2, 3.6]
width = 0.35
# Colors based on updated demo
c_idx_aligned = '#1f77b4'    # aligned index access (blue)
c_idx_unaligned = '#ff7f0e'  # unaligned index access (orange)
c_mat = '#d62728'            # materialize (red/pink)
c_proc = '#98df8a'           # query based on index (green)
# 1. Previous Methods - WAH (aligned only)
ax.bar(positions[0], wah_idx, width, color=c_idx_aligned, label='Aligned index access')
ax.bar(positions[0], wah_mat, width, bottom=wah_idx, color=c_mat, label='Materialize index results')
ax.bar(positions[0], wah_proc, width, bottom=wah_idx+wah_mat, color=c_proc, label='Query based on index')
# 2. Virtual Query Program - Perfect (aligned)
ax.bar(positions[1], pft_idx, width, color=c_idx_aligned)
ax.bar(positions[1], pft_mat, width, bottom=pft_idx, color=c_mat)
ax.bar(positions[1], pft_proc, width, bottom=pft_idx+pft_mat, color=c_proc)
# 2. Virtual Query Program - CandChk (unaligned)
ax.bar(positions[2], chk_idx, width, color=c_idx_unaligned, label='Unaligned index access')
ax.bar(positions[2], chk_mat, width, bottom=chk_idx, color=c_mat)
ax.bar(positions[2], chk_proc, width, bottom=chk_idx+chk_mat, color=c_proc)
# 3. Fused - Perfect (aligned) - single color since fused
ax.bar(positions[3], pft_fused, width, color=c_idx_aligned)
# 3. Fused - CandChk (unaligned) - single color since fused
ax.bar(positions[4], chk_fused, width, color=c_idx_unaligned)
ax.annotate('Unaligned\nbins not\nsupported\npreviously', xy=(positions[0] + 0.22, 0.05), fontsize=8,
            ha='left', va='bottom', style='italic', color='#444444')
# Calculate speedups
wah_total = wah_idx + wah_mat + wah_proc
pft_nofuse = pft_idx + pft_mat + pft_proc
chk_nofuse = chk_idx + chk_mat + chk_proc
# Arrow from WAH to VQP (shorter horizontal span)
speedup1 = wah_total / pft_nofuse
pct1 = (speedup1 - 1) * 100
ax.annotate('', xy=(positions[1] - 0.1, pft_nofuse + 0.08), xytext=(positions[0] + 0.25, wah_total - 0.1),
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.8))
ax.text((positions[0] + positions[1])/2 - 0.1, (wah_total + pft_nofuse)/2 - 0.4,
        f'{pct1:.0f}\\%\nFaster', fontsize=10, color='#d62728', ha='center', fontweight='bold')
# Arrow from VQP to Fused (shorter horizontal span)
speedup2 = chk_nofuse / chk_fused
pct2 = (speedup2 - 1) * 100
ax.annotate('', xy=(positions[4] - 0.2, chk_fused), xytext=(positions[2] + 0.2, chk_nofuse - 0.1),
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.8))
ax.text((positions[2] + positions[4])/2, (chk_nofuse + chk_fused)/2 - 0.5,
        f'{pct2:.0f}\\%\nFaster', fontsize=10, color='#d62728', ha='center', fontweight='bold')
# Labels and formatting
ax.set_ylabel(f'SF {sys.argv[1]} SSB Q3.2 (ms)', fontsize=11)
ax.set_xticks([positions[0], (positions[1]+positions[2])/2, (positions[3]+positions[4])/2])
ax.set_xticklabels(['Previous\nMethods', 'Virtual Query\nProgram', 'Fuse with Query\nExecution'], fontsize=10)
ax.legend(loc='upper right', fontsize=7)
ax.set_ylim(0, 1.5)
ax.set_xlim(-0.5, 4.2)
plt.tight_layout()
plt.savefig(f'fig_teaser_sf{sys.argv[1]}.pdf', bbox_inches='tight')
plt.savefig(f'fig_teaser_sf{sys.argv[1]}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_teaser_sf{sys.argv[1]}.pdf/png")
print(f"Teaser speedups: WAH->VQP={pct1:.0f}% faster, VQP->Fused={pct2:.0f}% faster")

# =============================================================================
# Figure 0b: Method 1 vs Method 2 (inline experiment)
# =============================================================================
fig, ax = plt.subplots(figsize=(4, 2.5))
sels = method12['Seletiv'].values
m1 = method12['Method1'].values
m2 = method12['Method2'].values

ax.plot(sels, m1, 'o-', label='Method 1', color='#d62728')
ax.plot(sels, m2, 's-', label='Method 2', color='#1f77b4')
ax.set_xlabel('Selectivity')
ax.set_ylabel('Time (ms)')
ax.legend(loc='upper left', fontsize=10)
ax.set_xscale('log', base=2)
ax.set_xticks(sels)
ax.set_xticklabels([f'$\\frac{{1}}{{{int(1/s)}}}$' for s in sels])
ax.set_ylim(0, max(m1) * 1.1)

plt.tight_layout()
plt.savefig(f'fig_method12_sf{sys.argv[1]}.pdf', bbox_inches='tight')
plt.savefig(f'fig_method12_sf{sys.argv[1]}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_method12_sf{sys.argv[1]}.pdf/png")

# Print analysis
crossover_idx = np.where(m1 < m2)[0]
if len(crossover_idx) > 0:
    cross_sel = sels[crossover_idx[0]]
    print(f"Method 1 vs 2: crossover at selectivity ~{cross_sel:.2f}")
print(f"At low sel ({sels[0]:.2f}): M1={m1[0]:.2f}ms, M2={m2[0]:.2f}ms, M2 is {(m1[0]/m2[0]-1)*100:.0f}% faster")
print(f"At high sel ({sels[-1]:.2f}): M1={m1[-1]:.2f}ms, M2={m2[-1]:.2f}ms, M1 is {(m2[-1]/m1[-1]-1)*100:.0f}% slower")
# Average speedup of M2 over M1
avg_speedup = np.mean(m1 / m2)
print(f"Average speedup of Method 2 over Method 1: {(avg_speedup-1)*100:.0f}%")

# =============================================================================
# Figure 1: Overall Performance on SSB (SF=20)
# =============================================================================
queries = ssb20['Case'].tolist()
# Ours (CandChk): worst-case alignment, fully fused
ours = ssb20['Candchk']
# WAH: stacked - index access, materialization (PftStg2), table lookup (PftStg3)
wah_idx = ssb20['WAH']
wah_mat = ssb20['PftStg2']
wah_tbl = ssb20['PftStg3']
wah_total = wah_idx + wah_mat + wah_tbl
# RTScan: stacked - index access, table lookup (no materialization)
# Note: RTScan only supports Q1.x (conjunctive), others are 99.999 (invalid)
rtscan_idx = ssb20['RTScan'].where(ssb20['RTScan'] < 50, np.nan)
rtscan_tbl = ssb20['PftStg3'].where(ssb20['RTScan'] < 50, np.nan)
rtscan_total = rtscan_idx + rtscan_tbl
# MyJoin: optimized baseline without indexing
myjoin = ssb20['Join']
# Add geometric mean
queries_with_gmean = queries + ['gmean']
ours_with_gmean = list(ours) + [gmean(ours)]
wah_idx_gm = list(wah_idx) + [gmean(wah_total)]
wah_mat_gm = list(wah_mat) + [0]
wah_tbl_gm = list(wah_tbl) + [0]
rtscan_idx_gm = list(rtscan_idx) + [gmean(rtscan_total.dropna())]
rtscan_tbl_gm = list(rtscan_tbl) + [0]
myjoin_with_gmean = list(myjoin) + [gmean(myjoin)]
# Create figure
fig, ax = plt.subplots(figsize=(16, 4))
x = np.arange(len(queries_with_gmean))
width = 0.2
# Bar positions (left to right): Ours, WAH, MyJoin, RTScan
# Ours (single bar)
ax.bar(x - 1.5*width, ours_with_gmean, width, label='Ours (Worst-case)', color='#1f77b4')
# WAH (stacked: index + materialization + table lookup)
ax.bar(x - 0.5*width, wah_idx_gm, width, label='WAH (Best-case)', color='#d62728')
ax.bar(x - 0.5*width, wah_mat_gm, width, bottom=wah_idx_gm, color='#ff9896')
ax.bar(x - 0.5*width, wah_tbl_gm, width, bottom=[a+b for a,b in zip(wah_idx_gm, wah_mat_gm)], color='#98df8a')
# MyJoin (single bar)
ax.bar(x + 0.5*width, myjoin_with_gmean, width, label='MyJoin', color='#2ca02c')
# RTScan (stacked: index + table lookup)
ax.bar(x + 1.5*width, rtscan_idx_gm, width, label='RTScan', color='#ff7f0e')
ax.bar(x + 1.5*width, rtscan_tbl_gm, width, bottom=rtscan_idx_gm, color='#98df8a')
ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels([q.replace('S', 'Q') for q in queries_with_gmean], rotation=45, ha='right')
ax.legend(loc='upper left', ncol=4, fontsize=11)
# ax.set_ylim(0, max(myjoin_with_gmean) * 1.15)
# Add vertical line before GeoMean
ax.axvline(x=len(queries) - 0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_overall_sf20.pdf', bbox_inches='tight')
plt.savefig('fig_overall_sf20.png', bbox_inches='tight', dpi=150)
print("Saved fig_overall_sf20.pdf/png")
# Print speedup summary
print("=== SF=20 Speedup Analysis ===")
print(f"Geometric mean times (ms):")
print(f"  Ours:     {gmean(ours):.3f}")
print(f"  WAH:      {gmean(wah_total):.3f}")
print(f"  MyJoin:   {gmean(myjoin):.3f}")
print(f"  RTScan:   {gmean(rtscan_total.dropna()):.3f} (Q1.x only)")
print(f"Speedups over Ours (CandChk, worst-case):")
print(f"  vs WAH:     {gmean(wah_total) / gmean(ours):.2f}x")
print(f"  vs MyJoin:  {gmean(myjoin) / gmean(ours):.2f}x\n")

# =============================================================================
# Figure 1b: MyJoin vs Crystal (SF=20) - Database Integration inline experiment
# =============================================================================
queries = ssb20['Case'].tolist()
myjoin20 = ssb20['Join']
crystal = ssb20['Crystal']

# Add geometric mean
queries_with_gmean = queries + ['gmean']
myjoin_gm = list(myjoin20) + [gmean(myjoin20)]
crystal_gm = list(crystal) + [gmean(crystal)]

fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(queries_with_gmean))
width = 0.35

ax.bar(x - width/2, myjoin_gm, width, label='Our Join Baseline', color='#1f77b4')
ax.bar(x + width/2, crystal_gm, width, label='Crystal', color='#ff7f0e')

ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels([q.replace('S', 'Q') for q in queries_with_gmean], rotation=45, ha='right')
ax.legend(loc='upper left', fontsize=11)
ax.set_ylim(0, max(crystal_gm) * 1.15)
ax.axvline(x=len(queries) - 0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig_myjoin_crystal.pdf', bbox_inches='tight')
plt.savefig('fig_myjoin_crystal.png', bbox_inches='tight', dpi=150)
print("Saved fig_myjoin_crystal.pdf/png")

# Print speedup
print("=== MyJoin vs Crystal (SF=20) ===")
print(f"Geometric mean: MyJoin={gmean(myjoin20):.3f}ms, Crystal={gmean(crystal):.3f}ms")
print(f"Speedup: {gmean(crystal) / gmean(myjoin20):.2f}x\n")

# =============================================================================
# Figure 2: Overall Performance on SSB (SF=100) - No RTScan, No Crystal
# =============================================================================
queries = ssb100['Case'].tolist()
# Ours (CandChk): worst-case alignment, fully fused
ours = ssb100['Candchk']
# WAH: stacked - index access, materialization, table lookup
wah_idx = ssb100['WAH']
wah_mat = ssb100['PftStg2']
wah_tbl = ssb100['PftStg3']
wah_total = wah_idx + wah_mat + wah_tbl
# MyJoin: optimized baseline without indexing
myjoin = ssb100['Join']
# Add geometric mean
queries_with_gmean = queries + ['gmean']
ours_with_gmean = list(ours) + [gmean(ours)]
wah_idx_gm = list(wah_idx) + [gmean(wah_total)]
wah_mat_gm = list(wah_mat) + [0]
wah_tbl_gm = list(wah_tbl) + [0]
myjoin_with_gmean = list(myjoin) + [gmean(myjoin)]
# Create figure
fig, ax = plt.subplots(figsize=(16, 4))
x = np.arange(len(queries_with_gmean))
width = 0.25
# Bar positions (left to right): Ours, WAH, MyJoin
# Ours (single bar)
ax.bar(x - width, ours_with_gmean, width, label='Ours (Worst-case)', color='#1f77b4')
# WAH (stacked: index + materialization + table lookup)
ax.bar(x, wah_idx_gm, width, label='WAH (Best-case)', color='#d62728')
ax.bar(x, wah_mat_gm, width, bottom=wah_idx_gm, color='#ff9896')
ax.bar(x, wah_tbl_gm, width, bottom=[a+b for a,b in zip(wah_idx_gm, wah_mat_gm)], color='#98df8a')
# MyJoin (single bar)
ax.bar(x + width, myjoin_with_gmean, width, label='MyJoin', color='#2ca02c')
ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels([q.replace('S', 'Q') for q in queries_with_gmean], rotation=45, ha='right')
ax.legend(loc='upper left', ncol=4, fontsize=11)
ax.set_ylim(0, max(myjoin_with_gmean) * 1.15)
# Add vertical line before GeoMean
ax.axvline(x=len(queries) - 0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'fig_overall_sf{sys.argv[1]}.pdf', bbox_inches='tight')
plt.savefig(f'fig_overall_sf{sys.argv[1]}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_overall_sf{sys.argv[1]}.pdf/png")
# Print speedup summary
print(f"=== SF={sys.argv[1]} Speedup Analysis ===")
print(f"Geometric mean times (ms):")
print(f"  Ours:     {gmean(ours):.3f}")
print(f"  WAH:      {gmean(wah_total):.3f}")
print(f"  MyJoin:   {gmean(myjoin):.3f}")
print(f"Speedups over Ours (CandChk, worst-case):")
print(f"  vs WAH:     {gmean(wah_total) / gmean(ours):.2f}x")
print(f"  vs MyJoin:  {gmean(myjoin) / gmean(ours):.2f}x\n")

# =============================================================================
# Figure 3: Alignment Scenarios (SF=100) - Small figure with selected queries
# =============================================================================
# Select representative queries: Q1.1, Q2.1, Q3.1, Q4.1, Q4.2, Q4.3
sel_queries_align = ['S11', 'S21', 'S31', 'S41', 'S42', 'S43']
sel_mask_align = ssb100['Case'].isin(sel_queries_align)
sel_ssb_align = ssb100[sel_mask_align].copy()

perfect = sel_ssb_align['Perfect']
manyors = sel_ssb_align['ManyOrs']
candchk = sel_ssb_align['Candchk']
myjoin = sel_ssb_align['Join']

# Add geometric mean (of ALL 13 queries, not just selected)
labels = [q.replace('S', 'Q') for q in sel_queries_align] + ['weighted\naverage']
perfect_vals = list(perfect) + [gmean(ssb100['Perfect'])]
manyors_vals = list(manyors) + [gmean(ssb100['ManyOrs'])]
candchk_vals = list(candchk) + [gmean(ssb100['Candchk'])]
myjoin_vals = list(myjoin) + [gmean(ssb100['Join'])]

# Create figure
fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(labels))
width = 0.2

# Bar positions: Perfect, ManyOrs, CandChk, Join
ax.bar(x - 1.5*width, perfect_vals, width, label='Perfect', color='#1f77b4')
ax.bar(x - 0.5*width, manyors_vals, width, label='ManyOrs', color='#aec7e8')
ax.bar(x + 0.5*width, candchk_vals, width, label='CandChk', color='#ff7f0e')
ax.bar(x + 1.5*width, myjoin_vals, width, label='Join', color='#2ca02c')

ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', ncol=4, fontsize=9)
ax.set_ylim(0, max(myjoin_vals) * 1.15)
ax.axvline(x=len(sel_queries_align) - 0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'fig_alignment_sf{sys.argv[1]}.pdf', bbox_inches='tight')
plt.savefig(f'fig_alignment_sf{sys.argv[1]}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_alignment_sf{sys.argv[1]}.pdf/png")

# Print analysis (still use all 13 queries)
queries = ssb100['Case'].tolist()
perfect_all = ssb100['Perfect']
manyors_all = ssb100['ManyOrs']
candchk_all = ssb100['Candchk']
myjoin_all = ssb100['Join']
print(f"=== Alignment Scenario Analysis (SF={sys.argv[1]}) ===")
print(f"Geometric mean times (ms):")
print(f"  Perfect:  {gmean(perfect_all):.3f}")
print(f"  ManyOrs:  {gmean(manyors_all):.3f}")
print(f"  CandChk:  {gmean(candchk_all):.3f}")
print(f"  Join:     {gmean(myjoin_all):.3f}")
print(f"Overhead vs ManyOrs (pivot):")
print(f"  Perfect:  {(1 - gmean(perfect_all) / gmean(manyors_all)) * 100:.1f}% faster")
print(f"  CandChk:  {(gmean(candchk_all) / gmean(manyors_all) - 1) * 100:.1f}% slower")
print(f"All scenarios vs Join:")
print(f"  Perfect:  {gmean(myjoin_all) / gmean(perfect_all):.2f}x faster")
print(f"  ManyOrs:  {gmean(myjoin_all) / gmean(manyors_all):.2f}x faster")
print(f"  CandChk:  {gmean(myjoin_all) / gmean(candchk_all):.2f}x faster")
print(f"Per-query CandChk overhead vs ManyOrs:")
for i, q in enumerate(queries):
    overhead = (candchk_all.iloc[i] / manyors_all.iloc[i] - 1) * 100
    print(f"  {q}: {overhead:.1f}%")
print()

# =============================================================================
# Figure 4: Synthetic Workloads - Selectivity and Skewness Sensitivity
# =============================================================================
# Filter to 32-bit data (has WAH baseline)
synth32 = synth100[synth100['BitW'] == 32].copy()
synth16 = synth100[synth100['BitW'] == 16].copy()

# NOTE: The WAH microbenchmark measures index-side bitmap processing. To make
# the synthetic comparison consistent with SSB figures (where WAH must
# materialize and then perform the downstream join/aggregation), we add the
# remaining two stages back: materialization (PftStg2) and downstream processing
# (PftStg3).
synth32['WAH_total'] = synth32['WAH'] + synth32['PftStg2'] + synth32['PftStg3']

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# Left plot: Selectivity sensitivity (fixed skew=1.04, 32-bit)
ax = axes[0]
skew_fixed = synth32[synth32['Skew'] == 1.04]
sels = skew_fixed['Seletiv'].values
ax.plot(sels, skew_fixed['Perfect'], 'o-', label='Perfect', color='#1f77b4')
ax.plot(sels, skew_fixed['ManyOrs'], 's-', label='ManyOrs', color='#aec7e8')
ax.plot(sels, skew_fixed['CandChk'], '^-', label='CandChk', color='#ff7f0e')
ax.plot(sels, skew_fixed['WAH_total'], 'D-', label='WAH(Perfect)', color='#d62728')
ax.plot(sels, skew_fixed['Join'], 'x-', label='Join', color='#2ca02c', alpha=0.7)
ax.set_xlabel('Selectivity (skew=1.04, 32-bit)')
ax.set_ylabel('Time (ms)')
ax.set_xscale('log', base=2)
ax.set_xticks(sels)
ax.set_xticklabels([f'1/{int(1/s)}' for s in sels])
ax.legend(loc='center left', fontsize=10)

# Right plot: Skewness sensitivity (fixed selectivity=0.03125=1/32, 32-bit)
ax = axes[1]
sel_fixed = synth32[synth32['Seletiv'] == 0.03125]
skews = sel_fixed['Skew'].values
ax.plot(skews, sel_fixed['Perfect'], 'o-', label='Perfect', color='#1f77b4')
ax.plot(skews, sel_fixed['ManyOrs'], 's-', label='ManyOrs', color='#aec7e8')
ax.plot(skews, sel_fixed['CandChk'], '^-', label='CandChk', color='#ff7f0e')
ax.plot(skews, sel_fixed['WAH_total'], 'D-', label='WAH(Perfect)', color='#d62728')
ax.plot(skews, sel_fixed['Join'], 'x-', label='Join', color='#2ca02c', alpha=0.7)
ax.set_xlabel('Zipfian Skewness (sel=1/32, 32-bit)')
ax.set_ylabel('Time (ms)')
ax.set_xticks(skews)
ax.legend(loc='center right', fontsize=10)

plt.tight_layout()
plt.savefig(f'fig_synth_sf{sys.argv[1]}.pdf', bbox_inches='tight')
plt.savefig(f'fig_synth_sf{sys.argv[1]}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_synth_sf{sys.argv[1]}.pdf/png")

# Print analysis
print("=== Synthetic Workload Analysis ===")
print("Selectivity sensitivity (skew=1.04, 32-bit):")
for sel in sels:
  row = skew_fixed[skew_fixed['Seletiv'] == sel].iloc[0]
  print(f"  sel=1/{int(1/sel):3d}: Perfect={row['Perfect']:.2f}, CandChk={row['CandChk']:.2f}, WAH={row['WAH_total']:.2f}, CandChk/WAH={row['CandChk']/row['WAH_total']:.2f}x")

print("\nSkewness sensitivity (sel=1/32, 32-bit):")
for skew in skews:
  row = sel_fixed[sel_fixed['Skew'] == skew].iloc[0]
  print(f"  skew={skew:.2f}: Perfect={row['Perfect']:.2f}, CandChk={row['CandChk']:.2f}, WAH={row['WAH_total']:.2f}")

# Compare 16-bit vs 32-bit trends
print("\n16-bit vs 32-bit comparison (skew=1.04):")
for sel in sels:
  r16 = synth16[(synth16['Skew'] == 1.04) & (synth16['Seletiv'] == sel)].iloc[0]
  r32 = synth32[(synth32['Skew'] == 1.04) & (synth32['Seletiv'] == sel)].iloc[0]
  print(f"  sel=1/{int(1/sel):3d}: 16-bit CandChk={r16['CandChk']:.2f}, 32-bit CandChk={r32['CandChk']:.2f}, 32-bit WAH={r32['WAH_total']:.2f}")

# =============================================================================
# Figure 5: Virtual Program Overhead (Ablation)
# =============================================================================
# Select representative queries: Q1.1, Q2.1, Q3.1-Q3.4, Q4.1
selected_queries = ['S11', 'S21', 'S31', 'S32', 'S33', 'S34', 'S41']
sel_mask = ssb100['Case'].isin(selected_queries)
sel_ssb = ssb100[sel_mask].copy()

# Calculate overhead percentages
pft_overhead = (sel_ssb['Perfect'] / sel_ssb['PftNprg'] - 1) * 100
a = pft_overhead < -2; pft_overhead[a] = -pft_overhead[a]
ors_overhead = (sel_ssb['ManyOrs'] / sel_ssb['OrsNprg'] - 1) * 100
a = ors_overhead < -2; ors_overhead[a] = -ors_overhead[a]
chk_overhead = (sel_ssb['Candchk'] / sel_ssb['ChkNprg'] - 1) * 100
a = chk_overhead < -2; chk_overhead[a] = -chk_overhead[a]
del a

# Time-weighted mean across ALL 13 queries
pft_weighted = (ssb100['Perfect'].sum() - ssb100['PftNprg'].sum()) / ssb100['PftNprg'].sum() * 100
ors_weighted = (ssb100['ManyOrs'].sum() - ssb100['OrsNprg'].sum()) / ssb100['OrsNprg'].sum() * 100
chk_weighted = (ssb100['Candchk'].sum() - ssb100['ChkNprg'].sum()) / ssb100['ChkNprg'].sum() * 100

# Append weighted average
pft_vals = list(pft_overhead) + [pft_weighted if pft_weighted >= -2 else -pft_weighted]
ors_vals = list(ors_overhead) + [ors_weighted if ors_weighted >= -2 else -ors_weighted]
chk_vals = list(chk_overhead) + [chk_weighted if chk_weighted >= -2 else -chk_weighted]
labels = [q.replace('S', 'Q') for q in selected_queries] + ['weighted\naverage']

fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(labels))
width = 0.25

ax.bar(x - width, pft_vals, width, label='Perfect', color='#1f77b4')
ax.bar(x, ors_vals, width, label='ManyOrs', color='#aec7e8')
ax.bar(x + width, chk_vals, width, label='CandChk', color='#ff7f0e')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=len(selected_queries) - 0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Overhead (\\%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(f'fig_ablation_vprg_sf{sys.argv[1]}.pdf', bbox_inches='tight')
plt.savefig(f'fig_ablation_vprg_sf{sys.argv[1]}.png', bbox_inches='tight', dpi=150)
print(f"\nSaved fig_ablation_vprg_sf{sys.argv[1]}.pdf/png")

# Print analysis
print(f"=== Virtual Program Overhead Analysis (SF={sys.argv[1]}) ===")
print(f"Time-weighted mean overhead (all 13 queries):")
print(f"  Perfect: {pft_weighted:.1f}%")
print(f"  ManyOrs: {ors_weighted:.1f}%")
print(f"  CandChk: {chk_weighted:.1f}%")
print(f"Per-query overhead:")
for i, q in enumerate(selected_queries):
    print(f"  {q}: Perfect={pft_vals[i]:.1f}%, ManyOrs={ors_vals[i]:.1f}%, CandChk={chk_vals[i]:.1f}%")

# =============================================================================
# Figure 6: Fusion Effectiveness (Ablation)
# =============================================================================
# Calculate nofuse totals for SSB
ssb100['PftNofuse'] = ssb100['PftStg1'] + ssb100['PftStg2'] + ssb100['PftStg3']
ssb100['OrsNofuse'] = ssb100['OrsStg1'] + ssb100['OrsStg2'] + ssb100['OrsStg3']
ssb100['ChkNofuse'] = ssb100['ChkStg1'] + ssb100['ChkStg2'] + ssb100['ChkStg3']

# Select representative SSB queries
sel_queries_fusion = ['S11', 'S21', 'S31', 'S34', 'S41']
sel_mask_fusion = ssb100['Case'].isin(sel_queries_fusion)
sel_ssb_fusion = ssb100[sel_mask_fusion].copy()

fig, axes = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'width_ratios': [3, 2]})

# Left: SSB stacked bar (nofuse breakdown) vs fused
ax = axes[0]
x = np.arange(len(sel_queries_fusion))
width = 0.2

# Stage colors (same for both Perfect and CandChk)
c_idx = '#1f77b4'      # index (stage 1)
c_mat = '#aec7e8'      # materialize (stage 2)
c_proc = '#98df8a'     # process (stage 3)

# Perfect: nofuse stacked vs fused
ax.bar(x - width, sel_ssb_fusion['PftStg1'], width, label='Index', color=c_idx)
ax.bar(x - width, sel_ssb_fusion['PftStg2'], width, bottom=sel_ssb_fusion['PftStg1'], label='Materialize', color=c_mat)
ax.bar(x - width, sel_ssb_fusion['PftStg3'], width, bottom=sel_ssb_fusion['PftStg1']+sel_ssb_fusion['PftStg2'], label='Process', color=c_proc)
ax.bar(x, sel_ssb_fusion['Perfect'], width, label='Perfect (fused)', color='#2ca02c', edgecolor='black', linewidth=0.5)

# CandChk: nofuse stacked vs fused (same stage colors)
ax.bar(x + width, sel_ssb_fusion['ChkStg1'], width, color=c_idx)
ax.bar(x + width, sel_ssb_fusion['ChkStg2'], width, bottom=sel_ssb_fusion['ChkStg1'], color=c_mat)
ax.bar(x + width, sel_ssb_fusion['ChkStg3'], width, bottom=sel_ssb_fusion['ChkStg1']+sel_ssb_fusion['ChkStg2'], color=c_proc)
ax.bar(x + 2*width, sel_ssb_fusion['Candchk'], width, label='CandChk (fused)', color='#d62728', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Time (ms)')
ax.set_xticks(x + 0.5*width)
ax.set_xticklabels([q.replace('S', 'Q') for q in sel_queries_fusion])
ax.legend(loc='upper left', fontsize=8, ncol=2)

# Right: Synthetic - fusion speedup vs selectivity
ax = axes[1]
synth32_fusion = synth100[synth100['BitW'] == 32].copy()
skew_fixed_fusion = synth32_fusion[synth32_fusion['Skew'] == 1.04].copy()
skew_fixed_fusion['PftNofuse'] = skew_fixed_fusion['PftStg1'] + skew_fixed_fusion['PftStg2'] + skew_fixed_fusion['PftStg3']
skew_fixed_fusion['OrsNofuse'] = skew_fixed_fusion['OrsStg1'] + skew_fixed_fusion['OrsStg2'] + skew_fixed_fusion['OrsStg3']
skew_fixed_fusion['ChkNofuse'] = skew_fixed_fusion['ChkStg1'] + skew_fixed_fusion['ChkStg2'] + skew_fixed_fusion['ChkStg3']
skew_fixed_fusion['PftSpeedup'] = skew_fixed_fusion['PftNofuse'] / skew_fixed_fusion['Perfect']
skew_fixed_fusion['OrsSpeedup'] = skew_fixed_fusion['OrsNofuse'] / skew_fixed_fusion['ManyOrs']
skew_fixed_fusion['ChkSpeedup'] = skew_fixed_fusion['ChkNofuse'] / skew_fixed_fusion['CandChk']

sels_fusion = skew_fixed_fusion['Seletiv'].values
ax.plot(sels_fusion, skew_fixed_fusion['PftSpeedup'], 'o-', label='Perfect', color='#2ca02c')
ax.plot(sels_fusion, skew_fixed_fusion['OrsSpeedup'], 's-', label='ManyOrs', color='#aec7e8')
ax.plot(sels_fusion, skew_fixed_fusion['ChkSpeedup'], '^-', label='CandChk', color='#d62728')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Selectivity')
ax.set_ylabel('Speedup')
ax.set_xscale('log', base=2)
ax.set_xticks(sels_fusion)
ax.set_xticklabels([f'$\\frac{{1}}{{{int(1/s)}}}$' for s in sels_fusion])
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0.9, 1.8)

plt.tight_layout()
plt.savefig(f'fig_ablation_fusion_sf{sys.argv[1]}.pdf', bbox_inches='tight')
plt.savefig(f'fig_ablation_fusion_sf{sys.argv[1]}.png', bbox_inches='tight', dpi=150)
print(f"\nSaved fig_ablation_fusion_sf{sys.argv[1]}.pdf/png")

# Print analysis
print("=== Fusion Effectiveness Analysis ===")
print("SSB (all 13 queries):")
pft_speedup = ssb100['PftNofuse'].sum() / ssb100['Perfect'].sum()
ors_speedup = ssb100['OrsNofuse'].sum() / ssb100['ManyOrs'].sum()
chk_speedup = ssb100['ChkNofuse'].sum() / ssb100['Candchk'].sum()
print(f"  Perfect speedup: {pft_speedup:.2f}x")
print(f"  ManyOrs speedup: {ors_speedup:.2f}x")
print(f"  CandChk speedup: {chk_speedup:.2f}x")
print(f"Stage 2 (eliminated by fusion):")
print(f"  Perfect: {ssb100['PftStg2'].sum():.2f}ms ({ssb100['PftStg2'].sum()/ssb100['PftNofuse'].sum()*100:.1f}% of nofuse)")
print(f"  CandChk: {ssb100['ChkStg2'].sum():.2f}ms ({ssb100['ChkStg2'].sum()/ssb100['ChkNofuse'].sum()*100:.1f}% of nofuse)")
print("Synthetic (skew=1.04):")
for sel in sels_fusion:
    row = skew_fixed_fusion[skew_fixed_fusion['Seletiv'] == sel].iloc[0]
    print(f"  sel=1/{int(1/sel):3d}: Perfect={row['PftSpeedup']:.2f}x, CandChk={row['ChkSpeedup']:.2f}x")

# =============================================================================
# Figure 7: Index Creation and Memory Usage (SF=20)
# =============================================================================
# Get per-column data (exclude Total and RTScan rows)
idx_cols = idxcreate20[~idxcreate20['Column'].isin(['Total', 'RTScanSieve', 'RTScanRays'])].copy()
idx_total = idxcreate20[idxcreate20['Column'] == 'Total'].iloc[0]
idx_rtscan_siev = idxcreate20[idxcreate20['Column'] == 'RTScanSieve'].iloc[0]
idx_rtscan_rays = idxcreate20[idxcreate20['Column'] == 'RTScanRays'].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Left: Construction time (log scale due to RTScan being huge)
ax = axes[0]
methods = ['Ours', 'WAH', 'RTScan']
times = [idx_total['My(ms)'], idx_total['WAH(ms)'],
         idx_rtscan_siev['My(MB)'] + idx_rtscan_rays['My(MB)']]
colors = ['#1f77b4', '#d62728', '#ff7f0e']
bars = ax.bar(methods, times, color=colors)
ax.set_ylabel('Construction Time (ms)')
ax.set_yscale('log')
# Add value labels (inside bars for RTScan to avoid overflow)
for i, (bar, val) in enumerate(zip(bars, times)):
    if i == 2:  # RTScan - place inside
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, #*.5 due to log
                f'{val:.0f}', ha='center', va='center', fontsize=16, color='white', fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                f'{val:.0f}', ha='center', va='bottom', fontsize=16)

# Right: Memory usage
ax = axes[1]
memory = [idx_total['My(MB)'], idx_total['WAH(MB)'],
          idx_rtscan_siev['WAH(MB)'] + idx_rtscan_rays['WAH(MB)']]
bars = ax.bar(methods, memory, color=colors)
ax.set_ylabel('Memory Usage (MB)')
# Add value labels (inside bar for RTScan)
for i, (bar, val) in enumerate(zip(bars, memory)):
    if i == 2:  # RTScan - place inside
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 300,
                f'{val:.0f}', ha='center', va='top', fontsize=16, color='white', fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:.0f}', ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.savefig('fig_creation.pdf', bbox_inches='tight')
plt.savefig('fig_creation.png', bbox_inches='tight', dpi=150)
print("\nSaved fig_creation.pdf/png")

# Print analysis
print("=== Index Creation Analysis (SF=20) ===")
print(f"Construction time:")
print(f"  Ours: {idx_total['My(ms)']:.1f}ms")
print(f"  WAH:  {idx_total['WAH(ms)']:.1f}ms ({idx_total['WAH(ms)']/idx_total['My(ms)']:.1f}x slower)")
# RTScan build time is split into sieve + rays and recorded in the `My(MB)`
# column in our input table (see `sf20.txt` / `sf140.txt`).
rtscan_time = idx_rtscan_siev['My(MB)'] + idx_rtscan_rays['My(MB)']
print(f"  RTScan: {rtscan_time:.1f}ms ({rtscan_time/idx_total['My(ms)']:.0f}x slower)")
print(f"Memory usage:")
print(f"  Ours: {idx_total['My(MB)']:.1f}MB")
print(f"  WAH:  {idx_total['WAH(MB)']:.1f}MB ({(idx_total['My(MB)']/idx_total['WAH(MB)']-1)*100:.0f}% more for Ours)")
rtscan_mem = idx_rtscan_siev['WAH(MB)'] + idx_rtscan_rays['WAH(MB)']
print(f"  RTScan: {rtscan_mem:.1f}MB ({rtscan_mem/idx_total['My(MB)']:.1f}x more than Ours)")


