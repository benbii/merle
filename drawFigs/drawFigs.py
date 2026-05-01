#!/usr/bin/python3
import sys
import os
from io import StringIO
from fractions import Fraction
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sel_label(s):
    """Format a selectivity float as a LaTeX fraction, e.g. 0.4 -> '$\\frac{2}{5}$'."""
    f = Fraction(s).limit_denominator(512)
    return f'$\\frac{{{f.numerator}}}{{{f.denominator}}}$'

# =============================================================================
# Data loading
# New format (all SFs): 5 blocks
#   block 0: WAH/RTScan baselines (per-query, index-only)
#   block 1: index creation / memory (per-column, Sparse+Dense+Balanced totals)
#   block 2: SSB main results — BinType in {Sparse, Medium, Dense, Perfect, AllAlig, MisAlig}
#             columns: Case BinType Join Ours Basefus NfuStg1 PrgStg1 Stg2 Stg3 Dedicat
#             Join is non-zero only on Sparse rows
#   block 3: Method1 vs Method2 micro-benchmark
#   block 4: synthetic workloads
# SF=20 additionally has crystal.txt in the same directory.
# =============================================================================
if len(sys.argv) > 2:
    os.chdir(sys.argv[2])

def load_sf(fname, has_crystal=False):
    a = open(fname).read().split('\n\n')
    wah_rtscan  = pd.read_csv(StringIO(a[0]), sep='\t')
    idxcreate   = pd.read_csv(StringIO(a[1]), sep='\t')
    ssb_all     = pd.read_csv(StringIO(a[2]), sep='\t')
    method12    = pd.read_csv(StringIO(a[3]), sep=r'\s+')
    synth       = pd.read_csv(StringIO(a[4]), sep='\t')
    # split SSB block into fixed-layout and alignment-scenario subsets
    ssb_fixed = ssb_all[ssb_all['BinType'].isin(['Sparse', 'Medium', 'Dense'])].copy()
    ssb_align = ssb_all[ssb_all['BinType'].isin(['Perfect', 'AllAlig', 'MisAlig'])].copy()
    crystal = None
    if has_crystal:
        crystal = pd.read_csv('crystal.txt', sep='\t')
    return wah_rtscan, idxcreate, ssb_fixed, ssb_align, method12, synth, crystal

wah20, idxcreate20, ssb20_fixed, ssb20_align, method12_20, synth20, crystal20 = \
    load_sf('sf20.txt', has_crystal=True)
sf = sys.argv[1]
wah_sf, idxcreate_sf, ssb_fixed, ssb_align, method12_sf, synth_sf, _ = \
    load_sf(f'sf{sf}.txt')

# Convenience: per-query WAH index-only times (used to build stacked WAH bars)
# WAH stacked = WAH index + Stg2 (materialize) + Stg3 (process) from Perfect rows
def wah_stacked(wah_rtscan, ssb_align):
    pft = ssb_align[ssb_align['BinType'] == 'Perfect'].set_index('Case')
    w = wah_rtscan.set_index('Case')
    merged = pft.join(w[['WAH']], how='left')
    return merged  # has WAH, Stg2, Stg3

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 15,
})

def gmean(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[arr > 0]
    return np.exp(np.mean(np.log(arr))) if len(arr) > 0 else np.nan

QUERIES = ['SSB11','SSB12','SSB13','SSB21','SSB22','SSB23',
           'SSB31','SSB32','SSB33','SSB34','SSB41','SSB42','SSB43']
QLABELS = [q.replace('SSB','Q').replace('Q1','Q1.').replace('Q2','Q2.')
            .replace('Q3','Q3.').replace('Q4','Q4.') for q in QUERIES]
# e.g. SSB11 -> Q1.1
QLABELS = ['Q'+q[3]+'.'+q[4] for q in QUERIES]

# =============================================================================
# Figure 0: TEASER - SSB Q3.2 progression (uses sfN data)
# =============================================================================
q32_sparse = ssb_fixed[(ssb_fixed['Case']=='SSB32') & (ssb_fixed['BinType']=='Sparse')].iloc[0]
q32_align  = ssb_align[(ssb_align['Case']=='SSB32')]
q32_pft    = q32_align[q32_align['BinType']=='Perfect'].iloc[0]
q32_mis    = q32_align[q32_align['BinType']=='MisAlig'].iloc[0]
wah_sf_idx = wah_sf[wah_sf['Case']=='SSB32'].iloc[0]

# WAH (perfect alignment): index-only + materialize + process
wah_idx  = wah_sf_idx['WAH']
wah_mat  = q32_pft['Stg2']
wah_proc = q32_pft['Stg3']
wah_total = wah_idx + wah_mat + wah_proc

# VQP no-fuse (MisAlig): PrgStg1 + Stg2 + Stg3
vqp_idx  = q32_mis['PrgStg1']
vqp_mat  = q32_mis['Stg2']
vqp_proc = q32_mis['Stg3']
vqp_total = vqp_idx + vqp_mat + vqp_proc

# Fused (MisAlig): Ours
fused = q32_mis['Ours']

c_idx_aligned   = '#1f77b4'
c_idx_unaligned = '#ff7f0e'
c_mat           = '#d62728'
c_proc          = '#98df8a'

positions = [0, 1.4, 1.8, 3.2, 3.6]
width = 0.35

fig, ax = plt.subplots(figsize=(4, 8/3))

# WAH bar (aligned only)
ax.bar(positions[0], wah_idx,  width, color=c_idx_aligned,   label='Aligned index access')
ax.bar(positions[0], wah_mat,  width, bottom=wah_idx,         color=c_mat,  label='Materialize index results')
ax.bar(positions[0], wah_proc, width, bottom=wah_idx+wah_mat, color=c_proc, label='Query based on index')

# VQP no-fuse Perfect (aligned)
q32_pft_vqp = q32_pft['PrgStg1']
ax.bar(positions[1], q32_pft_vqp, width, color=c_idx_aligned)
ax.bar(positions[1], q32_pft['Stg2'], width, bottom=q32_pft_vqp, color=c_mat)
ax.bar(positions[1], q32_pft['Stg3'], width, bottom=q32_pft_vqp+q32_pft['Stg2'], color=c_proc)

# VQP no-fuse MisAlig (unaligned)
ax.bar(positions[2], vqp_idx,  width, color=c_idx_unaligned, label='Unaligned index access')
ax.bar(positions[2], vqp_mat,  width, bottom=vqp_idx,         color=c_mat)
ax.bar(positions[2], vqp_proc, width, bottom=vqp_idx+vqp_mat, color=c_proc)

# Fused Perfect
ax.bar(positions[3], q32_pft['Ours'], width, color=c_idx_aligned)
# Fused MisAlig
ax.bar(positions[4], fused, width, color=c_idx_unaligned)

ax.annotate('Unaligned\nbins not\nsupported\npreviously', xy=(positions[0]+0.22, 0.05),
            fontsize=8, ha='left', va='bottom', style='italic', color='#444444')

pct1 = (wah_total / vqp_total - 1) * 100
pct2 = (vqp_total / fused - 1) * 100
ax.annotate('', xy=(positions[1]-0.1, vqp_total+0.02),
            xytext=(positions[0]+0.25, wah_total-0.02),
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.8))
ax.text((positions[0]+positions[1])/2-0.1, (wah_total+vqp_total)/3,
        f'{pct1:.0f}\\%\nFaster', fontsize=10, color='#d62728', ha='center', fontweight='bold')
ax.annotate('', xy=(positions[4]-0.2, fused),
            xytext=(positions[2]+0.2, vqp_total-0.02),
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.8))
ax.text((positions[2]+positions[4])/2, (vqp_total+fused)/3,
        f'{pct2:.0f}\\%\nFaster', fontsize=10, color='#d62728', ha='center', fontweight='bold')

ax.set_ylabel(f'SF {sf} SSB Q3.2 (ms)', fontsize=11)
ax.set_xticks([positions[0], (positions[1]+positions[2])/2, (positions[3]+positions[4])/2])
ax.set_xticklabels(['Previous\nMethods', 'Virtual Query\nProgram', 'Fuse with Query\nExecution'], fontsize=10)
ax.legend(loc='upper right', fontsize=7)
ax.set_xlim(-0.5, 4.2)
plt.tight_layout()
plt.savefig(f'fig_teaser_sf{sf}.pdf', bbox_inches='tight')
plt.savefig(f'fig_teaser_sf{sf}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_teaser_sf{sf}.pdf/png")
print(f"Teaser: WAH->VQP={pct1:.0f}% faster, VQP->Fused={pct2:.0f}% faster")

# =============================================================================
# Figure 0b: Method 1 vs Method 2 (uses sfN data)
# =============================================================================
fig, ax = plt.subplots(figsize=(4, 2.5))
sels = method12_sf['Seletiv'].values
m1   = method12_sf['Method1'].values
m2   = method12_sf['Method2'].values

ax.plot(sels, m1, 'o-', label='Method 1', color='#d62728')
ax.plot(sels, m2, 's-', label='Method 2', color='#1f77b4')
ax.set_xlabel('Selectivity')
ax.set_ylabel('Time (ms)')
ax.legend(loc='upper left', fontsize=10)
ax.set_xscale('log', base=2)
ax.set_xticks(sels)
ax.set_xticklabels([sel_label(s) for s in sels])
ax.set_ylim(0, max(m1) * 1.1)
plt.tight_layout()
plt.savefig(f'fig_method12_sf{sf}.pdf', bbox_inches='tight')
plt.savefig(f'fig_method12_sf{sf}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_method12_sf{sf}.pdf/png")

# =============================================================================
# Figure 1b: MyJoin vs Crystal (SF=20 only)
# =============================================================================
join20   = ssb20_fixed[ssb20_fixed['BinType']=='Sparse'].set_index('Case')['Join']
crystal  = crystal20.set_index('Case')['Crystal']
cases20  = QUERIES
jvals    = [join20[c] for c in cases20] + [gmean([join20[c] for c in cases20])]
cvals    = [crystal[c] for c in cases20] + [gmean([crystal[c] for c in cases20])]
xlabels  = QLABELS + ['gmean']

fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(xlabels))
w = 0.35
ax.bar(x - w/2, jvals, w, label='Our Join Baseline', color='#1f77b4')
ax.bar(x + w/2, cvals, w, label='Crystal',           color='#ff7f0e')
ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=45, ha='right')
ax.legend(loc='upper left', fontsize=11)
ax.set_ylim(0, max(cvals) * 1.15)
ax.axvline(x=len(cases20) - 0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_myjoin_crystal.pdf', bbox_inches='tight')
plt.savefig('fig_myjoin_crystal.png', bbox_inches='tight', dpi=150)
print("Saved fig_myjoin_crystal.pdf/png")
print(f"MyJoin vs Crystal gmean: {gmean([join20[c] for c in cases20]):.3f} vs "
      f"{gmean([crystal[c] for c in cases20]):.3f} ms, "
      f"speedup={gmean([crystal[c] for c in cases20])/gmean([join20[c] for c in cases20]):.2f}x")

# =============================================================================
# Helper: build overall-performance data for one SF dataset
# Returns per-query arrays ready for plotting.
# "Ours" = Sparse Ours (worst-case practical layout).
# WAH stacked = WAH index-only + Perfect Stg2 (materialize) + Perfect Stg3 (process).
# RTScan = RTScan index-only + Perfect Stg3 (process); NaN for unsupported queries.
# =============================================================================
def build_overall(ssb_fixed, ssb_align, wah_rtscan):
    sparse = ssb_fixed[ssb_fixed['BinType']=='Sparse'].set_index('Case')
    medium = ssb_fixed[ssb_fixed['BinType']=='Medium'].set_index('Case')
    dense  = ssb_fixed[ssb_fixed['BinType']=='Dense'].set_index('Case')
    pft    = ssb_align[ssb_align['BinType']=='Perfect'].set_index('Case')
    wah_w  = wah_rtscan.set_index('Case')

    ours_sp = np.array([sparse.loc[c,'Ours']  for c in QUERIES])
    ours_md = np.array([medium.loc[c,'Ours']  for c in QUERIES])
    ours_dn = np.array([dense.loc[c,'Ours']   for c in QUERIES])
    join    = np.array([sparse.loc[c,'Join']  for c in QUERIES])

    wah_idx  = np.array([wah_w.loc[c,'WAH']   for c in QUERIES])
    wah_mat  = np.array([pft.loc[c,'Stg2']    for c in QUERIES])
    wah_proc = np.array([pft.loc[c,'Stg3']    for c in QUERIES])
    wah_tot  = wah_idx + wah_mat + wah_proc

    rtscan_raw = np.array([wah_w.loc[c,'RTScan'] for c in QUERIES])
    rtscan_idx = np.where(rtscan_raw < 50, rtscan_raw, np.nan)
    rtscan_proc= np.where(rtscan_raw < 50, wah_proc,   np.nan)

    return (ours_sp, ours_md, ours_dn, join,
            wah_idx, wah_mat, wah_proc, wah_tot,
            rtscan_idx, rtscan_proc)

def plot_overall(ssb_fixed, ssb_align, wah_rtscan, fname, sf_label,
                 include_rtscan=False):
    (ours_sp, ours_md, ours_dn, join,
     wah_idx, wah_mat, wah_proc, wah_tot,
     rtscan_idx, rtscan_proc) = build_overall(ssb_fixed, ssb_align, wah_rtscan)

    gm_sp = gmean(ours_sp); gm_md = gmean(ours_md); gm_dn = gmean(ours_dn)
    gm_join = gmean(join)
    gm_wah  = gmean(wah_tot)

    xlabels = QLABELS + ['gmean']
    n = len(xlabels)
    x = np.arange(n)

    # bar groups: Sparse | Medium | Dense | WAH | Join [| RTScan]
    n_groups = 6 if include_rtscan else 5
    width = 0.8 / n_groups
    offsets = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * width

    c_sparse = '#1f77b4'
    c_medium = '#aec7e8'
    c_dense  = '#17becf'
    c_wah    = '#d62728'
    c_wah_m  = '#ff9896'
    c_proc   = '#98df8a'
    c_join   = '#2ca02c'
    c_rtscan = '#ff7f0e'

    fig, ax = plt.subplots(figsize=(16, 4))

    def vals(arr, gm): return list(arr) + [gm]

    ax.bar(x+offsets[0], vals(ours_sp, gm_sp), width, label='Sparse', color=c_sparse)
    ax.bar(x+offsets[1], vals(ours_md, gm_md), width, label='Medium', color=c_medium)
    ax.bar(x+offsets[2], vals(ours_dn, gm_dn), width, label='Dense', color=c_dense)

    wah_b = vals(wah_idx, gm_wah)
    wah_m = vals(wah_mat, 0)
    wah_p = vals(wah_proc, 0)
    ax.bar(x+offsets[3], wah_b, width, label='WAH (Perfect)', color=c_wah)
    ax.bar(x+offsets[3], wah_m, width, bottom=wah_b,            color=c_wah_m)
    ax.bar(x+offsets[3], wah_p, width,
           bottom=[a+b for a,b in zip(wah_b, wah_m)],           color=c_proc)

    ax.bar(x+offsets[4], vals(join, gm_join), width, label='Join', color=c_join)

    if include_rtscan:
        gm_rt = gmean(rtscan_idx[~np.isnan(rtscan_idx)] +
                      rtscan_proc[~np.isnan(rtscan_proc)])
        rt_b = list(rtscan_idx)  + [gm_rt]
        rt_p = list(rtscan_proc) + [0]
        ax.bar(x+offsets[5], rt_b, width, label='RTScan', color=c_rtscan)
        ax.bar(x+offsets[5], rt_p, width, bottom=rt_b,    color=c_proc)

    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.legend(loc='upper right', ncol=3, fontsize=13)
    ax.axvline(x=len(QUERIES)-0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{fname}.pdf', bbox_inches='tight')
    plt.savefig(f'{fname}.png', bbox_inches='tight', dpi=150)
    print(f"Saved {fname}.pdf/png")
    print(f"  Ours Sparse gmean={gm_sp:.3f}ms  Medium={gm_md:.3f}ms  Dense={gm_dn:.3f}ms")
    print(f"  WAH gmean={gm_wah:.3f}ms  Join gmean={gm_join:.3f}ms")
    print(f"  Speedup Sparse vs WAH={gm_wah/gm_sp:.2f}x  vs Join={gm_join/gm_sp:.2f}x")

# =============================================================================
# Figure 1: Overall Performance SF=20 (with RTScan)
# =============================================================================
plot_overall(ssb20_fixed, ssb20_align, wah20,
             'fig_overall_sf20', '20', include_rtscan=True)

# =============================================================================
# Figure 2: Overall Performance SF=N (no RTScan)
# =============================================================================
plot_overall(ssb_fixed, ssb_align, wah_sf,
             f'fig_overall_sf{sf}', sf, include_rtscan=False)

# =============================================================================
# Figure 3: Alignment Scenarios (Perfect / AllAlig / MisAlig) — sfN
# Selected queries + gmean of all 13
# =============================================================================
sel_q = ['SSB11','SSB21','SSB31','SSB41','SSB42','SSB43']
sel_labels = ['Q1.1','Q2.1','Q3.1','Q4.1','Q4.2','Q4.3']

def align_vals(ssb_align, bintype, queries):
    sub = ssb_align[ssb_align['BinType']==bintype].set_index('Case')
    return np.array([sub.loc[c,'Ours'] for c in queries])

pft_sel  = align_vals(ssb_align, 'Perfect', sel_q)
ali_sel  = align_vals(ssb_align, 'AllAlig', sel_q)
mis_sel  = align_vals(ssb_align, 'MisAlig', sel_q)
# Join from Sparse rows (only non-zero there)
join_sel = np.array([ssb_fixed[ssb_fixed['BinType']=='Sparse'].set_index('Case').loc[c,'Join']
                     for c in sel_q])

pft_all  = align_vals(ssb_align, 'Perfect', QUERIES)
ali_all  = align_vals(ssb_align, 'AllAlig', QUERIES)
mis_all  = align_vals(ssb_align, 'MisAlig', QUERIES)
join_all = np.array([ssb_fixed[ssb_fixed['BinType']=='Sparse'].set_index('Case').loc[c,'Join']
                     for c in QUERIES])

labels = sel_labels + ['gmean']
pft_v  = list(pft_sel)  + [gmean(pft_all)]
ali_v  = list(ali_sel)  + [gmean(ali_all)]
mis_v  = list(mis_sel)  + [gmean(mis_all)]
join_v = list(join_sel) + [gmean(join_all)]

fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(labels))
w = 0.18
ax.bar(x - 1.5*w, pft_v,  w, label='Perfect', color='#1f77b4')
ax.bar(x - 0.5*w, ali_v,  w, label='Aligned', color='#aec7e8')
ax.bar(x + 0.5*w, mis_v,  w, label='Misaligned', color='#ff7f0e')
ax.bar(x + 1.5*w, join_v, w, label='Join', color='#2ca02c')
ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', ncol=4, fontsize=12)
ax.set_ylim(0, max(join_v) * 1.15)
ax.axvline(x=len(sel_q)-0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'fig_alignment_sf{sf}.pdf', bbox_inches='tight')
plt.savefig(f'fig_alignment_sf{sf}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_alignment_sf{sf}.pdf/png")
print(f"=== Alignment (SF={sf}) ===")
print(f"  gmean: Perfect={gmean(pft_all):.3f}  AllAlig={gmean(ali_all):.3f}  "
      f"MisAlig={gmean(mis_all):.3f}  Join={gmean(join_all):.3f}")
print(f"  AllAlig vs Perfect: {(gmean(ali_all)/gmean(pft_all)-1)*100:.1f}% slower")
print(f"  MisAlig vs AllAlig: {(gmean(mis_all)/gmean(ali_all)-1)*100:.1f}% slower")
print(f"  Join vs MisAlig:    {gmean(join_all)/gmean(mis_all):.2f}x faster (ours)")

# =============================================================================
# Figure 4: Fusion Progression + VQP overhead — merged ablation (sfN)
# Left panel: grouped bars for selected queries showing 5-step progression
#   NfuStg1+Stg2+Stg3 (stacked) | Basefus | PrgStg1+Stg2+Stg3 (stacked) | Ours | Dedicat
#   shown for Sparse layout (most informative; largest absolute differences)
# Right panel: speedup of Ours vs NfuTotal across selectivity (synthetic)
# =============================================================================
sel_q_fus = ['SSB11','SSB21','SSB31','SSB34','SSB41']
sel_l_fus = ['Q1.1','Q2.1','Q3.1','Q3.4','Q4.1']

sp = ssb_fixed[ssb_fixed['BinType']=='Sparse'].set_index('Case')

c_stg1 = '#1f77b4'   # index stage
c_stg2 = '#aec7e8'   # materialize stage
c_stg3 = '#98df8a'   # process stage
c_base = '#9467bd'   # basefus (naive paste)
c_ours = '#d62728'   # ours (fully fused)
c_ded  = '#8c564b'   # dedicat (compile-time)

fig, ax = plt.subplots(figsize=(7, 3.2))

n_sel = len(sel_q_fus)
x = np.arange(n_sel)
# 5 bar groups per query; pack tightly
w = 0.13
offsets = np.array([-2, -1, 0, 1, 2]) * w

for i, (c, lbl) in enumerate(zip(sel_q_fus, sel_l_fus)):
    row = sp.loc[c]
    nfu1 = row['NfuStg1']; stg2 = row['Stg2']; stg3 = row['Stg3']
    prg1 = row['PrgStg1']
    base = row['Basefus']
    ours = row['Ours']
    ded  = row['Dedicat']

    xi = x[i]
    # NfuStg1+Stg2+Stg3 stacked
    ax.bar(xi+offsets[0], nfu1, w, color=c_stg1,
           label='No Fusion Index Phase' if i==0 else '')
    ax.bar(xi+offsets[0], stg2, w, bottom=nfu1, color=c_stg2,
           label='Materialization Phase' if i==0 else '')
    ax.bar(xi+offsets[0], stg3, w, bottom=nfu1+stg2, color=c_stg3,
           label='Process Phase' if i==0 else '')
    # Basefus solid
    ax.bar(xi+offsets[1], base, w, color=c_base,
           label='Basic Fusion' if i==0 else '')
    # PrgStg1+Stg2+Stg3 stacked
    ax.bar(xi+offsets[2], prg1, w, color=c_stg1, alpha=0.55,
           label='VQP-Only' if i==0 else '')
    ax.bar(xi+offsets[2], stg2, w, bottom=prg1, color=c_stg2, alpha=0.55)
    ax.bar(xi+offsets[2], stg3, w, bottom=prg1+stg2, color=c_stg3, alpha=0.55)
    # Ours solid
    ax.bar(xi+offsets[3], ours, w, color=c_ours,
           label='Ours' if i==0 else '')
    # Dedicat solid
    ax.bar(xi+offsets[4], ded, w, color=c_ded,
           label='Dedicated Kernel' if i==0 else '')

ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(sel_l_fus)
ax.legend(loc='upper left', fontsize=9, ncol=4)

plt.tight_layout()
plt.savefig(f'fig_ablation_sf{sf}.pdf', bbox_inches='tight')
plt.savefig(f'fig_ablation_sf{sf}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_ablation_sf{sf}.pdf/png")

# Print progression summary (all 13 queries, Sparse)
sp_all = ssb_fixed[ssb_fixed['BinType']=='Sparse'].set_index('Case')
nfu_tot = sp_all['NfuStg1'] + sp_all['Stg2'] + sp_all['Stg3']
prg_tot = sp_all['PrgStg1'] + sp_all['Stg2'] + sp_all['Stg3']
print(f"=== Fusion Progression (SF={sf}, Sparse, gmean) ===")
print(f"  NfuTotal={gmean(nfu_tot):.3f}  Basefus={gmean(sp_all['Basefus']):.3f}  "
      f"PrgTotal={gmean(prg_tot):.3f}  Ours={gmean(sp_all['Ours']):.3f}  "
      f"Dedicat={gmean(sp_all['Dedicat']):.3f}")
print(f"  NfuTotal->Ours speedup: {gmean(nfu_tot)/gmean(sp_all['Ours']):.2f}x")
print(f"  Ours vs Dedicat overhead: {(gmean(sp_all['Ours'])/gmean(sp_all['Dedicat'])-1)*100:.1f}%")
print(f"  Basefus->Ours speedup: {gmean(sp_all['Basefus'])/gmean(sp_all['Ours']):.2f}x  "
      f"(answers MR4: general-purpose fusion not sufficient)")

# =============================================================================
# Figure 5: Synthetic Workloads — Selectivity and Skewness (sfN)
# =============================================================================
synth32 = synth_sf[synth_sf['BitW']==32].copy()
synth16 = synth_sf[synth_sf['BitW']==16].copy()
synth32['WAH_total'] = synth32['WAH'] + synth32['PftStg2'] + synth32['PftStg3']

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.4, wspace=0.35)

# Top-left: selectivity sensitivity (skew=1.12, 32-bit)
ax = fig.add_subplot(gs[0, 0])
skew_fixed = synth32[synth32['Skew']==1.12]
sels = skew_fixed['Seletiv'].values
ax.plot(sels, skew_fixed['Perfect'], 'o-', label='Perfect',     color='#1f77b4')
ax.plot(sels, skew_fixed['ManyOrs'], 's-', label='Aligned',     color='#aec7e8')
ax.plot(sels, skew_fixed['CandChk'], '^-', label='Misaligned',  color='#ff7f0e')
ax.plot(sels, skew_fixed['WAH_total'],'D-',label='WAH(Perfect)',color='#d62728')
ax.plot(sels, skew_fixed['Join'],    'x-', label='Join',        color='#2ca02c', alpha=0.7)
ax.set_xlabel('Selectivity (skew=1.12, 32-bit)')
ax.set_ylabel('Time (ms)')
ax.set_xscale('log', base=2)
ax.set_xticks(sels)
ax.set_xticklabels([sel_label(s) for s in sels])
ax.legend(loc='upper left', fontsize=9)

# Top-right: skewness sensitivity (sel=1/4, 32-bit)
ax = fig.add_subplot(gs[0, 1])
sel_fixed = synth32[synth32['Seletiv']==0.25]
skews = sel_fixed['Skew'].values
ax.plot(skews, sel_fixed['Perfect'], 'o-', label='Perfect',     color='#1f77b4')
ax.plot(skews, sel_fixed['ManyOrs'], 's-', label='Aligned',     color='#aec7e8')
ax.plot(skews, sel_fixed['CandChk'], '^-', label='Misaligned',  color='#ff7f0e')
ax.plot(skews, sel_fixed['WAH_total'],'D-',label='WAH(Perfect)',color='#d62728')
ax.plot(skews, sel_fixed['Join'],    'x-', label='Join',        color='#2ca02c', alpha=0.7)
ax.set_xlabel('Zipfian Skewness (sel=1/4, 32-bit)')
ax.set_ylabel('Time (ms)')
ax.set_xticks(skews)
ax.legend(loc='upper left', fontsize=9)

# Bottom: fusion speedup across selectivity (spans both columns)
ax = fig.add_subplot(gs[1, :])
sf116 = synth32[synth32['Skew']==1.16].copy()
sf116['NfuTotal'] = sf116['PftStg1'] + sf116['PftStg2'] + sf116['PftStg3']
sf116['FusSpeedup'] = sf116['NfuTotal'] / sf116['Perfect']
sf116['OrsNfu'] = sf116['OrsStg1'] + sf116['OrsStg2'] + sf116['OrsStg3']
sf116['ChkNfu'] = sf116['ChkStg1'] + sf116['ChkStg2'] + sf116['ChkStg3']
sf116['OrsSpeedup'] = sf116['OrsNfu'] / sf116['ManyOrs']
sf116['ChkSpeedup'] = sf116['ChkNfu'] / sf116['CandChk']
fus_sels = sf116['Seletiv'].values
ax.plot(fus_sels, sf116['FusSpeedup'], 'o-', label='Perfect',   color='#1f77b4')
ax.plot(fus_sels, sf116['OrsSpeedup'], 's-', label='Aligned',   color='#aec7e8')
ax.plot(fus_sels, sf116['ChkSpeedup'], '^-', label='Misaligned', color='#ff7f0e')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Selectivity (skew=1.16, 32-bit)')
ax.set_ylabel('Fusion Speedup')
ax.set_xscale('log', base=2)
ax.set_xticks(fus_sels)
ax.set_xticklabels([sel_label(s) for s in fus_sels])
ax.legend(loc='lower left', fontsize=10)
ax.set_ylim(0.9, max(sf116['FusSpeedup'].max(), sf116['ChkSpeedup'].max()) * 1.1)

plt.savefig(f'fig_synth_sf{sf}.pdf', bbox_inches='tight')
plt.savefig(f'fig_synth_sf{sf}.png', bbox_inches='tight', dpi=150)
print(f"Saved fig_synth_sf{sf}.pdf/png")

print("=== Synthetic Workload Analysis ===")
print("Selectivity sensitivity (skew=1.12, 32-bit):")
for s in sels:
    row = skew_fixed[skew_fixed['Seletiv']==s].iloc[0]
    print(f"  sel=1/{int(round(1/s)):3d}: Perfect={row['Perfect']:.3f}  "
          f"CandChk={row['CandChk']:.3f}  WAH={row['WAH_total']:.3f}  "
          f"CandChk/WAH={row['CandChk']/row['WAH_total']:.2f}x")
print("Skewness sensitivity (sel=1/32, 32-bit):")
for sk in skews:
    row = sel_fixed[sel_fixed['Skew']==sk].iloc[0]
    print(f"  skew={sk:.2f}: Perfect={row['Perfect']:.3f}  "
          f"CandChk={row['CandChk']:.3f}  WAH={row['WAH_total']:.3f}")
print("16-bit vs 32-bit (skew=1.12):")
for s in sels:
    r16 = synth16[(synth16['Skew']==1.12) & (synth16['Seletiv']==s)].iloc[0]
    r32 = synth32[(synth32['Skew']==1.12) & (synth32['Seletiv']==s)].iloc[0]
    print(f"  sel=1/{int(round(1/s)):3d}: 16b CandChk={r16['CandChk']:.3f}  "
          f"32b CandChk={r32['CandChk']:.3f}  32b WAH={r32['WAH_total']:.3f}")

# =============================================================================
# Figure 6: Index Creation and Memory Usage (SF=20, Sparse/Balanced/Dense vs WAH/RTScan)
# Left: construction time (log scale); Right: memory usage
# =============================================================================
def get_total(idxcreate, tag):
    row = idxcreate[idxcreate['Column'] == tag]
    if len(row) == 0:
        return None
    return row.iloc[0]

t_sparse   = get_total(idxcreate20, 'Total_Sparse')
t_balanced = get_total(idxcreate20, 'Total_Balanced')
t_dense    = get_total(idxcreate20, 'Total_Dense')
t_rtsiev   = get_total(idxcreate20, 'RTScanSieve')
t_rtrays   = get_total(idxcreate20, 'RTScanRays')

methods = ['Ours\nSparse', 'Ours\nBalanced', 'Ours\nDense', 'WAH\nSparse', 'WAH\nBalanced', 'WAH\nDense', 'RTScan']
times   = [t_sparse['My(ms)'], t_balanced['My(ms)'], t_dense['My(ms)'],
           t_sparse['WAH(ms)'], t_balanced['WAH(ms)'], t_dense['WAH(ms)'],
           t_rtsiev['My(MB)'] + t_rtrays['My(MB)']]   # RTScan time stored in My(MB) column
memory  = [t_sparse['My(MB)'], t_balanced['My(MB)'], t_dense['My(MB)'],
           t_sparse['WAH(MB)'], t_balanced['WAH(MB)'], t_dense['WAH(MB)'],
           t_rtsiev['WAH(MB)'] + t_rtrays['WAH(MB)']]

c_ours   = ['#1f77b4', '#aec7e8', '#17becf']
c_wah    = ['#d62728', '#ff9896', '#ffbbbb']
c_rtscan = '#ff7f0e'
colors   = c_ours + c_wah + [c_rtscan]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

ax = axes[0]
bars = ax.bar(methods, times, color=colors)
ax.set_ylabel('Construction Time (ms)')
ax.set_yscale('log')
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.4,
            f'{val:.0f}', ha='center', va='bottom', fontsize=10)

ax = axes[1]
bars = ax.bar(methods, memory, color=colors)
ax.set_ylabel('Memory Usage (MB)')
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=10)
rtscan_mem = t_rtsiev['WAH(MB)'] + t_rtrays['WAH(MB)']
for bar, val in zip(bars, memory):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rtscan_mem*0.01,
            f'{val:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('fig_creation.pdf', bbox_inches='tight')
plt.savefig('fig_creation.png', bbox_inches='tight', dpi=150)
print("Saved fig_creation.pdf/png")
print("=== Index Creation (SF=20) ===")
rtscan_time = t_rtsiev['My(MB)'] + t_rtrays['My(MB)']
print(f"  Time  — Sparse: {t_sparse['My(ms)']:.1f}ms  Balanced: {t_balanced['My(ms)']:.1f}ms  "
      f"Dense: {t_dense['My(ms)']:.1f}ms")
print(f"  WAH   — Sparse: {t_sparse['WAH(ms)']:.1f}ms  Balanced: {t_balanced['WAH(ms)']:.1f}ms  "
      f"Dense: {t_dense['WAH(ms)']:.1f}ms")
print(f"  RTScan: {rtscan_time:.0f}ms  ({rtscan_time/t_sparse['My(ms)']:.0f}x vs Ours Sparse)")
print(f"  Memory — Sparse: {t_sparse['My(MB)']:.0f}MB  Balanced: {t_balanced['My(MB)']:.0f}MB  "
      f"Dense: {t_dense['My(MB)']:.0f}MB")
print(f"  WAH   — Sparse: {t_sparse['WAH(MB)']:.0f}MB  Balanced: {t_balanced['WAH(MB)']:.0f}MB  "
      f"Dense: {t_dense['WAH(MB)']:.0f}MB")
print(f"  Ours Sparse vs WAH Sparse: time {t_sparse['WAH(ms)']/t_sparse['My(ms)']:.1f}x faster, "
      f"memory {(t_sparse['My(MB)']/t_sparse['WAH(MB)']-1)*100:.0f}% more")

# =============================================================================
# Figure 7: H800 / server-GPU figures — overall + alignment + ablation
# Reuse the same plot functions, driven by h800_sf300.txt loaded separately.
# Only generated when sf argument is NOT 300 (i.e. called as drawFigs.py 140 .)
# When called as drawFigs.py 300 . the figures above already cover H800.
# To generate H800 figures from a 5090 run: call drawFigs.py 140 . and this
# section loads h800_sf300.txt from the same directory.
# =============================================================================
h800_path = 'h800_sf300.txt'
if os.path.exists(h800_path) and sf != '300':
    wah300, idxcreate300, ssb300_fixed, ssb300_align, method12_300, synth300, _ = \
        load_sf(h800_path)

    # H800 overall — half-column figure with selected queries + gmean
    sel_q_h800 = ['SSB11','SSB21','SSB31','SSB32','SSB41','SSB43']
    sel_l_h800 = ['Q1.1','Q2.1','Q3.1','Q3.2','Q4.1','Q4.3']
    sp300 = ssb300_fixed[ssb300_fixed['BinType']=='Sparse'].set_index('Case')
    md300 = ssb300_fixed[ssb300_fixed['BinType']=='Medium'].set_index('Case')
    dn300 = ssb300_fixed[ssb300_fixed['BinType']=='Dense'].set_index('Case')
    wah300_w = wah300.set_index('Case')
    pft300 = ssb300_align[ssb300_align['BinType']=='Perfect'].set_index('Case')

    ours_sp300 = np.array([sp300.loc[c,'Ours'] for c in QUERIES])
    ours_md300 = np.array([md300.loc[c,'Ours'] for c in QUERIES])
    ours_dn300 = np.array([dn300.loc[c,'Ours'] for c in QUERIES])
    join300    = np.array([sp300.loc[c,'Join'] for c in QUERIES])
    wah_idx300 = np.array([wah300_w.loc[c,'WAH'] for c in QUERIES])
    wah_mat300 = np.array([pft300.loc[c,'Stg2'] for c in QUERIES])
    wah_proc300= np.array([pft300.loc[c,'Stg3'] for c in QUERIES])
    wah_tot300 = wah_idx300 + wah_mat300 + wah_proc300

    # Selected + gmean
    def sel_vals(arr, queries, sel_qs):
        q_idx = [QUERIES.index(q) for q in sel_qs]
        return [arr[i] for i in q_idx] + [gmean(arr)]

    xlabels300 = sel_l_h800 + ['gmean']
    n300 = len(xlabels300)
    x300 = np.arange(n300)
    n_groups = 5
    width300 = 0.8 / n_groups
    off300 = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * width300

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x300+off300[0], sel_vals(ours_sp300, QUERIES, sel_q_h800), width300,
           label='Ours Sparse', color='#1f77b4')
    ax.bar(x300+off300[1], sel_vals(ours_md300, QUERIES, sel_q_h800), width300,
           label='Ours Medium', color='#aec7e8')
    ax.bar(x300+off300[2], sel_vals(ours_dn300, QUERIES, sel_q_h800), width300,
           label='Ours Dense', color='#17becf')
    wah_b300 = sel_vals(wah_idx300, QUERIES, sel_q_h800)
    wah_m300 = sel_vals(wah_mat300, QUERIES, sel_q_h800)
    wah_p300 = sel_vals(wah_proc300, QUERIES, sel_q_h800)
    ax.bar(x300+off300[3], wah_b300, width300, label='WAH (Best-case)', color='#d62728')
    ax.bar(x300+off300[3], wah_m300, width300, bottom=wah_b300, color='#ff9896')
    ax.bar(x300+off300[3], wah_p300, width300,
           bottom=[a+b for a,b in zip(wah_b300, wah_m300)], color='#98df8a')
    ax.bar(x300+off300[4], sel_vals(join300, QUERIES, sel_q_h800), width300,
           label='Join', color='#2ca02c')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x300)
    ax.set_xticklabels(xlabels300, rotation=45, ha='right')
    ax.legend(loc='upper left', ncol=3, fontsize=13)
    ax.axvline(x=len(sel_q_h800)-0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fig_overall_sf300.pdf', bbox_inches='tight')
    plt.savefig('fig_overall_sf300.png', bbox_inches='tight', dpi=150)
    print("Saved fig_overall_sf300.pdf/png")
    print(f"=== H800 SF=300 Overall ===")
    print(f"  Ours Sparse gmean={gmean(ours_sp300):.3f}ms  Medium={gmean(ours_md300):.3f}ms  "
          f"Dense={gmean(ours_dn300):.3f}ms")
    print(f"  WAH gmean={gmean(wah_tot300):.3f}ms  Join gmean={gmean(join300):.3f}ms")
    print(f"  Speedup Sparse vs WAH={gmean(wah_tot300)/gmean(ours_sp300):.2f}x  "
          f"vs Join={gmean(join300)/gmean(ours_sp300):.2f}x")

    # H800 ablation (fusion progression)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x300f = np.arange(n_sel)
    w300f = 0.11
    offsets300f = np.array([-2, -1, 0, 1, 2]) * w300f
    for i, (c, lbl) in enumerate(zip(sel_q_fus, sel_l_fus)):
        row = sp300.loc[c]
        nfu1 = row['NfuStg1']; stg2 = row['Stg2']; stg3 = row['Stg3']
        prg1 = row['PrgStg1']
        base = row['Basefus']
        ours = row['Ours']
        ded  = row['Dedicat']
        xi = x300f[i]
        ax.bar(xi+offsets300f[0], nfu1, w300f, color=c_stg1,
               label='Index Phase of No Fusion' if i==0 else '')
        ax.bar(xi+offsets300f[0], stg2, w300f, bottom=nfu1, color=c_stg2,
               label='Materialization Phase' if i==0 else '')
        ax.bar(xi+offsets300f[0], stg3, w300f, bottom=nfu1+stg2, color=c_stg3,
               label='Processing Phase' if i==0 else '')
        ax.bar(xi+offsets300f[1], base, w300f, color=c_base,
               label='Basic Fusion' if i==0 else '')
        ax.bar(xi+offsets300f[2], prg1, w300f, color=c_stg1, alpha=0.55,
               label='VQP Fusion Only' if i==0 else '')
        ax.bar(xi+offsets300f[2], stg2, w300f, bottom=prg1, color=c_stg2, alpha=0.55)
        ax.bar(xi+offsets300f[2], stg3, w300f, bottom=prg1+stg2, color=c_stg3, alpha=0.55)
        ax.bar(xi+offsets300f[3], ours, w300f, color=c_ours,
               label='Ours (fully fused)' if i==0 else '')
        ax.bar(xi+offsets300f[4], ded,  w300f, color=c_ded,
               label='Dedicated' if i==0 else '')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x300f)
    ax.set_xticklabels(sel_l_fus)
    ax.legend(loc='upper right', fontsize=10, ncol=4)
    plt.tight_layout()
    plt.savefig('fig_ablation_sf300.pdf', bbox_inches='tight')
    plt.savefig('fig_ablation_sf300.png', bbox_inches='tight', dpi=150)
    print("Saved fig_ablation_sf300.pdf/png")
    nfu_tot300 = sp300['NfuStg1'] + sp300['Stg2'] + sp300['Stg3']
    print(f"=== Fusion Progression (SF=300, H800, Sparse, gmean) ===")
    print(f"  NfuTotal={gmean(nfu_tot300):.3f}  Basefus={gmean(sp300['Basefus']):.3f}  "
          f"Ours={gmean(sp300['Ours']):.3f}  Dedicat={gmean(sp300['Dedicat']):.3f}")
    print(f"  NfuTotal->Ours speedup: {gmean(nfu_tot300)/gmean(sp300['Ours']):.2f}x")
    print(f"  Ours vs Dedicat overhead: {(gmean(sp300['Ours'])/gmean(sp300['Dedicat'])-1)*100:.1f}%")
    print(f"  Basefus->Ours speedup: {gmean(sp300['Basefus'])/gmean(sp300['Ours']):.2f}x")
