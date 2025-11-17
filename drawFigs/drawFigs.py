#!/usr/bin/python3
import sys
import os
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load result files into dataframes
if len(sys.argv) > 1:
  os.chdir(sys.argv[1])
a = open('sf20.txt').read().split('\n\n')
overhead20 = pd.read_csv(StringIO(a[0]), sep='\t')
synth20 = pd.read_csv(StringIO(a[5]), sep='\t')
ssb20 = pd.concat([
  pd.read_csv(StringIO(a[1]), sep='\t'),
  pd.read_csv(StringIO(a[2]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[3]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[4]), sep='\t').iloc[:, 1:],
  pd.read_csv('crystal.txt', sep='\t').iloc[:, 1:],
], axis=1)
a = open('sf100.txt').read().split('\n\n')
overhead100 = pd.read_csv(StringIO(a[0]), sep='\t')
synth100 = pd.read_csv(StringIO(a[5]), sep='\t')
ssb100 = pd.concat([
  pd.read_csv(StringIO(a[1]), sep='\t'),
  pd.read_csv(StringIO(a[2]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[3]), sep='\t').iloc[:, 1:],
  pd.read_csv(StringIO(a[4]), sep='\t').iloc[:, 1:]
], axis=1)

# print('SF20:', overhead20, ssb20, synth20, 'SF100:', overhead100, synth100,
#       sep='\n\n')

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
