#!/usr/bin/python3
import os
import pandas as pd
import numpy as np
import struct
import argparse

parser = argparse.ArgumentParser(description="Process a | separated table into column files.")
parser.add_argument('--inPath', type=str, required=True, help='Path to the input table file')
parser.add_argument('--outDir', type=str, required=True, help='Output directory for column files')
parser.add_argument('--needFill0', action='store_true', help='Fill missing IDs in the first column with zeroes')
parser.add_argument('--pad0', action='store_true', help='Pad the 0th column')
parser.add_argument('--useCols', type=int, nargs='*', default=[], help='List of column indices to use')
args = parser.parse_args()
os.makedirs(args.outDir, exist_ok=True)
df = pd.read_csv(args.inPath, sep='|', header=None, usecols=args.useCols)
print(df.head())

# Convert columns to appropriate types
encodeDicts = {}
for col in df.columns:
  if df[col].dtype == np.int32:
    continue
  elif df[col].dtype == np.int64:
    df[col] = df[col].astype(np.int32)
    continue
  elif df[col].dtype == np.float64 or df[col].dtype == np.float32:
    # Convert floating points to int(x * 100)
    df[col] = pd.to_numeric(df[col], errors='coerce')\
                .apply(lambda x: int(x * 100)).astype(np.int32)
    continue

  # Attempt to convert date (remove '-')
  try:
    df[col] = pd.to_numeric(df[col].str.replace('-', ''), errors='raise')\
                .astype(np.int32)
  except ValueError:
    # Categorical string encoding
    encoding_dict = {val: idx + 1 for idx, val in enumerate(df[col].unique())}
    df[col] = df[col].map(encoding_dict).fillna(0).astype(np.int32)
    encodeDicts[col] = encoding_dict

# Write encoding dictionaries to files
for colIdx, d in encodeDicts.items():
  with open(os.path.join(args.outDir, f"col{colIdx}dict.txt"), 'w') as f:
    f.write(str(d))
for colIdx, colVals in df.items():
  outBinPath = os.path.join(args.outDir, f"col{colIdx}.bin")
  with open(outBinPath, 'wb') as f:
    if args.needFill0:
      # Fill missing IDs cause the current join impl supports join by subscript only
      # Only required by the order.tbl in TPC-H
      idxs, colVals = df[0].to_numpy(), colVals.to_numpy()
      toWrite = np.full(idxs[-1] + 1, -1, dtype=np.int32)
      toWrite[idxs] = colVals
      f.write(b''.join(toWrite))
    else:
      if args.pad0:
        # Add a zeroth row so that IDs in fact table is the subscript to dimension
        # tables (rather than subscript+1)
        f.write(struct.pack('<I', 0))
      f.write(b''.join(colVals.to_numpy()))
