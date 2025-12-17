#!/usr/bin/python
import pandas as pd
import numpy as np
import struct
import os
import sys

def dumpCol(col: pd.Series, outPath: str, pad0: bool, strLut: dict[str, int], first: bool):
    if col.dtype == np.float32 or col.dtype == np.float64:
        col = col.apply(lambda x: int(x * 100)).astype(np.int32)
    elif col.dtype != np.int32 and col.dtype != np.int64:
        try:  # date type?
            col = pd.to_numeric(col.str.replace('-', ''), downcast='unsigned', errors='raise')
        except ValueError:
            col = col.map(strLut).astype(np.int32)
            # m = col.map(strLut)
            # msk = m.isna()
            # if msk.any():
            #     print(col[msk].tolist())
            # col = m.astype(np.int32)

    col = pd.to_numeric(col, downcast='unsigned')
    if col.dtype == np.uint8: nByte = 1
    elif col.dtype == np.uint16: nByte = 2
    elif col.dtype == np.uint32: nByte = 4
    else : nByte = 8
    with open(outPath, 'wb' if first else 'ab') as f:
        if first:
            print(outPath, col.head(), sep='\n')
            f.write(struct.pack('<I', nByte)) # nByte is 4B
            if pad0: f.write(b'\x00' * nByte) # write the appropriate padding
        else: print('.', end='', flush=True)
        f.write(col.values.tobytes())

ssbStrLut = {
    "TRUCK": 0, "MAIL": 1, "REG AIR": 2, "AIR": 3, "FOB": 4, "RAIL": 5, "SHIP": 6,
    "5-LOW": 5, "4-NOT SPECI":4, "3-MEDIUM": 3, "2-HIGH": 2, "1-URGENT": 1,
    "AUTOMOBILE": 0, "HOUSEHOLD": 1, "BUILDING": 2, "MACHINERY": 3, "FURNITURE": 4,

    "MOROCCO  0": 0  , "MOROCCO  1": 1  , "MOROCCO  2": 2  , "MOROCCO  3": 3  , "MOROCCO  4": 4  ,
    "MOROCCO  5": 5  , "MOROCCO  6": 6  , "MOROCCO  7": 7  , "MOROCCO  8": 8  , "MOROCCO  9": 9  ,
    "ETHIOPIA 0": 10 , "ETHIOPIA 1": 11 , "ETHIOPIA 2": 12 , "ETHIOPIA 3": 13 , "ETHIOPIA 4": 14 ,
    "ETHIOPIA 5": 15 , "ETHIOPIA 6": 16 , "ETHIOPIA 7": 17 , "ETHIOPIA 8": 18 , "ETHIOPIA 9": 19 ,
    "ALGERIA  0": 20 , "ALGERIA  1": 21 , "ALGERIA  2": 22 , "ALGERIA  3": 23 , "ALGERIA  4": 24 ,
    "ALGERIA  5": 25 , "ALGERIA  6": 26 , "ALGERIA  7": 27 , "ALGERIA  8": 28 , "ALGERIA  9": 29 ,
    "MOZAMBIQU0": 30 , "MOZAMBIQU1": 31 , "MOZAMBIQU2": 32 , "MOZAMBIQU3": 33 , "MOZAMBIQU4": 34 ,
    "MOZAMBIQU5": 35 , "MOZAMBIQU6": 36 , "MOZAMBIQU7": 37 , "MOZAMBIQU8": 38 , "MOZAMBIQU9": 39 ,
    "KENYA    0": 40 , "KENYA    1": 41 , "KENYA    2": 42 , "KENYA    3": 43 , "KENYA    4": 44 ,
    "KENYA    5": 45 , "KENYA    6": 46 , "KENYA    7": 47 , "KENYA    8": 48 , "KENYA    9": 49 ,
    "UNITED KI0": 50 , "UNITED KI1": 51 , "UNITED KI2": 52 , "UNITED KI3": 53 , "UNITED KI4": 54 ,
    "UNITED KI5": 55 , "UNITED KI6": 56 , "UNITED KI7": 57 , "UNITED KI8": 58 , "UNITED KI9": 59 ,
    "FRANCE   0": 60 , "FRANCE   1": 61 , "FRANCE   2": 62 , "FRANCE   3": 63 , "FRANCE   4": 64 ,
    "FRANCE   5": 65 , "FRANCE   6": 66 , "FRANCE   7": 67 , "FRANCE   8": 68 , "FRANCE   9": 69 ,
    "GERMANY  0": 70 , "GERMANY  1": 71 , "GERMANY  2": 72 , "GERMANY  3": 73 , "GERMANY  4": 74 ,
    "GERMANY  5": 75 , "GERMANY  6": 76 , "GERMANY  7": 77 , "GERMANY  8": 78 , "GERMANY  9": 79 ,
    "RUSSIA   0": 80 , "RUSSIA   1": 81 , "RUSSIA   2": 82 , "RUSSIA   3": 83 , "RUSSIA   4": 84 ,
    "RUSSIA   5": 85 , "RUSSIA   6": 86 , "RUSSIA   7": 87 , "RUSSIA   8": 88 , "RUSSIA   9": 89 ,
    "ROMANIA  0": 90 , "ROMANIA  1": 91 , "ROMANIA  2": 92 , "ROMANIA  3": 93 , "ROMANIA  4": 94 ,
    "ROMANIA  5": 95 , "ROMANIA  6": 96 , "ROMANIA  7": 97 , "ROMANIA  8": 98 , "ROMANIA  9": 99 ,
    "JORDAN   0": 100, "JORDAN   1": 101, "JORDAN   2": 102, "JORDAN   3": 103, "JORDAN   4": 104,
    "JORDAN   5": 105, "JORDAN   6": 106, "JORDAN   7": 107, "JORDAN   8": 108, "JORDAN   9": 109,
    "EGYPT    0": 110, "EGYPT    1": 111, "EGYPT    2": 112, "EGYPT    3": 113, "EGYPT    4": 114,
    "EGYPT    5": 115, "EGYPT    6": 116, "EGYPT    7": 117, "EGYPT    8": 118, "EGYPT    9": 119,
    "SAUDI ARA0": 120, "SAUDI ARA1": 121, "SAUDI ARA2": 122, "SAUDI ARA3": 123, "SAUDI ARA4": 124,
    "SAUDI ARA5": 125, "SAUDI ARA6": 126, "SAUDI ARA7": 127, "SAUDI ARA8": 128, "SAUDI ARA9": 129,
    "IRAN     0": 130, "IRAN     1": 131, "IRAN     2": 132, "IRAN     3": 133, "IRAN     4": 134,
    "IRAN     5": 135, "IRAN     6": 136, "IRAN     7": 137, "IRAN     8": 138, "IRAN     9": 139,
    "IRAQ     0": 140, "IRAQ     1": 141, "IRAQ     2": 142, "IRAQ     3": 143, "IRAQ     4": 144,
    "IRAQ     5": 145, "IRAQ     6": 146, "IRAQ     7": 147, "IRAQ     8": 148, "IRAQ     9": 149,
    "ARGENTINA0": 150, "ARGENTINA1": 151, "ARGENTINA2": 152, "ARGENTINA3": 153, "ARGENTINA4": 154,
    "ARGENTINA5": 155, "ARGENTINA6": 156, "ARGENTINA7": 157, "ARGENTINA8": 158, "ARGENTINA9": 159,
    "CANADA   0": 160, "CANADA   1": 161, "CANADA   2": 162, "CANADA   3": 163, "CANADA   4": 164,
    "CANADA   5": 165, "CANADA   6": 166, "CANADA   7": 167, "CANADA   8": 168, "CANADA   9": 169,
    "PERU     0": 170, "PERU     1": 171, "PERU     2": 172, "PERU     3": 173, "PERU     4": 174,
    "PERU     5": 175, "PERU     6": 176, "PERU     7": 177, "PERU     8": 178, "PERU     9": 179,
    "BRAZIL   0": 180, "BRAZIL   1": 181, "BRAZIL   2": 182, "BRAZIL   3": 183, "BRAZIL   4": 184,
    "BRAZIL   5": 185, "BRAZIL   6": 186, "BRAZIL   7": 187, "BRAZIL   8": 188, "BRAZIL   9": 189,
    "UNITED ST0": 190, "UNITED ST1": 191, "UNITED ST2": 192, "UNITED ST3": 193, "UNITED ST4": 194,
    "UNITED ST5": 195, "UNITED ST6": 196, "UNITED ST7": 197, "UNITED ST8": 198, "UNITED ST9": 199,
    "CHINA    0": 200, "CHINA    1": 201, "CHINA    2": 202, "CHINA    3": 203, "CHINA    4": 204,
    "CHINA    5": 205, "CHINA    6": 206, "CHINA    7": 207, "CHINA    8": 208, "CHINA    9": 209,
    "INDIA    0": 210, "INDIA    1": 211, "INDIA    2": 212, "INDIA    3": 213, "INDIA    4": 214,
    "INDIA    5": 215, "INDIA    6": 216, "INDIA    7": 217, "INDIA    8": 218, "INDIA    9": 219,
    "JAPAN    0": 220, "JAPAN    1": 221, "JAPAN    2": 222, "JAPAN    3": 223, "JAPAN    4": 224,
    "JAPAN    5": 225, "JAPAN    6": 226, "JAPAN    7": 227, "JAPAN    8": 228, "JAPAN    9": 229,
    "VIETNAM  0": 230, "VIETNAM  1": 231, "VIETNAM  2": 232, "VIETNAM  3": 233, "VIETNAM  4": 234,
    "VIETNAM  5": 235, "VIETNAM  6": 236, "VIETNAM  7": 237, "VIETNAM  8": 238, "VIETNAM  9": 239,
    "INDONESIA0": 240, "INDONESIA1": 241, "INDONESIA2": 242, "INDONESIA3": 243, "INDONESIA4": 244,
    "INDONESIA5": 245, "INDONESIA6": 246, "INDONESIA7": 247, "INDONESIA8": 248, "INDONESIA9": 249,
}

tpchStrLut = {
    "TRUCK": 0, "MAIL": 1, "REG AIR": 2, "AIR": 3, "FOB": 4, "RAIL": 5, "SHIP": 6,
    "5-LOW": 5, "4-NOT SPECIFIED":4, "3-MEDIUM": 3, "2-HIGH": 2, "1-URGENT": 1,
    "AUTOMOBILE": 0, "HOUSEHOLD": 1, "BUILDING": 2, "MACHINERY": 3, "FURNITURE": 4,
    "N": 0, "R": 1, "A": 2,
    "O": 0, "F": 1, "P": 2,
    "DELIVER IN PERSON": 0, "TAKE BACK RETURN": 1, "NONE": 2, "COLLECT COD": 3
}

for i in range(1, 6):
    tpchStrLut[f"Manufacturer#{i}"] = i
    for j in range(1, 6):
        tpchStrLut[f"Brand#{i}{j}"] = (i<<3) + j
        for k in range(1, 41):
            ssbStrLut[f"MFGR#{i}{j}{k}"] = (i-1)*200 + (j-1)*40 + (k-1)

d = ['turquoise', 'deep', 'frosted', 'seashell', 'blanched', 'almond', 'lace',
     'red', 'violet', 'cream', 'drab', 'midnight', 'pale', 'blush', 'medium',
     'coral', 'tomato', 'bisque', 'beige', 'powder', 'indian', 'cyan',
     'honeydew', 'orange', 'sandy', 'mint', 'aquamarine', 'azure', 'lime',
     'grey', 'puff', 'burlywood', 'tan', 'sky', 'cornflower', 'blue',
     'cornsilk', 'burnished', 'chiffon', 'white', 'forest', 'wheat',
     'gainsboro', 'hot', 'antique', 'saddle', 'linen', 'moccasin', 'rosy',
     'light', 'green', 'olive', 'black', 'ghost', 'ivory', 'peach', 'smoke',
     'royal', 'pink', 'slate', 'navajo', 'dodger', 'yellow', 'steel',
     'magenta', 'lavender', 'rose', 'misty', 'spring', 'navy', 'purple',
     'khaki', 'thistle', 'firebrick', 'lawn', 'dim', 'chocolate', 'snow',
     'metallic', 'lemon', 'plum', 'brown', 'sienna', 'papaya', 'chartreuse',
     'dark', 'floral', 'peru', 'salmon', 'orchid', 'maroon', 'goldenrod']
for i, e in enumerate(d):
    ssbStrLut[e] = i
    for j, f in enumerate(d):
        ssbStrLut[f'{e} {f}'] = (i<<8) + j

for i, e in enumerate(["PKG", "CASE", "DRUM", "BAG", "PACK", "CAN", "BOX", "JAR"]):
    for j, f in enumerate(["JUMBO", "LG", "WRAP", "MED", "SM"]):
        ssbStrLut[f'{f} {e}'] = (i<<4) + j
        tpchStrLut[f'{f} {e}'] = (i<<4) + j
for i, e in enumerate(["PROMO", "LARGE", "STANDARD", "SMALL", "MEDIUM", "ECONOMY"]):
    for j, f in enumerate(["POLISHED", "PLATED", "POLISHED", "BRUSHED", "BURNISHED", "ANODIZED"]):
        for k, g in enumerate(["COPPER", "BRASS", "TIN", "STEEL", "COPPER", "NICKEL"]):
            ssbStrLut[f'{e} {f} {g}'] = i*36 + j*6 + k
            tpchStrLut[f'{e} {f} {g}'] = i*36 + j*6 + k

if len(sys.argv) >= 1:
    os.chdir(sys.argv[1])
# SSB
if os.access("./lineorder.tbl", os.R_OK):
    os.makedirs("ssbCols", exist_ok=True)
    # Process lineorder.tbl in chunks
    chunkSz = 5000000 # 5M rows
    df = pd.read_csv("./supplier.tbl", sep='|', header=None, usecols=[3])
    # dumpCol(df[3], "./ssbCols/suppCity", True, False, ssbStrLut)
    suppCity = np.fromiter(map(lambda x: ssbStrLut[x], df[3]), np.uint8)
    del df

    first = True
    for chunk in pd.read_csv("./lineorder.tbl", sep='|', header=None,
                             usecols=(2,3,4,5,8,9,11,12,13), chunksize=chunkSz):
        dumpCol(chunk[2],  "./ssbCols/loCustKey", False, ssbStrLut, first)
        dumpCol(chunk[3],  "./ssbCols/loPartKey", False, ssbStrLut, first)
        dumpCol(chunk[5],  "./ssbCols/loOrderDate", False, ssbStrLut, first)
        dumpCol(chunk[8],  "./ssbCols/loQuantity", False, ssbStrLut, first)
        dumpCol(chunk[9],  "./ssbCols/loExtendedPrice", False, ssbStrLut, first)
        dumpCol(chunk[11], "./ssbCols/loDiscount", False, ssbStrLut, first)
        dumpCol(chunk[12], "./ssbCols/loRevenue", False, ssbStrLut, first)
        dumpCol(chunk[13], "./ssbCols/loSupplyCost", False, ssbStrLut, first)
        with open("./ssbCols/loSuppCity", 'wb' if first else 'ab') as f:
            if first:
                f.write(struct.pack('<I', 1))
            f.write(bytes(suppCity[k - 1] for k in chunk[4]))
        first = False

    # df = pd.read_csv("./date.tbl", sep='|', header=None, usecols=[10])
    # dumpCol(df[10], "./ssbCols/WeekNumInYear", False, False, ssbStrLut)
    df = pd.read_csv("./customer.tbl", sep='|', header=None, usecols=(3,7))
    dumpCol(df[3], "./ssbCols/custCity", True, ssbStrLut, True)
    dumpCol(df[7], "./ssbCols/custMktSegment", True, ssbStrLut, True)
    df = pd.read_csv("./part.tbl", sep='|', header=None, usecols=(1,4,5,6,7,8))
    dumpCol(df[1], "./ssbCols/partName", True, ssbStrLut, True)
    dumpCol(df[4], "./ssbCols/partMfgr", True, ssbStrLut, True)
    dumpCol(df[5], "./ssbCols/partColor", True, ssbStrLut, True)
    dumpCol(df[6], "./ssbCols/partType", True, ssbStrLut, True)
    dumpCol(df[7], "./ssbCols/partSize", True, ssbStrLut, True)
    dumpCol(df[8], "./ssbCols/partContainer", True, ssbStrLut, True)
    exit(0)
print('Cannot find SSB columns in current dir!')
exit(10)


# TPCH
'''
os.makedirs("tpchCols", exist_ok=True)

# Process lineitem.tbl in chunks
chunkSz = 200 * 1024 * 1024 // 100  # Estimate ~100 bytes per row for 200MB chunks
first = True
for chunk in pd.read_csv("./lineitem.tbl", sep='|', header=None, usecols=list(range(0,15)), chunksize=chunkSz):
    dumpCol(chunk[0],  "./tpchCols/liOrderKey", False, tpchStrLut, first)
    dumpCol(chunk[1],  "./tpchCols/liPartKey", False, tpchStrLut, first)
    dumpCol(chunk[2],  "./tpchCols/liSuppKey", False, tpchStrLut, first)
    dumpCol(chunk[3],  "./tpchCols/liLineNumber", False, tpchStrLut, first)
    dumpCol(chunk[4],  "./tpchCols/liQuantuty", False, tpchStrLut, first)
    dumpCol(chunk[5],  "./tpchCols/liExtendedPrice", False, tpchStrLut, first)
    dumpCol(chunk[6],  "./tpchCols/liDiscount", False, tpchStrLut, first)
    dumpCol(chunk[7],  "./tpchCols/liTax", False, tpchStrLut, first)
    dumpCol(chunk[8],  "./tpchCols/liReturnFlag", False, tpchStrLut, first)
    dumpCol(chunk[9],  "./tpchCols/liLineStatus", False, tpchStrLut, first)
    dumpCol(chunk[10], "./tpchCols/liShipDate", False, tpchStrLut, first)
    dumpCol(chunk[11], "./tpchCols/liCommitDate", False, tpchStrLut, first)
    dumpCol(chunk[12], "./tpchCols/liReceiptDate", False, tpchStrLut, first)
    dumpCol(chunk[13], "./tpchCols/liShipInstruct", False, tpchStrLut, first)
    dumpCol(chunk[14], "./tpchCols/liShipMode", False, tpchStrLut, first)
    first = False

# Process orders.tbl in chunks (no pad0 for Order table because its key column is treated as UUID rather than incrementing)
first = True
for chunk in pd.read_csv("./orders.tbl", sep='|', header=None, usecols=list(range(0,6)), chunksize=chunkSz):
    dumpCol(chunk[0],  "./tpchCols/ordOrderKey", False, tpchStrLut, first)
    dumpCol(chunk[1],  "./tpchCols/ordCustKey", False, tpchStrLut, first)
    dumpCol(chunk[2],  "./tpchCols/ordOrderStatus", False, tpchStrLut, first)
    dumpCol(chunk[3],  "./tpchCols/ordTotalPrice", False, tpchStrLut, first)
    dumpCol(chunk[4],  "./tpchCols/ordOrderDate", False, tpchStrLut, first)
    dumpCol(chunk[5],  "./tpchCols/ordOrderPriority", False, tpchStrLut, first)
    first = False
df = pd.read_csv("./partsupp.tbl", sep='|', header=None, usecols=list(range(0,4)))
dumpCol(df[0],  "./tpchCols/pasuPartKey", False, tpchStrLut, True)
dumpCol(df[1],  "./tpchCols/pasuSuppKey", False, tpchStrLut, True)
dumpCol(df[2],  "./tpchCols/pasuAvailQty", False, tpchStrLut, True)
dumpCol(df[3],  "./tpchCols/pasuSupplyCost", False, tpchStrLut, True)

df = pd.read_csv("./supplier.tbl", sep='|', header=None, usecols=(3,5))
dumpCol(df[3],  "./tpchCols/suppNationKey", True, tpchStrLut, True)
dumpCol(df[5],  "./tpchCols/suppAcctBal", True, tpchStrLut, True)
df = pd.read_csv("./part.tbl", sep='|', header=None, usecols=(2,3,4,5,6,7))
dumpCol(df[2], "./tpchCols/partName", True, tpchStrLut, True)
dumpCol(df[3], "./tpchCols/partMfgr", True, tpchStrLut, True)
dumpCol(df[4], "./tpchCols/partBrand", True, tpchStrLut, True)
dumpCol(df[5], "./tpchCols/partType", True, tpchStrLut, True)
dumpCol(df[6], "./tpchCols/partSize", True, tpchStrLut, True)
dumpCol(df[7], "./tpchCols/partContainer", True, tpchStrLut, True)
df = pd.read_csv("./customer.tbl", sep='|', header=None, usecols=(3,5,6))
dumpCol(df[3], "./tpchCols/custNation", True, tpchStrLut, True)
dumpCol(df[5], "./tpchCols/custAcctBal", True, tpchStrLut, True)
dumpCol(df[6], "./tpchCols/custMktSegment", True, tpchStrLut, True)
'''
