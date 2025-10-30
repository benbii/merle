import numpy as np
import multiprocessing

def finiteZipf(n, size, skew):
    ranks = np.arange(1, n + 1)
    probabilities = 1 / ranks**skew
    probabilities /= probabilities.sum()  # Normalize probabilities
    samples = np.random.choice(ranks, size=size, p=probabilities)
    return samples

def dumpInvlst(n, size, skew):
    samples = finiteZipf(n=n, size=size, skew=skew)
    dens = np.bincount(samples - 1, minlength=n)
    for i in range(n):
        invList = np.random.choice(size, dens[i], replace=False)
        with open(f"tbl/zipf/{int(skew*10)}l{i}.txt", "w") as file:
            file.write(",".join(map(str, invList)))
    return skew

# for s in [1.2, 1.4, 1.6, 1.8, 2.0]:
#     dumpInvlst(32, int(1e8), s)
def foo(s): return dumpInvlst(32, int(1e8), s)
with multiprocessing.Pool(5) as pool:
    print(pool.map(foo, [1.2, 1.4, 1.6, 1.8, 2.0]))
