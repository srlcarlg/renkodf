import sys
from time import perf_counter

import pandas as pd
from renkodf import Renko

df_ticks = pd.read_parquet("../examples/data/EURGBP_T1_cT.parquet")
df_ticks.rename(columns={'bid': 'close'}, inplace=True)

# ~14,7 Million
for _ in range(6):
    df_ticks = pd.concat([df_ticks, df_ticks])
print(len(df_ticks))

t0 = perf_counter()
r1 = Renko(df_ticks, 0.0003)
t1 = perf_counter()

# ~10 seconds
print(f"{round(t1 - t0, 2)} seconds")

# final size
# ~1.56 mb
# ~0.024 mb without concat loop
print("SIZE : ",
      sys.getsizeof(r1._df_renko) / 1024.0 ** 2,
      "MiB")

"""
# Not suitable for massive datasets, severe slowdown on loop performance.
# also, strongly overhead => https://www.mail-archive.com/python-list@python.org/msg443129.html

df_ticks = pd.read_parquet("../examples/data/EURGBP_T1_cT.parquet")
df_ticks.rename(columns={'bid': 'close'}, inplace=True)
print(len(df_ticks))

import tracemalloc
tracemalloc.start()

r1 = Renko(df_ticks, 0.0003)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / (1024 * 1024):.2f} MB")
print(f"Peak memory usage: {peak / (1024 * 1024):.2f} MB")
"""
