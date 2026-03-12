import numpy as np

data = np.fromfile("synthetic_data/data_shard_0000.bin", dtype=np.float32)

print(data[:10])      # first 10 numbers
print(data.shape)     # total amount of float32 values
print(data.dtype)