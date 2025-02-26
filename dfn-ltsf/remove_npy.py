import glob
import os

paths = glob.glob("*/*/*.npy")
print(len(paths))
print(paths)

for path in paths:
    os.remove(path)