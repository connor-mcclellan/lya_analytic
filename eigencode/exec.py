from pathlib import Path
import subprocess
import numpy as np

directory = Path(input("Path to data directory: ")).resolve()
nmin = int(input("Starting n (spatial) eigenmode: "))
nmax = int(input("Ending n (spatial) eigenmode: "))
mmax = int(input("Ending m (frequency) eigenmode: "))
cores = int(input("Number of cores: "))

nlims = list(set(np.linspace(nmin, nmax+1, cores+1).astype(int)))
for i in range(len(nlims)-1):
    subprocess.Popen(['screen', '-dm', 'python', 'efunctions.py', '--nmin', str(nlims[i]), '--nmax', str(nlims[i+1] - 1), '--mmax', str(mmax), '-p', str(directory)])
