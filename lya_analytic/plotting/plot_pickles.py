import pickle
from glob import glob
import matplotlib.pyplot as plt

files = glob('*.p')

for f in sorted(files):
    print(f)
    dump = pickle.load(open(f, 'rb'))
    plt.plot(dump[0], dump[1], label=f.split('.p')[0], marker='o', ms=2, alpha=0.5)
plt.legend()
plt.xlabel('t')
plt.ylabel('log x')
plt.show()
