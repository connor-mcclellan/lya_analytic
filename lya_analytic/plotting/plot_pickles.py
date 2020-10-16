import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

files = glob('*.p')

def solution(x):
    return np.exp(x)

for i, f in enumerate(sorted(files)):
    print(f)
    dump = pickle.load(open(f, 'rb'))
    plt.plot(dump[0], dump[1], label=f.split('.p')[0], marker='o', ms=2, alpha=0.5)
    if i==0:
        plt.plot(dump[0], np.log10(solution(dump[0])), label='solution', marker='o', ms=2, alpha=0.5)


plt.legend()
plt.xlabel('t')
plt.ylabel('log x')
plt.show()
