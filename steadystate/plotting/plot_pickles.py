import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

files = glob('*fracerr.p')

for i, f in enumerate(sorted(files)):
    print(f)
    dump = pickle.load(open(f, 'rb'))
    plt.plot(dump[0], dump[1], label=f.split('.p')[0], marker='o', ms=2, alpha=0.5)
#    if i==1:
#        plt.plot(dump[0], np.log10(np.e) * (dump[0]+1000), label='solution', marker='o', ms=2, alpha=0.5)


plt.legend()
plt.xlabel('t')
plt.ylabel('fractional error in log x')
plt.show()
