from pathlib import Path
from util import construct_sol, waittime
from constants import fundconst,lymanalpha
import matplotlib.pyplot as plt
import numpy as np
import pdb

fc=fundconst()
la=lymanalpha()

directory = Path('./data/210521_m500').resolve()
Jsoln, ssoln, intJsoln, p = construct_sol(directory, 20, 500)
tlc = p.radius/fc.clight

t = tlc * np.arange(0.1,140.0,0.1)
P = waittime(Jsoln, ssoln, intJsoln, t, p)
plt.plot(t/tlc, P*tlc)
plt.yscale('log')
plt.show()
