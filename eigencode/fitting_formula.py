from pathlib import Path
from util import construct_sol
import pdb

directory = Path('./data/210521_m500').resolve()
Jsoln, ssoln, intJsoln, p = construct_sol(directory, 20, 500)
pdb.set_trace()
