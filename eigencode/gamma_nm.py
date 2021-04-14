from constants import fundconst
from parameters import Parameters
import numpy as np
import matplotlib.pyplot as plt
fc=fundconst()

mmax=100

def gamma(n, m, p): 
     return 2**(-1/3) * np.pi**(13/6)*n**(4/3)*(m-7/8)**(2/3)*fc.clight/p.radius/(p.a * p.tau0)**(1/3)    


filename = './data/eigenmode_data_xinit0_tau1e7_n6_m100_xuniform_masteronly.npy'
array = np.load(filename, allow_pickle=True, fix_imports=True, )
energy = array[0]
temp = array[1]
tau0 = array[2]
radius = array[3]
alpha_abs = array[4]
prob_dest = array[5]
xsource = array[6]
nmax = array[7]
nsigma = array[8]
nomega = array[9]
tdiff = array[10]
sigma = array[11]
ssoln = array[12]
Jsoln = array[13]
p = Parameters(temp,tau0,radius,energy,xsource,alpha_abs,prob_dest,nsigma,nmax)


color = ['c', 'm', 'k', 'r', 'limegreen', 'b']
for n in range(1, nmax+1):
    gamma_analytic = n**(-4/3)*gamma(n, np.arange(mmax), p)
    gamma_sweep = -n**(-4/3)*ssoln[n-1][:mmax]

    plt.plot(np.arange(0, mmax), 1/gamma_analytic, '--', c=color[n-1], alpha=0.5)#, label='$\gamma_{nm}$ analytic')
    plt.plot(np.arange(1, mmax+1), 1/gamma_sweep, '-', c=color[n-1], alpha=0.5)#, label='$\gamma$ sweep')
#plt.title('n={}'.format(n))
plt.ylabel('$n^{4/3}t_{nm}(s)$')
plt.xlabel('m')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.show()
    #plt.savefig('gamma_n{}.pdf'.format(n))
    #plt.close()


'''
for m in range(0, 40):
    gamma_analytic = gamma(np.arange(1, nmax+1), m, p)
    gamma_sweep = -ssoln[:, m]

    n = np.arange(1, 7)
    plt.plot(n, gamma_analytic, '--', label='$\gamma_{nm}$ analytic')
    plt.plot(n, gamma_sweep, '-', label='$\gamma$ sweep')
    plt.title('m={}'.format(m))
    plt.ylabel('\gamma')
    plt.xlabel('n')
    plt.tight_layout()
    plt.legend()
    plt.show()
#    plt.savefig('gamma_m{}.pdf'.format(n))
#    plt.close()
'''
