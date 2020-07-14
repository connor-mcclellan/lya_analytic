import numpy as np
import scipy as sci

def prob_ct_tau(t,tau):
    """
    Compute (derivative) of cumulative probability
    """
    y = np.exp(-t)
    prob = 0.
    n = 1
    sol = sci.optimize.root_scalar(tanf,args=(tau),bracket=[0.51*np.pi,1.49*np.pi])
    lamn = sol.root
    dlamn = lamn
    In = 0.5*tau*(1+(1.5*tau-1)/((1.5*tau-1)**2+lamn*lamn))
    yn2 = (y)**(lamn*lamn/(np.pi*np.pi))*np.cos(lamn)*(1.5*tau**2/(1-1.5*tau)/In)*(lamn*lamn/(np.pi*np.pi))
    while (abs(yn2) > 1.e-17 ): 
        prob = prob + yn2
        n = n+1;
        bracket=[lamn+(1.-0.1/n)*dlamn,lamn+(1.+0.1/n)*dlamn]
        sol=sci.optimize.root_scalar(tanf,args=(tau),bracket=bracket)
        dlamn = sol.root-lamn
        lamn = sol.root
        In = 0.5*tau*(1+(1.5*tau-1)/((1.5*tau-1)**2+lamn*lamn))
        yn2 = (y)**(lamn*lamn/(np.pi*np.pi))*np.cos(lamn)*(1.5*tau**2/(1-1.5*tau)/In)*(lamn*lamn/(np.pi*np.pi))
    return prob


def tanf(x,tau):
    """
    Used by prob_ct_tau to evalue time/path distribution
    """
    return np.tan(x) - x/(1.-1.5*tau)
