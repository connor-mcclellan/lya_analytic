import numpy as np
import scipy.optimize as opt

def prob_ct_tau(t,tau):
    """
    Sets up the time/path distributions for a point source at origin in the limit for any tau
    in diffusion limit
    """    
    y = np.exp(-t)
    prob = 0.    
    n = 1
    #sol = sci.optimize.root_scalar(tanf,args=(tau),x0=np.pi*n,x1=(np.pi+0.1)*n)    
    sol = opt.root_scalar(tanf,args=(tau),bracket=[0.51*np.pi,1.49*np.pi])
    lamn = sol.root
    dlamn = lamn
    In = 0.5*tau*(1+(1.5*tau-1)/((1.5*tau-1)**2+lamn*lamn))
    yn2 = (y)**(lamn*lamn/(np.pi*np.pi))*np.cos(lamn)*(1.5*tau**2/(1-1.5*tau)/In)
    #print n,lamn,lamn/np.pi,yn2
    # Compute the sum to the limit of double precision
    while (abs(yn2) > 1.e-17 ): 
        prob = prob + yn2
        n = n+1;
        #sol=opt.root_scalar(tanf,args=(tau),x0=np.pi*n,x1=(np.pi+0.1)*n)
        bracket=[lamn+(1.-0.1/n)*dlamn,lamn+(1.+0.1/n)*dlamn]
        #print 'l: ',lamn/np.pi,dlamn/np.pi
        sol=opt.root_scalar(tanf,args=(tau),bracket=bracket)
        dlamn = sol.root-lamn
        lamn = sol.root
        In = 0.5*tau*(1+(1.5*tau-1)/((1.5*tau-1)**2+lamn*lamn))
        yn2 = (y)**(lamn*lamn/(np.pi*np.pi))*np.cos(lamn)*(1.5*tau**2/(1-1.5*tau)/In)
        #print n,lamn,lamn/np.pi,yn2,prob    
    return prob


def tanf(x,tau):
    """
    Used by prob_ct_tau to evalue time/path distribution
    """
    return np.tan(x) - x/(1.-1.5*tau)
