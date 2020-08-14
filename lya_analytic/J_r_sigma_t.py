import numpy as np
from solutions.util import j0
from tqdm import tqdm

def evaluate_Jrt(path, r, t, n, omega):

    d_omega = np.diff(omega)[0]
    R = r[-1]
    Jrt = np.zeros((len(r), len(t)))

    # Sum & discretized integral

    # Reversing these loops would be faster, but might have to get clever
    for i in tqdm(range(len(t))):
        for j in tqdm(range(len(r))):
            n_sum = 0.
            for k in tqdm(range(len(n))):
                omega_integral = 0.
                for l in tqdm(range(len(omega)):
                    kappa_n = n[k] * np.pi / R

                    # Load in data for this n and omega
                    with np.load(path+'J_omega{}_n{}.npy'.format(l, k)) as Jdump:
                        J_n_sigma_omega = Jdump[:, 0] + 1j*Jdump[:, 1]

                    # Eq 34
                    omega_integral += d_omega / (2.*np.pi) * J_n_sigma_omega * j0(kappa_n / r[j]) * np.exp(-1j*omega[l]*t[i])
                n_sum += omega_integral
            Jrt[j, i] = n_sum
    return Jrt


if __name__ == "__main__":
    r = np.linspace(0, 1e11, 1e3)
    t = np.linspace(0, 1e3, 1e3)
    
    path = './outputs/n8_sigma1e6_omega64/'
    n = np.load(path+'n_grid.npy')
    omega = np.load(path+'omega_grid.npy')

    Jrt = evaluate_Jrt(path, r, t, n, omega)
    np.save(path+'Jrt.npy', Jrt)
