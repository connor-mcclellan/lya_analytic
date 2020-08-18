import numpy as np
from solutions.util import j0
from tqdm import tqdm

def evaluate_Jrt(path, r, sigma, t, n, omega):

    d_omega = np.diff(omega)[0]
    R = r[-1]
    Jrt = np.zeros((len(r), len(sigma), len(t)), dtype=np.complex)

    # Sum & discretized integral
    for k in tqdm(range(len(n))):
        for l in tqdm(range(len(omega))):
            kappa_n = n[k] * np.pi / R

            # Load in data for this n and omega
            Jdump = np.load(path+'J_omega{}_n{}.npy'.format(l, k))
            J_n_sigma_omega = Jdump[:, 0] + 1j*Jdump[:, 1]

            for i in tqdm(range(len(t))):
                for j in tqdm(range(len(r))):
                    # Eq 34
                    Jrt[j, :, i] += d_omega / (2.*np.pi) * J_n_sigma_omega * j0(kappa_n, r[j]) * np.exp(-1j*omega[l]*t[i])
    return Jrt


if __name__ == "__main__":
    r = np.linspace(1e-1, 1e11, int(1e1))
    t = np.linspace(0, 30, int(1e1))
    
    path = './outputs/n8_sigma1e6_omega64/'
    n = np.load(path+'n_grid.npy')
    omega = np.load(path+'omega_grid.npy')
    sigma = np.load(path+'sigma_grid.npy')

    Jrt = evaluate_Jrt(path, r, sigma, t, n, omega)
    np.save(path+'Jrt_t0to30.npy', Jrt)
