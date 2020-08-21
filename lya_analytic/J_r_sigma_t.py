import numpy as np
from solutions.util import j0
import h5py
from astropy.utils.console import ProgressBar
from scipy.interpolate import interp1d
import pdb

def evaluate_Jrt(Jnomega, r, sigma, t, output, axis=1):

    full_sigma = Jnomega['sigma'][:]
    omega = Jnomega['omega'][:]
    n = Jnomega['n'][:]
    d_omega = np.diff(omega)[0]
    R = r[-1]

    lengths = [len(r), len(sigma), len(t)]
    names = ['r', 'sigma', 't']
    arrays = [r, sigma, t]

    name = names.pop(axis)
    array = arrays.pop(axis)
    length = lengths.pop(axis)

    pb = ProgressBar(len(t)*len(r)*len(n)*len(omega)*len(sigma))

    Jt = np.zeros(len(t), dtype=np.complex)
    for i in range(len(t)):
        Jr = np.zeros(len(r), dtype=np.complex)
        for j in range(len(r)):
            Js = np.zeros(len(sigma), dtype=np.complex)
            # Sum & discretized integral
            for k in range(len(n)):
                kappa_n = n[k] * np.pi / R
                for l in range(len(omega)):
                    # Load fourier coefficients for this n and omega
                    Jdump = Jnomega['J_omega{}_n{}'.format(l, k)][:]
                    J_interp = interp1d(full_sigma, Jdump)
                    for m in range(len(sigma)):
                        iterators = [j, m, i]
                        print(iterators)
                        (Jr, Js, Jt)[axis][iterators[axis]] += d_omega / (2.*np.pi) * J_interp(sigma[m]) * j0(kappa_n, r[j]) * np.exp(-1j*omega[l]*t[i])
                        pb.update()
                    iterator = iterators.pop(axis)
                    setname = 'J_{}{}_{}{}'.format(names[0], iterators[0], names[1], iterators[1])

      # This doesn't work --- order of the loops matters. Maybe can't generalize?
            if not np.all(Js==0.):
                # Axis is sigma
                output.create_dataset(setname, data=Js)
        if not np.all(Jr==0.):
            # Axis is r
            output.create_dataset(setname, data=Js)
    if not np.all(Jt==0.):
        # Axis is r
        pdb.set_trace()
        output.create_dataset(setname, data=Jt)

    output.create_dataset(names[0], data=arrays[0])
    output.create_dataset(names[1], data=arrays[1])
    output.close()


def evaluate_H(path, r, sigma, t, n, omega):

    d_omega = np.diff(omega)[0]
    R = r[-1]
    H = np.zeros((len(r), len(sigma), len(t)), dtype=np.complex)
    full_sigma = np.load(path+'sigma_grid.npy')

    # Sum & discretized integral
    for k in tqdm(range(len(n))):
        for l in tqdm(range(len(omega))):
            kappa_n = n[k] * np.pi / R

            # Load in data for this n and omega
            Jdump = np.load(path+'J_omega{}_n{}.npy'.format(l, k))
            J_n_sigma_omega = Jdump[:, 0] + 1j*Jdump[:, 1]
            J_interp = interp1d(full_sigma, J_n_sigma_omega)

            for i in tqdm(range(len(t))):
                for j in tqdm(range(len(r))):
                    # Eq 34
                    j0_prime = np.cos(kappa_n*r[j])/r[j] - np.sin(kappa_n*r[j])/kappa_n/r[j]**2.
                    H[j, :, i] += d_omega / (2.*np.pi) * J_interp(sigma) * j0_prime * np.exp(-1j*omega[l]*t[i])
    return Jrt


if __name__ == "__main__":
    r = [1e11, ]
    t = np.linspace(0, 30, int(1e2))
    sigma_eval = [-1e6, ]
    Jnomega = h5py.File('./outputs/n4_sigma999999_omega2.hdf5', 'r')
    output = h5py.File('./outputs/r{}_sigma{}_t{}.hdf5'.format(len(r), len(sigma_eval), len(t)), 'w')
    evaluate_Jrt(Jnomega, r, sigma_eval, t, output, axis=2)
