from glob import glob
import numpy as np

def read_output(directory):

    # Grab n, omega, and sigma grids
    n_grid = np.load(directory+'n_grid.npy')
    omega_grid = np.load(directory+'omega_grid.npy')
    sigma_grid = np.load(directory+'sigma_grid.npy')

    # Find files in this directory
    files = glob(directory+'J_omega*.npy')

    # Create empty 4d array to store data
    J = np.zeros((len(n_grid), len(sigma_grid), len(omega_grid), 2))

    for i in range(len(n_grid)):
        for j in range(len(omega_grid)):
            try:
                Jdump = np.load(directory+'J_omega{}_n{}.npy'.format(j, i))
                J[i, :, j, 0] = Jdump[:, 0] # REAL PART
                J[i, :, j, 1] = Jdump[:, 1] # IMAG PART
            except:
                print('Missing data: J_omega{}_n{}.npy'.format(j, i))
                nans = np.full((len(sigma_grid)), np.nan)
                J[i, :, j, 0] = nans # REAL PART
                J[i, :, j, 1] = nans # IMAG PART
    return J

if __name__ == "__main__":
    directory = "./outputs/n8_sigma1e6_omega64/"
    J = read_output(directory)
