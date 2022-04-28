import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})

def loadspec(infile):
    athena_spectrum = np.load(infile, allow_pickle=True).flat[0]

    counts = athena_spectrum['counts'][0, 0, 0, :]
    xbins = athena_spectrum['xfaces']
    errs = np.sqrt(counts)
    print(np.sum(counts))
    norm = np.diff(xbins) * np.sum(counts)
    counts /= norm
    errs /= norm
    xbins = (xbins[1:]+xbins[:-1])/2
    return xbins, counts, errs

x_ft, Hp, Hsp, Hh, xc, count, err, x0, xinit, _, _, _, _, _, _, a, tau0 = pickle.load(open('binx_output_tau10000000.0.p', 'rb'))

xbins, counts, errs = loadspec("acc10000_dl_1m.out1.npy")
oax, oac, oae = loadspec("acc1_dl_50k.out1.npy")
xc, count, err = loadspec('acc1000.out1.npy')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax1.errorbar(oax, oac, yerr=oae, fmt='.', marker='^', alpha=0.7, ms=2, elinewidth=0.5, linewidth=0., capsize=0.25, color='gray', label='Overaccelerated')
ax1.errorbar(xbins, counts, yerr=errs, fmt='.', marker='s', alpha=0.7, ms=1.25, elinewidth=0.5, linewidth=0., capsize=0.25, color='c', label='Accelerated')
ax1.errorbar(xc, count, yerr=err, fmt='.', marker='o', ms=0.5, alpha=0.7, elinewidth=0.5, linewidth=0., capsize=0.25, color='k', label='Standard')

ax2.scatter(oax, oac-count, marker='^', alpha=0.7, c='gray', s=2)
ax2.scatter(xbins, counts-count, marker='s', alpha=0.7, c='c', s=1.25)

ax1.legend(bbox_to_anchor=(1.0, 0.7), loc='upper left', fontsize='x-small', frameon=False)
ax2.axhline(0.0, ls='--', c='gray', lw=1, alpha=0.25)
ax1.set_yscale('log')
ax2.set_xlabel('x')
ax1.set_ylabel('P(x)')
ax2.set_ylabel('Residuals')
plt.subplots_adjust(top=0.959,
bottom=0.15,
left=0.141,
right=0.773,
hspace=0.144,
wspace=0.2)
plt.show()

