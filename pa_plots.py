import pdb
from astropy.utils.console import ProgressBar
from scipy.interpolate import interp1d
import math
import numpy as np
import astropy.constants as c
from solutions.util import read_bin, voigtx_fast, Line, Params, scinot
from solutions import ftsoln
from solutions import fits
from solutions.prob_ct_tau import prob_ct_tau
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **{'family': 'serif',
                         'serif': ['Computer Modern Roman']})
from matplotlib.ticker import LogFormatterExponent
from pathlib import Path
import pickle


'''
An attempt to recreate Phil's plots for his Monte Carlo data.
'''

def get_input_info(filename):
    f = open(filename, 'r')
    g = f.readlines()
    f.close()
    for i in g:
        for j in i.split():
            k = j.find("tau0=")
            if k != -1:
                tau0 = j[k + 5:]
            k = j.find("temp=")
            if k != -1:
                temp = j[k + 5:]
            k = j.find("x_init=")
            if k != -1:
                x_init = j[k + 7:]
            k = j.find("prob_abs=")
            if k != -1:
                prob_abs = j[k + 9:]
            k = j.find("rmax=")
            if k != -1:
                rmax = j[k + 5:]
    return np.float(tau0), np.float(temp), np.float(
        x_init), np.float(prob_abs), np.float(rmax)


def bin_x(x, n, mytitle, filename, tau0, xinit, temp, radius, L, delta, a, p):
    count = np.zeros(n)
    x0 = 2.0 * (a * tau0)**0.333
    # n bins, n+1 bin edges
    xmax = np.amax(x)
    xmin = np.amin(x)
    dx = (xmax - xmin) / n
    xe = np.zeros(n + 1)
    for i in range(n + 1):
        xe[i] = xmin + i * dx
    xc = np.zeros(n)
    for i in range(n):
        xc[i] = 0.5 * (xe[i] + xe[i + 1])

    for xval in x:
        if xval <= xmin:
            j = 0
        elif xval >= xmax:
            j = n - 1
        else:
            j = np.rint(math.floor((xval - xmin) / dx))
            j = int(j)  # j=j.astype(int)
        if (xe[j + 1] - xval) * (xval - xe[j]) < 0.0:
            print(j, (xe[j + 1] - xval) * (xval - xe[j]))
        count[j] = count[j] + 1.0
    err = np.sqrt(np.abs(1.0 * count))
    count = count / x.size / dx
    err = err / x.size / dx

    x_ft, sigma_ft, Jp_ft, Hp_ft, Jsp_ft, Hsp_ft, Jh_ft, Hh_ft = ftsoln.ftsoln_wrapper(
        tau0, xinit, temp, radius, L)
    norm = 4.0 * np.pi * radius**2 * delta * 4.0 * np.pi / L
    print('norm: ', norm)
    print("check norm of data=", np.sum(count) * dx)

    # CM: Interpolate phix*H on uniform x grid
    print('Interpolating solutions on uniform x grid...\n')

    # Make an array of uniformly spaced x-values (min, max, npoints)
    xuniform = np.linspace(np.min(x_ft), np.max(x_ft), len(x_ft))

    # Find sigma at each x-value
    sigma_xuniform = (p.beta / a) * xuniform**3.

    # Calculate line profile at all the x points needed
    phix = voigtx_fast(a, x_ft)
    phix_xuniform = voigtx_fast(a, xuniform)
    phix_xc = voigtx_fast(a, xc)

    # Interpolate solutions from _ft points
    hsp_interp = interp1d(x_ft, Hsp_ft * phix * norm)
    hp_interp = interp1d(x_ft, Hp_ft * phix * norm)
    hh_interp = interp1d(x_ft, Hh_ft * phix * norm)

    # Apply interpolation to uniformly distributed x values, divide by line
    # profile at those x positions
    hsp_xuniform = hsp_interp(xuniform) / phix_xuniform
    hp_xuniform = hp_interp(xuniform) / phix_xuniform
    hh_xuniform = hh_interp(xuniform) / phix_xuniform

    ymax1 = np.amax((Hp_ft) * norm)
    ymax2 = np.amax((Hsp_ft) * norm)
    ymax3 = np.amax((Hh_ft) * norm)
    ymax4 = np.amax(count)
    ymax = max([ymax1, ymax2, ymax3, ymax4]) * 1.1
    ymin = np.amin(Hh_ft * norm) * 1.1

# Solutions with only H_0 subtracted
#    plt.figure()
#    plt.plot(x_ft, (Hp_ft - Hsp_ft) * norm, label=r'$H_{\rm d} - H_0$')
#    plt.plot(x_ft, Hh_ft * norm, label=r'$H_{\rm bc}$')
#    plt.errorbar(xc, count - hsp_ft_func(xc), yerr=err, fmt='.', label="Monte Carlo - $H_0$")
#    plt.title(mytitle)
#    plt.legend(loc='best')
#    plt.xlabel(r'$x$', fontsize=15)
#    plt.ylabel(r'$P(x)$', fontsize=15)
#    plt.savefig("./plots/1m_x_pdf_subtracted.pdf", format='pdf')
#    plt.close()

    return (xuniform, hp_xuniform, hsp_xuniform, hh_xuniform, xc, count, err, x0, xinit, ymin, ymax, phix_xc, hp_interp, hsp_interp, hh_interp)


def residual_plot(xuniform, hp_xuniform, hsp_xuniform, hh_xuniform, xc, count, err, x0, xinit, ymin, ymax, phix_xc, hp_interp, hsp_interp, hh_interp, logscale=False):

#    color = plt.cm.coolwarm(np.arange(4)/3)
    color = ['b', 'r', 'orange', 'purple', 'gray']
    alpha = 0.7

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]}, figsize=(7, 5))

    # Top left panel: linear-scale solutions
    ax1.axvline(xinit, c=color[4], lw=1, alpha=0.5)
    ax1.plot(xuniform, hp_xuniform, '--', label=r'$H_{\rm d}$', alpha=alpha, c=color[0], linewidth=1)
    ax1.plot(xuniform, hsp_xuniform + hh_xuniform, '-', label=r'$H_{\rm 0+bc}$', alpha=alpha, c=color[1], linewidth=1)
    ax1.plot(xuniform, hsp_xuniform, '-.', label=r'$H_0$', alpha=alpha, c=color[2], linewidth=1)
    ax1.errorbar(xc, count, yerr=err, fmt='.', label="MC", alpha=0.75, ms=3., c='k', elinewidth=0.25, capsize=0.5)

    ax1.text(xinit+0.5, 0.03, r'x$_{\rm init}$', rotation=90, fontsize=8)
    ax1.set_xlim((min(xc)-2, max(xc)+2))
    ax1.set_ylabel(r'$P(x)$')
    ax1.grid(linestyle='--', alpha=0.25)
    ax1.set_ylim((ymin-0.005, ymax))
    ax1.plot(xuniform, hh_xuniform, ':', label=r'$H_{\rm bc}$', alpha=alpha, c=color[3], linewidth=1)
    ax1.legend(bbox_to_anchor=(1.04, 0.8), loc='upper left', fontsize='x-small', frameon=False)

    # Top right panel: log-scale solutions
    ax2.plot(xuniform, hp_xuniform, '--', label=r'$H_{\rm d}$', alpha=alpha, c=color[0], linewidth=1)
    ax2.plot(xuniform, hsp_xuniform + hh_xuniform, '-', label=r'$H_{\rm 0+bc}$', alpha=alpha, c=color[1], linewidth=1)
    ax2.plot(xuniform, hsp_xuniform, '-.', label=r'$H_0$', alpha=alpha, c=color[2], linewidth=1)
    ax2.errorbar(xc, count, yerr=err, fmt='.', label="MC", alpha=0.75, ms=3., c='k', elinewidth=0.25, capsize=0.5)
    ax2.axvline(xinit, c=color[4], lw=1, alpha=0.5)
    ax2.set_xlim((min(xc)-2, max(xc)+2))
#    ax2.text(1.23, 1, mytitle, transform=ax2.transAxes, ha='left', va='top')
    ax2.plot(xuniform, np.abs(hh_xuniform), ':', label=r'$H_{\rm bc}$', alpha=alpha, c=color[3], linewidth=1)
    ax2.set_ylim((0.00000005, 0.1))
    ax2.set_ylabel('$\log{P(x)}$')
    ax2.set_yscale('log')
    ax2.grid(linestyle='--', alpha=0.25)
    ax2.yaxis.set_major_formatter(LogFormatterExponent())
#    ax2.yaxis.tick_right()

    # Bottom left panel: linear-scale residuals
#    ax3.set_ylim((-0.1, 1))
    ax3.axvline(xinit, c=color[4], lw=1, alpha=0.5)
    ax3.plot(xc, hp_interp(xc)/phix_xc - count, '.', label=r'$H_{\rm d} - \rm MC$', alpha=alpha, c=color[0], linewidth=1, marker='^', markersize=2)
    ax3.plot(xc, (hsp_interp(xc) + hh_interp(xc))/phix_xc - count, '.', label=r'$H_{\rm 0 + bc} - \rm MC$', alpha=alpha, c=color[1], linewidth=1, marker='s', markersize=2)
    ax3.plot(xc, hsp_interp(xc)/phix_xc - count, '.', label=r'$H_{0} - \rm MC$', alpha=alpha, c=color[2], linewidth=1, marker='o', markersize=2)

    ax3.grid(linestyle='--', alpha=0.25)
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel('Residuals')
    ax3.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='x-small', frameon=False)

    plt.subplots_adjust(top=0.97,
bottom=0.11,
left=0.11,
right=0.80,
hspace=0.1,
wspace=0.0)

    plt.show()
#    plt.savefig("./plots/"+filename+"/pdf_xinit{:.1f}.pdf".format(xinit), format='pdf')
#    plt.savefig("./plots/pdf_xinit{:.1f}.pdf".format(xinit), format='pdf')
    plt.close()


def comparison_plot(*args):

    color = ['b', 'r', 'orange', 'purple', 'gray']
    alpha = 0.7

    fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]}, figsize=(7, 5))
    for i, arg in enumerate(args):
        xuniform, hp_xuniform, hsp_xuniform, hh_xuniform, xc, count, err, x0, xinit, ymin, ymax, phix_xc, hp_interp, hsp_interp, hh_interp = arg
        axi = ax[i]
        
        #linear-scale solutions
        axi.axvline(xinit, c=color[4], lw=1, alpha=0.5)
        axi.plot(xuniform, hp_xuniform, '--', label=r'$H_{\rm d}$', alpha=alpha, c=color[0], linewidth=1)
        axi.plot(xuniform, hsp_xuniform + hh_xuniform, '-', label=r'$H_{\rm 0+bc}$', alpha=alpha, c=color[1], linewidth=1)
        axi.plot(xuniform, hsp_xuniform, '-.', label=r'$H_0$', alpha=alpha, c=color[2], linewidth=1)
        axi.errorbar(xc, count, yerr=err, fmt='.', label="MC", alpha=0.75, ms=3., c='k', elinewidth=0.25, capsize=0.5)

        axi.text(xinit+0.5, 0.03, r'x$_{\rm init}$', rotation=90, fontsize=8)
        axi.set_xlim((min(xc)-2, max(xc)+2))
        axi.set_ylabel(r'$P(x)$')
        axi.grid(linestyle='--', alpha=0.25)
        axi.set_ylim((ymin-0.005, ymax))
        axi.plot(xuniform, hh_xuniform, ':', label=r'$H_{\rm bc}$', alpha=alpha, c=color[3], linewidth=1)
        axi.legend(bbox_to_anchor=(1.04, 0.8), loc='upper left', fontsize='x-small', frameon=False)

    plt.subplots_adjust(top=0.97,
                        bottom=0.11,
                        left=0.11,
                        right=0.80,
                        hspace=0.1,
                        wspace=0.0)
    plt.show()


def bin_time(t, n):

    # Use matplotlib.pyplot.hist to bin and normalize data
    count, bins, _ = plt.hist(t, bins=np.logspace(np.log10(min(t)), np.log10(max(t)), n), density=True)

    # We don't actually need the figure though, so clear the axis
    plt.cla()

    # Calculate bin centers. 1/2 * (bin 1 through bin n + bin 0 to bin n-1)
    tc = 0.5 * (bins[1:] + bins[:-1])

    # Phil's theory line (close to log normal fit)
    xbar = np.average(np.log10(t))
    xsqbar = np.average((np.log10(t))**2)
    xsigma = np.sqrt(xsqbar - xbar**2)
    theory = 1.0 / (np.sqrt(2.0 * np.pi) * xsigma) * \
        np.exp(-(np.log10(tc) - xbar)**2 / (2.0 * xsigma**2))

    return tc, count, theory

def rms_error(data, solution):
    x, mc_y, mc_err = data
    sol_x, _, sol_hsp, sol_hh = solution
    sol_y = interp1d(sol_x, sol_hsp + sol_hh)(x)

    return np.sqrt(np.sum((mc_y - sol_y)**2.)/len(mc_y))

def relative_error(data, solution):
    x, mc_y, mc_err = data
    sol_x, _, sol_hsp, sol_hh = solution
    sol_y = interp1d(sol_x, sol_hsp + sol_hh)(x)

    mask = [mc_err>0.]
    return np.sum(np.abs(mc_y[mask] - sol_y[mask])/mc_err[mask])

def multiplot_time(tc, t0, tau0):
    prob = np.zeros((len(t0), len(tc)))
    for j in range(len(t0)):
        for i in range(len(tc)):
            prob[j, i] = prob_ct_tau(tc[i] / t0[j], tau0) / t0[j]
        plt.plot(tc, 2.3 * prob[j] * tc, '--', label='t0={:.2f}'.format(t0[j]), alpha=0.25)


if __name__ == '__main__':

    filename = '1M tau0_10000000.0_xinit_6.0_temp_10000.0_probabs_0.0'
    data_dir = '/home/connor/Documents/lya_analytic/data/'+filename+'/'
#    Path("./plots/"+filename).mkdir(parents=True, exist_ok=True)


    generate_new = False

    lya = Line(1215.6701, 0.4164, 6.265e8)
    p = Params(line=lya, temp=1e4, tau0=1e7, num_dens=1701290465.5139434, 
           energy=1., R=1e11, sigma_source=0., n_points=1e4)
    L = 1.0
    tau0, temp, xinit, prob_abs, radius = get_input_info(data_dir + 'input')
    vth = np.sqrt(2.0 * c.k_B.cgs.value * temp / c.m_p.cgs.value)
    delta = lya.nu0 * vth / c.c.cgs.value
    a = lya.gamma / (4.0 * np.pi * delta)
    
    mytitle = r'$\tau_0=${}'.format(scinot(tau0))+'\n'+r'$x_{{\rm init}}={:.1f}$'.format(xinit)+'\n'+'$T=${}'.format(scinot(temp))
    

    #mu, x, time = read_bin(data_dir)
    #soln, mc = bin_x(x, 64, mytitle, filename, tau0, xinit, temp, radius, L, delta, a, p)

    if generate_new:
        mu, x, time = np.load(data_dir + 'mu_x_time.npy')  
        binx_output = bin_x(x, 64, mytitle, filename, tau0, xinit, temp, radius, L, delta, a, p)
        pickle.dump(binx_output, open('binx_output_xinit{:.1f}.p'.format(xinit), 'wb'))
        residual_plot(*binx_output)
    else:
        binx_output = pickle.load(open('binx_output_xinit{:.1f}.p'.format(xinit), 'rb'))
        residual_plot(*binx_output)
    # Test convergence of solution
#    print('RMS Deviation: ', rms_error(mc, soln))
#    print('Relative error: ', relative_error(mc, soln))

    ##### Make time plots ######
#    tc, count, theory = bin_time(time, 64)
#    plt.plot(tc, 2.3 * tc * count, ".", label="data")
#    plt.plot(tc, theory, label="theory")
#    plt.plot(tc, 2.3 * fits.lognorm_fit(tc, xdata=time)[0] * tc, '--', label='Log Norm Fit')

    # Save output array for faster plotting using escape_times.py
#    dat = np.array([tc, 2.3 * tc * count, theory])
#    np.save('./outputs/1m_bin_time', dat)

    # Series solution lines
#    t0 = np.linspace(8., 11., 5)
#    multiplot_time(tc, t0, tau0)

    # Plot labels and aesthetics
#    plt.title(mytitle)
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.legend(loc='best')
#    plt.xlabel(r'$t$', fontsize=15)
#    plt.ylabel(r'$2.3tP(t)$', fontsize=15)
#    plt.ylim((1e-4, 1e1))
#    plt.show()
#    plt.savefig("./plots/"+filename+"/1m_time_pdf.pdf", format='pdf')
#    plt.close()

