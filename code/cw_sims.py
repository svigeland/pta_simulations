#!/usr/bin/env python

from __future__ import division
import numpy as np
import glob
import os
import sys

import libstempo as T2
import libstempo.toasim as LT

from enterprise.signals import parameter
from enterprise.pulsar import Tempo2Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
import enterprise.constants as const

from enterprise_extensions import models
from enterprise_extensions import model_utils
from enterprise_extensions.frequentist import F_statistic

import logging
logging.getLogger().setLevel(logging.ERROR)


day_in_sec = 24.*3600
year_in_sec = 365.25*day_in_sec


def filter_by_mjd(psrs, end_time):
    
    """Filter the TOAs by MJD"""
    #### Filter TOAs by time ##########
    idxs = []
    first_toa = np.amin([p.toas.min() for p in psrs])/day_in_sec

    for psr in psrs:
        psr.filter_data(start_time=first_toa, end_time=end_time)
        if psr.toas.size==0:
            idxs.append(psrs.index(psr))
        else:
            timespan = (psr.toas[-1]-psr.toas[0])/year_in_sec
            if timespan < 3.0:
                idxs.append(psrs.index(psr))

    #### Remove empty pulsars, Reverse to keep correct idx order.

    for ii in reversed(idxs):
        del psrs[ii]

    return psrs


def compute_max_chirpmass(log10_fgw, q=1):
    """
    Function to compute the maximum chirp mass of a SMBHB that can emit at a given GW frequency.
    The GW frequency cutoff is defined as the frequency at the ISCO (a=0)
    :param fgw: The GW frequency [Hz]
    :param q: The mass ratio (default is q=1)

    return: log10 of the chirp mass in solar masses
    """

    return -log10_fgw - np.log10(6**(1.5)*np.pi) + 0.6*np.log10(q/(1+q)**2) - np.log10(const.Tsun)


def make_sim(datadir, fgw, h, endtime=None, psrlist=None):

    #make libstempo pulsar objects
    parfiles = sorted(glob.glob(datadir + '*.par'))
    timfiles = sorted(glob.glob(datadir + '*.tim'))
    
    if psrlist is not None:
        psrlist = list(np.loadtxt(psrlist, dtype='str'))
    else:
        psrlist = [p.split('/')[-1][:-4] for p in parfiles]

    libs_psrs = []

    for p,t in zip(parfiles, timfiles):
        
        if p.split('/')[-1][:-4] in psrlist:

            libs_psrs.append(T2.tempopulsar(p, t, maxobs=30000))

    Tmaxyr = np.array([(max(p.toas()) - min(p.toas()))/3.16e7 for p in libs_psrs]).max()

    # draw parameter values
    gwtheta = np.arccos(np.random.uniform(-1, 1))
    gwphi = np.random.uniform(0, 2*np.pi)
    phase0 = np.random.uniform(0, 2*np.pi)
    inc = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, np.pi)
    
    coal = True
    while coal:
        mc = 10**np.random.uniform(6, min(compute_max_chirpmass(np.log10(fgw)),10))
        tcoal = 2e6 * (mc/1e8)**(-5/3) * (fgw/1e-8)**(-8/3)
        if tcoal > Tmaxyr:
            coal = False

    dist = 4 * np.sqrt(2/5) * (mc*4.9e-6)**(5/3) * (np.pi*fgw)**(2/3) / h
    dist /= 1.0267e14   # covert distance to Mpc

    for lp in libs_psrs:
        LT.add_cgw(lp, gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc, pdist=1.0, 
                   pphase=None, psrTerm=True, evolve=False, 
                   phase_approx=True, tref=0)
    
    for lp in libs_psrs:
        lp.fit(iters=2)
        
    # convert to enterprise pulsar objects
    psrs = []
    for lp in libs_psrs:

        psr = Tempo2Pulsar(lp)

        # remove any bad toas where the residual is huge
        res_limit = np.mean(np.abs(psr.residuals)) + 5.*np.std(np.abs(psr.residuals))
        badtoas = np.argwhere(np.abs(psr.residuals) > res_limit)

        if len(badtoas) > 0:
            mask = np.ones_like(psr.toas,dtype=bool)
            for b in badtoas[0]:
                mask[b] = False
        
            model_utils.mask_filter(psr,mask)
        
        psrs.append(psr)

    if endtime is None:
        return psrs
    else:
        return filter_by_mjd(psrs, endtime)
    
    
def initialize_pta_sim(psrs, fgw):
    
    # continuous GW signal
    s = models.cw_block_circ(log10_fgw=np.log10(fgw), psrTerm=True)
    
    # white noise
    efac = parameter.Constant(1.0)
    s += white_signals.MeasurementNoise(efac=efac)
    
    # linearized timing model
    s += gp_signals.TimingModel(use_svd=True)

    model = [s(psr) for psr in psrs]
    pta = signal_base.PTA(model)
    
    return pta


def compute_det_prob(fgw, h, nreal, fap, 
                     datadir, endtime=None, psrlist=None):

    count = 0
    for ii in range(nreal):

        psrs = make_sim(datadir, fgw, h, endtime=endtime, psrlist=psrlist)
        pta = initialize_pta_sim(psrs, fgw)
        
        setpars = {}
        for psr in psrs:
            setpars.update({'{0}_efac'.format(psr.name): 1.0})
            
        fp = F_statistic.FpStat(psrs, params=setpars, pta=pta)
        fap0 = fp.compute_fap(fgw)
        
        if fap0 < fap:
            count += 1
            
    return count/nreal


# uses Brent's method to estimate the root
def compute_x(a, fa, b, fb, c, fc):
    
    R = fb/fc
    S = fb/fa
    T = fa/fc
    
    P = S*(T*(R-T)*(c-b) - (1.-R)*(b-a))
    Q = (T-1.)*(R-1.)*(S-1.)
    
    return b + P/Q


def load_outfile(outfile, hmin, hmax):

    # TO DO: make sure that I can handle all possible shapes of data
    # what if there is just one line of data?

    data = np.loadtxt(outfile)
    det_probs = np.unique(data[:,1])
    
    if len(det_probs) == 1:
        
        b, fb = None, None
        
        if np.unique(data[:,1])[0] < 0:
            
            c, fc = hmax, None
            
            idx = np.where(data[:,1] == det_probs[0])[0]
            if len(idx) > 1:
                data2 = data[idx]
                ii = np.where(data2[:,0] == max(data2[:,0]))[0]
                a, fa = data2[int(ii)]
            else:
                a, fa = data[int(idx)]

        else:
            
            a, fa = hmin, None

            idx = np.where(data[:,1] == det_probs[0])[0]
            if len(idx) > 1:
                data2 = data[idx]
                ii = np.where(data2[:,0] == min(data2[:,0]))[0]
                c, fc = data2[int(ii)]
            else:
                c, fc = data[int(idx)]
            
    else:

        b, fb = None, None

        idx = np.where(data[:,1] < 0)[0]
        if len(idx) > 1:
            data2 = data[idx]
            ii = np.where(data2[:,0] == max(data2[:,0]))[0]
            a, fa = data2[int(ii)]
        else:
            a, fa = data[int(idx)]
    
        idx = np.where(data[:,1] > 0)[0]
        if len(idx) > 1:
            data2 = data[idx]
            ii = np.where(data2[:,0] == min(data2[:,0]))[0]
            c, fc = data2[int(ii)]
        else:
            c, fc = data[int(idx)]
            
    print('Initializing from file with a = {0:.2e}, c = {1:.2e}'.format(a, c))

    return a, fa, b, fb, c, fc

                
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Gravitational Wave Search via Enterprise')

    parser.add_argument('--freq', help='GW frequency for search', default=1e-8)
    parser.add_argument('--hmin', help='Minimum GW strain', default=1e-17)
    parser.add_argument('--hmax', help='Maximum GW strain', default=1e-12)
    parser.add_argument('--htol', help='Fractional error in GW strain', default=0.1)
    parser.add_argument('--det_prob', help='Detection probability', default=0.95)
    parser.add_argument('--datadir', help='Directory of the par and tim files',
                        default='../data/partim/')
    parser.add_argument('--endtime', help='Observation end date [MJD]',
                        default=57000)
    parser.add_argument('--psrlist', help='List of pulsars to use',
                        default=None)
    parser.add_argument('--nreal', help='Number of realizations', default=100)
    parser.add_argument('--fap', help='False alarm probability', default=1e-4)
    parser.add_argument('--outdir', help='Directory to put the detection curve files', 
                        default='/home/sarah.vigeland/cw_sims/det_curve/')

    args = parser.parse_args()
    
    fgw = float(args.freq)
    nreal = int(args.nreal)
    endtime = float(args.endtime)
    psrlist = args.psrlist
    fap = float(args.fap)
    det_prob = float(args.det_prob)
    datadir = args.datadir

    outfile = '{0}/{1}.txt'.format(args.outdir, args.freq)
    
    if os.path.isfile(outfile) and os.stat(outfile).st_size > 1:
        
        print('Resuming from a previous calculation...')
        sys.stdout.flush()
        
        a, fa, b, fb, c, fc = load_outfile(outfile, float(args.hmin), float(args.hmax))
    
    else:

        a, c = float(args.hmin), float(args.hmax)
        fa, fc = None, None
        b = None

    with open(outfile, 'a') as f:
    
        if fa is None:
            fa = compute_det_prob(fgw, a, nreal, fap, 
                                  datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(a, fa))
            f.flush()

        if fc is None:
            fc = compute_det_prob(fgw, c, nreal, fap, 
                                  datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(c, fc))
            f.flush()

        # initially perform a bisection search
        while b is None:
    
            x = 10**((np.log10(a) + np.log10(c))/2)
    
            fx = compute_det_prob(fgw, x, nreal, fap, 
                                  datadir, endtime=endtime, psrlist=psrlist) - det_prob
        
            f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))
            f.flush()
    
            if np.abs(fx - fa) < 1/nreal and np.sign(fa) == np.sign(fx):
                a, fa = x, fx
            elif np.abs(fx - fc) < 1/nreal and np.sign(fc) == np.sign(fx):
                c, fc = x, fx
            else:
                b, fb = x, fx

        # use Brent's root finding algorithm to estimate the value of the root
        x = compute_x(a, fa, b, fb, c, fc)

        while np.abs(x-b) > float(args.htol)*b:
    
            fx = compute_det_prob(fgw, x, nreal, fap, 
                                  datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))
            f.flush()
        
            # if fx is very close to fa, fb, or fc, simply replace that point
            if np.abs(fx - fa) < 1/nreal:    
                a, fa = x, fx
            elif np.abs(fx - fb) < 1/nreal:
                b, fb = x, fx
            elif np.abs(fx - fc) < 1/nreal:
                c, fc = x, fx
            else:
                if np.sign(fx) == np.sign(fc):
                    c, fc = x, fx            
                else:
                    a, fa = b, fb
                    b, fb = x, fx
        
            x = compute_x(a, fa, b, fb, c, fc)
            
        fx = compute_det_prob(fgw, x, nreal, fap, 
                              datadir, endtime=endtime, psrlist=psrlist) - det_prob
        f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))
        f.flush()
