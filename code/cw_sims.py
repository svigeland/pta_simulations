#!/usr/bin/env python

from __future__ import division
import numpy as np
import glob
import os
import sys
import time
import logging

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

logger = logging.getLogger(__name__)


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
    Function to compute the maximum chirp mass of a SMBHB that can emit
    at a given GW frequency.
    The GW frequency cutoff is defined as the frequency at the ISCO (a=0)
    :param fgw: The GW frequency [Hz]
    :param q: The mass ratio (default is q=1)

    return: log10 of the chirp mass in solar masses
    """

    return (-log10_fgw - np.log10(6**(1.5)*np.pi)
            + 0.6*np.log10(q/(1+q)**2) - np.log10(const.Tsun))


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

    Tmaxyr = np.array([(max(p.toas()) - min(p.toas()))/3.16e7
                       for p in libs_psrs]).max()

    # draw parameter values
    gwtheta = np.arccos(np.random.uniform(-1, 1))
    gwphi = np.random.uniform(0, 2*np.pi)
    phase0 = np.random.uniform(0, 2*np.pi)
    inc = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, np.pi)
    
    mc_max = min(compute_max_chirpmass(np.log10(fgw)),10)
    mc = 10**np.random.uniform(6, mc_max)

    dist = 4 * np.sqrt(2/5) * (mc*4.9e-6)**(5/3) * (np.pi*fgw)**(2/3) / h
    dist /= 1.0267e14   # covert distance to Mpc

    for lp in libs_psrs:
        LT.add_cgw(lp, gwtheta, gwphi, mc, dist, fgw,
                   phase0, psi, inc, pdist=1.0,
                   pphase=None, psrTerm=True, evolve=False, 
                   phase_approx=True, tref=0)
    
    for lp in libs_psrs:
        lp.fit(iters=2)
        
    # convert to enterprise pulsar objects
    psrs = []
    for lp in libs_psrs:

        psr = Tempo2Pulsar(lp)

        # remove any bad toas where the residual is huge
        res_limit = (np.mean(np.abs(psr.residuals))
                     + 5.*np.std(np.abs(psr.residuals)))
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
    detect = 0
    t0 = time.time()
    
    for _ in range(nreal):

        try:
            psrs = make_sim(datadir, fgw, h, endtime=endtime, psrlist=psrlist)
        except:
            psrs = []
        
        if len(psrs) > 0:
            
            setpars = {}
            for psr in psrs:
                setpars.update({'{0}_efac'.format(psr.name): 1.0})

            pta = initialize_pta_sim(psrs, fgw)
            fp = F_statistic.FpStat(psrs, params=setpars, pta=pta)
            fap0 = fp.compute_fap(fgw)

            count += 1

            if fap0 < fap:
                detect += 1

    if count == 0:
        logger.error('No simulated data sets were generated!')
        return np.inf
    else:
        msg = 'Computed {0} realizations in {1} s'.format(count, time.time()-t0)
        logger.info(msg)
        return detect/count
