#!/usr/bin/env python

from __future__ import division
import numpy as np
import glob
import os
import sys
import logging

import cw_sims

logging.getLogger().setLevel(logging.ERROR)


def load_outfile(outfile, hmin, hmax):

    # TO DO: make sure that I can handle all possible shapes of data
    # what if there is just one line of data?

    data = np.loadtxt(outfile)
    
    if len(data.shape) == 1:
        data = np.array([data])
    
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

    parser.add_argument('--freq', help='GW frequency for search (DEFAULT: 1e-8)', default=1e-8)
    parser.add_argument('--hmin', help='Minimum GW strain (DEFAULT: 1e-17)', default=1e-17)
    parser.add_argument('--hmax', help='Maximum GW strain (DEFAULT: 1e-12)', default=1e-12)
    parser.add_argument('--htol', help='Fractional error in GW strain (DEFAULT: 0.1)', default=0.1)
    parser.add_argument('--nreal', help='Number of realizations', default=100)
    parser.add_argument('--det_prob', help='Detection probability (DEFAULT: 0.95)', default=0.95)
    parser.add_argument('--fap', help='False alarm probability (DEFAULT: 1e-4)', default=1e-4)
    parser.add_argument('--datadir', help='Directory of the par and tim files',
                        default='../data/partim/')
    parser.add_argument('--endtime', help='Observation end date [MJD]',
                        default=None)
    parser.add_argument('--psrlist', help='List of pulsars to use',
                        default=None)
    parser.add_argument('--outdir', help='Directory to put the detection curve files', 
                        default='det_curve/')

    args = parser.parse_args()
    
    fgw = float(args.freq)
    nreal = int(args.nreal)
    if args.endtime is None:
        endtime = args.endtime
    else:
        endtime = float(args.endtime)
    psrlist = args.psrlist
    fap = float(args.fap)
    det_prob = float(args.det_prob)
    datadir = args.datadir

    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except OSError:
            pass

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
            fa = cw_sims.compute_det_prob(fgw, a, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(a, fa))
            f.flush()

        if fc is None:
            fc = cw_sims.compute_det_prob(fgw, c, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(c, fc))
            f.flush()

        # initially perform a bisection search
        while b is None:
    
            x = 10**((np.log10(a) + np.log10(c))/2)
    
            fx = cw_sims.compute_det_prob(fgw, x, nreal, fap,
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
        x = cw_sims.compute_x(a, fa, b, fb, c, fc)

        while np.abs(x-b) > float(args.htol)*b:
    
            fx = cw_sims.compute_det_prob(fgw, x, nreal, fap,
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
        
            x = cw_sims.compute_x(a, fa, b, fb, c, fc)
            
        fx = cw_sims.compute_det_prob(fgw, x, nreal, fap,
                                      datadir, endtime=endtime, psrlist=psrlist) - det_prob
        f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))
        f.flush()
