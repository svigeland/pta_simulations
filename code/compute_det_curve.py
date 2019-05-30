#!/usr/bin/env python

from __future__ import division
import numpy as np
import glob
import os
import sys
import logging

import cw_sims

logging.getLogger().setLevel(logging.ERROR)


def load_outfile(outfile, hmin, hmax, recalculate=False):
    """
    Utility function that loads in the results from a previous calculation.
    Previous strain amplitudes and detection probabilities are loaded
    and used to define the minimum and maximum strain amplitudes for the search
    
    :param outfile: Name of output file
    :param hmin: Minimum value of strain amplitude
    :param hmax: Maximum value of strain amplitude
    """

    data = np.loadtxt(outfile)
    
    if len(data.shape) == 1:
        data = np.array([data])
    
    # remove any points where the detection probability is not finite
    if np.any(~np.isfinite(data[:,1])):
        idx = np.where(np.isfinite(data[:,1]))[0]
        data = data[idx]
    
    det_probs = np.unique(data[:,1])

    if len(det_probs) == 1:
        
        # if there is only one unique value of the detection probability,
        # use that value and the corresponding strain amplitude
        # to define one side of the bracket
        # initialize the other side of the bracket to the default value

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
        
        # if there is more than one unique value of the detection probability,
        # find the values of the detection probability that are closest to zero
        # and use the corresponding strain amplitudes to define the bracket

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

    # check that a < c, and if not, expand the bounds
    if a > c:
        a /= 2
        c *= 2
        fa, fc = None, None
        print('There is a problem with the specified bounds!')
        print('Searching over the interval [{0:.1e}, {1:.1e}]...'.format(a, c))
    
    if recalculate:
        fa, fc = None, None

    return a, fa, None, None, c, fc


def isclose(f1, f2, tol):

    if np.abs(f1-f2) < tol and np.sign(f1) == np.sign(f2):
        return True
    else:
        return False


def linear_interp(a, fa, c, fc):
    
    return 10**(np.log10(a) - (np.log10(c)-np.log10(a))*fa/(fc-fa))


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Gravitational Wave Simulations via Enterprise')

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
    parser.add_argument('--max_iter', help='Maximum number of iterations to perform', default=10)
    parser.add_argument('--recalculate', action='store_true', default=False,
                        help='When loading from a file, should I recalculate the detection probabilities?')

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

    max_iter = int(args.max_iter)

    datadir = args.datadir
    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except OSError:
            pass

    outfile = '{0}/{1}.txt'.format(args.outdir, args.freq)

    # if the outfile exists and is not empty, use the results from a previous run
    # to define the bounds of the search
    # otherwise search over the entire range
    if os.path.isfile(outfile) and os.stat(outfile).st_size > 1:
        print('Resuming from a previous calculation...')
        sys.stdout.flush()
        a, fa, b, fb, c, fc = load_outfile(outfile, float(args.hmin), float(args.hmax),
                                           args.recalculate)
    else:
        a, c = float(args.hmin), float(args.hmax)
        fa, fc = None, None
        b, fb = None, None
        print('Searching over the interval [{0:.1e}, {1:.1e}]...'.format(a, c))
        sys.stdout.flush()

    iter = 0
    
    with open(outfile, 'a') as f:
    
        if fa is None:
            fa = cw_sims.compute_det_prob(fgw, a, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(a, fa))
            f.flush()
            iter += 1
        
        # if fa > 0, try a smaller value for a
        while fa > 0:
            a /= 2
            fa = cw_sims.compute_det_prob(fgw, a, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(a, fa))
            f.flush()
            iter += 1

        if fc is None:
            fc = cw_sims.compute_det_prob(fgw, c, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(c, fc))
            f.flush()
            iter += 1

        # if fc < 0, try a larger value for c
        while fc < 0:
            c *= 2
            fc = cw_sims.compute_det_prob(fgw, c, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(c, fc))
            f.flush()
            iter += 1

        # initially perform a bisection search
        while b is None and (c-a) > float(args.htol)*a and iter < max_iter:
            
            print('Performing bisection search...')
            sys.stdout.flush()
    
            x = 10**((np.log10(a) + np.log10(c))/2)
            fx = cw_sims.compute_det_prob(fgw, x, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
        
            f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))
            f.flush()
            iter += 1
            
            if c/a > 10 or isclose(fx, fa, 1/nreal) or isclose(fx, fc, 1/nreal):
                if np.sign(fa) == np.sign(fx):
                    a, fa = x, fx
                else:
                    c, fc = x, fx
            else:
                b, fb = x, fx

        # now use Brent's method
        if b is not None and iter < max_iter:

            print('Switching to Brent\'s method...')
            print('Values are a = {0:.2e}, b = {1:.2e}, c = {2:.2e}'.format(a, b, c))
            sys.stdout.flush()
        
            # use Brent's root finding algorithm to estimate the value of the root
            x = cw_sims.compute_x(a, fa, b, fb, c, fc)

            # if x is not bracketed by a and c, use linear interpolation to generate the new point
            if x < a or x > c:
                if np.sign(fb) == np.sign(fc):
                    x = linear_interp(a, fa, b, fb)
                else:
                    x = linear_interp(b, fb, c, fc)
            
            while np.abs(x-b) > float(args.htol)*b and iter < max_iter:
    
                fx = cw_sims.compute_det_prob(fgw, x, nreal, fap,
                                              datadir, endtime=endtime, psrlist=psrlist) - det_prob
                f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))
                f.flush()
                iter += 1
        
                # if fx is very close to fa, fb, or fc, replace that point with the new point
                if isclose(fx, fa, 1/nreal):
                    a, fa = x, fx
                elif isclose(fx, fb, 1/nreal):
                    b, fb = x, fx
                elif isclose(fx, fc, 1/nreal):
                    c, fc = x, fx
                else:
                    # otherwise reorder a, b, and c
                    # this section relies on the fact that the detection probability
                    # should be a monotonically increasing function, but may not be
                    # due to the finite number of realizations
                    if np.sign(fb) == np.sign(fa):
                        if np.sign(fx) == np.sign(fc):
                            c, fc = x, fx
                        else:
                            if fx > fb:
                                a, fa = b, fb
                                b, fb = x, fx
                            elif fx > fa:
                                b, fb = x, fx
                    else:
                        if np.sign(fx) == np.sign(fa):
                            a, fa = x, fx
                        else:
                            if fx < fb:
                                c, fc = b, fb
                                b, fb = x, fx
                            elif fx < fc:
                                b, fb = x, fx
        
                x = cw_sims.compute_x(a, fa, b, fb, c, fc)
                if x < a or x > c:
                    if np.sign(fb) == np.sign(fc):
                        x = linear_interp(a, fa, b, fb)
                    else:
                        x = linear_interp(b, fb, c, fc)

            fx = cw_sims.compute_det_prob(fgw, x, nreal, fap,
                                          datadir, endtime=endtime, psrlist=psrlist) - det_prob
            f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))
            f.flush()

        print('Search complete.')
        print('{0} iterations were performed.'.format(iter))
