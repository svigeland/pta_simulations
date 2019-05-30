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

        if hmin > a:
            a, fa = hmin, None
    
        idx = np.where(data[:,1] > 0)[0]
        if len(idx) > 1:
            data2 = data[idx]
            ii = np.where(data2[:,0] == min(data2[:,0]))[0]
            c, fc = data2[int(ii)]
        else:
            c, fc = data[int(idx)]
                
        if hmax < c:
            c, fc = hmax, None
            
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


def bisection(a, fa, c, fc):

    return 10**((np.log10(a) + np.log10(c))/2), (c-a)/2


def linear_interp(a, fa, c, fc):
    
    return 10**(np.log10(a) - (np.log10(c)-np.log10(a))*fa/(fc-fa)), (c-a)/2


def inv_quad_interp(a, fa, b, fb, c, fc):
    
    R = fb/fc
    S = fb/fa
    T = fa/fc
    
    P = S*(T*(R-T)*(c-b) - (1.-R)*(b-a))
    Q = (T-1.)*(R-1.)*(S-1.)
    
    return b + P/Q, np.abs(P/Q)


def compute_x(a, fa, b, fb, c, fc, verbose=False):
    
    # check that all of the values are in the correct order
    if a > c or fa > 0 or fc < 0:
        x, xerr = None, None
    
    else:
    
        # check that a < b < c, and fa < fb < fc
        # if not, we will not use b to compute the root
        if b is not None:
            if a > b or b > c or fa > fb or fb > fc:
                b, fb = None, None

        if verbose:
            print('Finding new point... interval is [{0:.2e}, {1:.2e}]'.format(a, c))
            if b is not None:
                print('Midpoint value is {0.2e}'.format(b))
            sys.stdout.flush()
    
        # if only the endpoints of the bracket are defined, perform a bisection search
        # otherwise use quadratic interpolation
        if b is None:
            x, xerr = bisection(a, fa, c, fc)
    
            if verbose:
                print('Generating new point using bisection method...')
                print('x = {0:.2e}, xerr = {1:.2e}'.format(x, xerr))
                sys.stdout.flush()
        else:
            x, xerr = inv_quad_interp(a, fa, b, fb, c, fc)

            # if inverse quadratic interpolation generates a root outside of the bracket,
            # use linear interpolation instead
            if x < a or x > c:
                if np.sign(fb) == np.sign(fc):
                    x, xerr = linear_interp(a, fa, b, fb)
                else:
                    x, xerr = linear_interp(b, fb, c, fc)
                if verbose:
                    print('Generating new point using linear interpolation...')
                    print('x = {0:.2e}, xerr = {1:.2e}'.format(x, xerr))
                    sys.stdout.flush()
            else:
                if verbose:
                    print('Generating new point using inverse quadratic interpolation...')
                    print('x = {0:.2e}, xerr = {1:.2e}'.format(x, xerr))
                    sys.stdout.flush()

    return x, xerr


def set_new_bounds(a, fa, b, fb, c, fc, x, fx, nreal):

    if isclose(fx, fa, 1/nreal):
        a, fa = x, fx
    elif isclose(fx, fc, 1/nreal):
        c, fc = x, fx
    elif b is None:
        if xerr/x > 2:
            if np.sign(fa) == np.sign(fx):
                a, fa = x, fx
            else:
                c, fc = x, fx
        else:
            b, fb = x, fx
    else:
        if isclose(fx, fb, 1/nreal):
            b, fb = x, fx
        else:
            if np.sign(fb) == np.sign(fa):
                if np.sign(fx) == np.sign(fc):
                    c, fc = x, fx
                else:
                    a, fa = b, fb
                    b, fb = x, fx
            else:
                if np.sign(fx) == np.sign(fa):
                    a, fa = x, fx
                else:
                    c, fc = b, fb
                    b, fb = x, fx

    return a, fa, b, fb, c, fc


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
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print extra messages (helpful for debugging purposes')

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
    htol = float(args.htol)

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
    
    if fa is None:
        fa = cw_sims.compute_det_prob(fgw, a, nreal, fap, datadir,
                                      endtime=endtime, psrlist=psrlist) - det_prob
        iter += 1
        
        # if fa > 0, try a smaller value for a
        while fa > 0 and iter < max_iter:
            a /= 2
            fa = cw_sims.compute_det_prob(fgw, a, nreal, fap, datadir,
                                          endtime=endtime, psrlist=psrlist) - det_prob
            iter += 1

        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:>6.3f}\n'.format(a, fa))

    if fc is None:
        fc = cw_sims.compute_det_prob(fgw, c, nreal, fap, datadir,
                                      endtime=endtime, psrlist=psrlist) - det_prob
        iter += 1

        # if fc < 0, try a larger value for c
        while fc < 0 and iter < max_iter:
            c *= 2
            fc = cw_sims.compute_det_prob(fgw, c, nreal, fap, datadir,
                                          endtime=endtime, psrlist=psrlist) - det_prob
            iter += 1

        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:>6.3f}\n'.format(c, fc))

    x, xerr = compute_x(a, fa, b, fb, c, fc,
                        verbose=args.verbose)
        
    while x is not None and xerr/x > htol and iter < max_iter:
        
        fx = cw_sims.compute_det_prob(fgw, x, nreal, fap, datadir,
                                      endtime=endtime, psrlist=psrlist) - det_prob
        iter += 1

        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))

        # redefine the points a, b, c to incorporate x
        a, fa, b, fb, c, fc = set_new_bounds(a, fa, b, fb, c, fc,
                                             x, fx, nreal)

        # check that f is monotonically increasing between a, b, and c
        # if not, adjust the endpoints a and c
        if fb is not None:
            
            while fa > fb and iter < max_iter:

                if verbose:
                    print('Adjusting lower bound of interval...')
                    sys.stdout.flush()
                        
                a /= 2
                fa = cw_sims.compute_det_prob(fgw, a, nreal, fap, datadir,
                                              endtime=endtime, psrlist=psrlist) - det_prob
                iter += 1
                        
                with open(outfile, 'a') as f:
                    f.write('{0:.2e}  {1:>6.3f}\n'.format(a, fa))

            while fc < fb and iter < max_iter:
                        
                if verbose:
                    print('Adjusting upper bound of interval...')
                    sys.stdout.flush()

                c *= 2
                fc = cw_sims.compute_det_prob(fgw, c, nreal, fap, datadir,
                                              endtime=endtime, psrlist=psrlist) - det_prob
                iter += 1
                        
                with open(outfile, 'a') as f:
                    f.write('{0:.2e}  {1:>6.3f}\n'.format(c, fc))

        x, xerr = compute_x(a, fa, b, fb, c, fc,
                            verbose=args.verbose)

    if x is None:
        print('I could not find the root!')
    else:
        fx = cw_sims.compute_det_prob(fgw, x, nreal, fap, datadir,
                                      endtime=endtime, psrlist=psrlist) - det_prob
        
        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:>6.3f}\n'.format(x, fx))

        print('Search complete.')
        print('{0} iterations were performed.'.format(iter))
        print('Best estimate for the root: {0:.2e} +/- {1:.2e}'.format(x, xerr))
