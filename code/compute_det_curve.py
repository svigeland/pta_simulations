#!/usr/bin/env python

from __future__ import division
import numpy as np
import glob
import os
import sys
import logging

import cw_sims

logger = logging.getLogger(__name__)


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
    if np.any(~np.isfinite(data[:,2])):
        idx = np.where(np.isfinite(data[:,2]))[0]
        data = data[idx]
    
    det_probs = np.unique(data[:,2])

    if len(det_probs) == 1:
        
        # if there is only one unique value of the detection probability,
        # use that value and the corresponding strain amplitude
        # to define one side of the bracket
        # initialize the other side of the bracket to the default value

        if np.unique(data[:,2])[0] < 0:
            
            c, fc = hmax, None
            
            idx = np.where(data[:,2] == det_probs[0])[0]
            if len(idx) > 1:
                data2 = data[idx]
                ii = np.where(data2[:,0] == max(data2[:,0]))[0]
                a, fa = data2[int(ii)]
            else:
                a, fa = data[int(idx)]

        else:
            
            a, fa = hmin, None

            idx = np.where(data[:,2] == det_probs[0])[0]
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

        idx = np.where(data[:,2] < 0)[0]
        if len(idx) > 1:
            data2 = data[idx]
            ii = np.where(data2[:,0] == max(data2[:,0]))[0]
            a, fa = data2[int(ii)]
        else:
            a, fa = data[int(idx)]

        if hmin > a:
            a, fa = hmin, None
    
        idx = np.where(data[:,2] > 0)[0]
        if len(idx) > 1:
            data2 = data[idx]
            ii = np.where(data2[:,0] == min(data2[:,0]))[0]
            c, fc = data2[int(ii)]
        else:
            c, fc = data[int(idx)]
                
        if hmax < c:
            c, fc = hmax, None

    msg = 'Initializing from file'
    msg += ' with a = {0:.2e}, c = {1:.2e}'.format(a, c)
    logger.info(msg)

    # check that a < c, and if not, expand the bounds
    if a > c:
        a /= 2
        c *= 2
        fa, fc = None, None
        logger.warning('There is a problem with the specified bounds!')
        msg = 'Searching over the interval'
        msg += ' [{0:.1e}, {1:.1e}]...'.format(a, c)
        logger.warning(msg)

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


def compute_x(a, fa, b, fb, c, fc, nreal):
    
    # check that all of the values are in the correct order
    if a > c or fa > 0 or fc < 0:
        msg = 'The root is not contained within the interval'
        msg += ' [{0:.2e}, {1:.2e}]!'.format(a, c)
        logger.error(msg)
        x, xerr = None, None
    
    else:
        
        # check that a < b < c, and fa < fb < fc
        # if not, we will not use b to compute the root
        if b is not None:
            if isclose(fb, fa, 1/nreal) or isclose(fb, fc, 1/nreal):
                b, fb = None, None
            elif a > b or b > c or fa > fb or fb > fc:
                b, fb = None, None
        msg = 'Finding new point...'
        msg += ' interval is [{0:.2e}, {1:.2e}]'.format(a, c)
        logger.debug(msg)
    
        # if only the endpoints of the bracket are defined,
        # perform a bisection search
        # otherwise use quadratic interpolation
        if b is None:
            x, xerr = bisection(a, fa, c, fc)
    
            msg = 'Generating new point using bisection method...'
            msg += ' x = {0:.2e}, xerr = {1:.2e}'.format(x, xerr)
            logger.debug(msg)
        else:
            x, xerr = inv_quad_interp(a, fa, b, fb, c, fc)

            # if inverse quadratic interpolation generates a root
            # outside of the bracket, use linear interpolation instead
            if x < a or x > c:
                if np.sign(fb) == np.sign(fc):
                    x, xerr = linear_interp(a, fa, b, fb)
                else:
                    x, xerr = linear_interp(b, fb, c, fc)
                msg = 'Generating new point using linear interpolation...'
                msg += ' x = {0:.2e}, xerr = {1:.2e}'.format(x, xerr)
                logger.debug(msg)
            else:
                msg = 'Generating new point using'
                msg += ' inverse quadratic interpolation...'
                msg += ' x = {0:.2e}, xerr = {1:.2e}'.format(x, xerr)
                logger.debug(msg)

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
    
    parser = argparse.ArgumentParser(description='Continuous Gravitational Wave Simulations')

    parser.add_argument('--freq', default=1e-8,
                        help='GW frequency for search (DEFAULT: 1e-8)')
    parser.add_argument('--hmin', default=1e-17,
                        help='Minimum GW strain (DEFAULT: 1e-17)')
    parser.add_argument('--hmax', default=1e-12,
                        help='Maximum GW strain (DEFAULT: 1e-12)')
    parser.add_argument('--htol', default=0.1,
                        help='Fractional error in GW strain (DEFAULT: 0.1)')
    parser.add_argument('--nreal', default=100,
                        help='Number of realizations')
    parser.add_argument('--det_prob', default=0.95,
                        help='Detection probability (DEFAULT: 0.95)')
    parser.add_argument('--fap', default=1e-4,
                        help='False alarm probability (DEFAULT: 1e-4)')
    parser.add_argument('--datadir', default='../data/partim/',
                        help='Directory of the par and tim files')
    parser.add_argument('--endtime', default=None,
                        help='Observation end date [MJD]')
    parser.add_argument('--psrlist', default=None,
                        help='List of pulsars to use')
    parser.add_argument('--outdir', default='det_curve/',
                        help='Directory to put the detection curve files')
    parser.add_argument('--max_iter', default=10,
                        help='Maximum number of iterations to perform')
    parser.add_argument('--recalculate', action='store_true', default=False,
                        help='Recalculate the detection probabilities when loading from file')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Sets logger level to DEBUG')

    args = parser.parse_args()
    
    # the logging level for __main__ can be either INFO or DEBUG,
    # depending on the arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # set the logging level of enterprise.pulsar to ERROR
    logging.getLogger('enterprise.pulsar').setLevel(logging.ERROR)

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

    # if the outfile exists and is not empty, use the results
    # from a previous run to define the bounds of the search
    # otherwise search over the entire range
    if os.path.isfile(outfile) and os.stat(outfile).st_size > 1:
        logger.info('Resuming from a previous calculation...')
        a, fa, b, fb, c, fc = load_outfile(outfile, float(args.hmin),
                                           float(args.hmax), args.recalculate)
    else:
        a, c = float(args.hmin), float(args.hmax)
        fa, fc = None, None
        b, fb = None, None
        msg = 'Searching over the interval'
        msg += ' [{0:.1e}, {1:.1e}]...'.format(a, c)
        logger.info(msg)

    iter = 0
    
    if fa is None:
        fa = cw_sims.compute_det_prob(fgw, a, nreal, fap, datadir,
                                      endtime=endtime,
                                      psrlist=psrlist) - det_prob
        iter += 1
        
        # if fa > 0, try a smaller value for a
        while fa > 0 and iter < max_iter:
            logger.warning('Adjusting lower bound of interval...')
            a /= 2
            fa = cw_sims.compute_det_prob(fgw, a, nreal, fap, datadir,
                                          endtime=endtime,
                                          psrlist=psrlist) - det_prob
            iter += 1

        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:.2e}  {2:>6.3f}\n'.format(a, a, fa))

    if fc is None:
        fc = cw_sims.compute_det_prob(fgw, c, nreal, fap, datadir,
                                      endtime=endtime,
                                      psrlist=psrlist) - det_prob
        iter += 1

        # if fc < 0, try a larger value for c
        while fc < 0 and iter < max_iter:
            logger.warning('Adjusting upper bound of interval...')
            c *= 2
            fc = cw_sims.compute_det_prob(fgw, c, nreal, fap, datadir,
                                          endtime=endtime,
                                          psrlist=psrlist) - det_prob
            iter += 1

        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:.2e}  {1:>6.3f}\n'.format(c, c, fc))

    x, xerr = compute_x(a, fa, b, fb, c, fc, nreal)
        
    while x is not None and xerr/x > htol and iter < max_iter:
        
        fx = cw_sims.compute_det_prob(fgw, x, nreal, fap, datadir,
                                      endtime=endtime,
                                      psrlist=psrlist) - det_prob
        iter += 1

        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:.2e}  {2:>6.3f}\n'.format(x, xerr, fx))

        # redefine the points a, b, c to incorporate x
        a, fa, b, fb, c, fc = set_new_bounds(a, fa, b, fb, c, fc,
                                             x, fx, nreal)

        # check that f is monotonically increasing between a, b, and c
        # if not, adjust the endpoints a and c
        if fb is not None:
            
            while fa > fb and iter < max_iter:
                logger.warning('Adjusting lower bound of interval...')
                a /= 2
                fa = cw_sims.compute_det_prob(fgw, a, nreal, fap, datadir,
                                              endtime=endtime,
                                              psrlist=psrlist) - det_prob
                iter += 1
                        
                with open(outfile, 'a') as f:
                    f.write('{0:.2e}  {1:.2e}  {2:>6.3f}\n'.format(a, a, fa))

            while fc < fb and iter < max_iter:
                logger.warning('Adjusting upper bound of interval...')
                c *= 2
                fc = cw_sims.compute_det_prob(fgw, c, nreal, fap, datadir,
                                              endtime=endtime,
                                              psrlist=psrlist) - det_prob
                iter += 1
                        
                with open(outfile, 'a') as f:
                    f.write('{0:.2e}  {1:.2e}  {2:>6.3f}\n'.format(c, c, fc))

        x, xerr = compute_x(a, fa, b, fb, c, fc, nreal)

    if x is None:
        logger.error('I could not find the root!')
    else:
        fx = cw_sims.compute_det_prob(fgw, x, nreal, fap, datadir,
                                      endtime=endtime,
                                      psrlist=psrlist) - det_prob
        iter += 1

        with open(outfile, 'a') as f:
            f.write('{0:.2e}  {1:.2e}  {2:>6.3f}\n'.format(x, xerr, fx))

        logger.info('Search complete.')
        logger.info('{0} iterations were performed.'.format(iter))
        msg = 'Best estimate for the root:'
        msg += ' {0:.2e} +/- {1:.2e}'.format(x, xerr)
        logger.info(msg)
