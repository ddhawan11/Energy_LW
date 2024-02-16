import numpy as np
import mpmath as mp

import aux

def get_xroot(order):
    "Roots for order'th polynomial"
    return np.cos(np.pi/order * (np.arange(order) + .5))

def get_fittau_matrix(order):
    "Fitting matrix from roots to Chebyshev coefficients"
    n = np.arange(order)[:,None]
    k = np.arange(order)[None,:]
    coeff = 2./order * np.cos(np.pi/order * n * (k + 0.5))
    coeff[0,:] /= 2
    return coeff

def get_evaltau_matrix(order):
    "Evaluation matrix from Chebyshev coefficients to roots"
    x = get_xroot(order)
    return np.polynomial.chebyshev.chebvander(x, order-1)

def gen_Imz(mmax, stat='fermi'):
    """Generates transformation polynomials from Chebyshev to Matsubara"""
    Ioddsum = np.zeros(mmax + 1, object)
    Ievensum = np.zeros(mmax + 1, object)
    Icurr = np.zeros(mmax + 1, object)

    if stat == 'fermi':
        # Initialize the sum to half of the first term allows a compact
        # statement of the recursion.
        Icurr[1] = 2
        Ievensum[1] = 1
        sign = -1
    elif stat == 'bose':
        sign = +1
    else:
        raise ValueError("statistics must be either fermi or bose, but is %s" % stat)

    # First term
    yield Icurr.copy()
    for m in range(1, mmax):
        if m % 2:
            # Odd term
            Icurr[0] = 0
            Icurr[1:] = 2 * m * Ievensum[:-1]
            Icurr[1] += -1 - sign
            Ioddsum += Icurr
        else:
            # Even term
            Icurr[0] = 0
            Icurr[1:] = 2 * m * Ioddsum[:-1]
            Icurr[1] += 1 - sign
            Ievensum += Icurr
        yield Icurr.copy()

def cheb_integrate(n):
    """Evaluate the integral over [-1,1] of the n-th Chebyshev"""
    def _general(n):
        return ((-1.)**n + 1)/(1 - n**2)

    n = np.asarray(n)
    result = np.zeros(n.shape, float)
    n_is_one = n == 1
    result[n_is_one] = 0
    result[~n_is_one] = _general(n[~n_is_one])
    return result

def eval_Imz(poly, stat, n, prec):
    """Evaluate transformation polys for desired frequency"""
    if stat == 'fermi':
        with mp.workdps(prec):
            z = 1/(-1j * (n + 0.5) * mp.pi)
            return complex(mp.polyval(tuple(poly[::-1]), z))
    elif stat == 'bose':
        # The case n == 0 has to be handled specially, as the polynomial has
        # a pole there.
        if n == 0:
            order = poly.nonzero()[0].max(initial=0)
            return cheb_integrate(order)

        with mp.workdps(prec):
            z = 1/(-1j * n * mp.pi)
            return complex(mp.polyval(tuple(poly[::-1]), z))
    else:
        raise ValueError("stat must be either fermi or bose")

def _sign_changes(fn):
    "Given a discretized 1D function, return the location of the extrema"
    fn = np.asarray(fn)
    sign_flip = fn[1:] * fn[:-1] < 0
    return sign_flip.nonzero()[0]

def iwroots(poly, stat, prec):
    """Get points close to the roots of Chebyshev on the Matsubara axis"""
    if stat not in ('fermi', 'bose'):
        raise ValueError("Illegal value of stat: %s" % stat)
    l = poly.nonzero()[0].max(initial=0)
    nw_bound = (l//2)**2     # Empirical bound
    fn = np.asarray([eval_Imz(poly, stat, n, prec) for n in range(nw_bound+1)])

    # Even functions are purely real, odd functions are purely imaginary
    if l % 2:
        fn = fn.imag
    else:
        fn = aux.checkreal(fn)
    n0 = _sign_changes(fn)

    # The first roots are close to, but just before the first Matsubaras, so
    # they are invisible at the Matsubara points
    nfirst = np.arange(n0.min())
    n0 = np.hstack((nfirst, n0))

    if stat == 'fermi':
        n0 = np.hstack((-n0[::-1]-1, n0))
        #assert n0.size == l
    else:
        n0 = n0[1:]   # remove the zero
        n0 = np.hstack((-n0[::-1], 0, n0))

    return n0

def get_matstf(order, stat, prec=None):
    """Transform matrix Chebyshev coefficients to Matsubara roots"""
    if stat not in ('fermi', 'bose'):
        raise ValueError("Illegal value of stat: %s" % stat)
    if (order % 2 == 1) != (stat == 'bose'):
        raise ValueError("Bose (Fermi) must have even (odd) points")
    if prec is None:
        prec = 2.5 * order

    poly = list(gen_Imz(order + 1, stat))
    roots = iwroots(poly[order], stat, prec)
    tfmat = np.reshape([eval_Imz(poly_i, stat, n_i, prec)
                        for n_i in roots for poly_i in poly[:-1]],
                       (order, order))
    return roots, tfmat
