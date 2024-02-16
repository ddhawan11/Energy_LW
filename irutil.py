from __future__ import print_function, division
from builtins import range

import numpy as np
import irbasis
import aux

def funique(x, tol=2e-16):
    """Removes duplicates from an 1D array within tolerance"""
    x = np.sort(x)
    unique = np.ediff1d(x, to_end=2*tol) > tol
    x = x[unique]
    return x

def find_roots(ulx_data, xoffset, tol=2e-16):
    """Find all roots in the piecewise polynomial representation"""
    nsec, npoly = ulx_data.shape
    if xoffset.shape != (nsec+1,):
        raise ValueError("Invalid section edges shape")

    xsegm = xoffset[1:] - xoffset[:-1]
    roots = []
    for i in range(nsec):
        x0s = np.roots(ulx_data[i, ::-1])
        x0s = [(x0 + xoffset[i]).real for x0 in x0s
               if -tol < x0 < xsegm[i]+tol and np.abs(x0.imag) < tol]
        roots += x0s

    roots = np.asarray(roots)
    roots = np.hstack((-roots[::-1], roots))
    roots = funique(roots, tol)
    return roots

def get_x_sampling(roots):
    """Get sampling points for x, given roots"""
    if roots.min() < -1 or roots.max() > 1:
        raise ValueError("domain of x")

    aug_roots = np.hstack((-1., roots, 1.))
    aug_roots.sort()
    x_sampling = .5 * (aug_roots[:-1] + aug_roots[1:])
    return x_sampling

def ftau_fitl(ftau, Ultau):
    """Fit scalar function"""
    ntau, norb = ftau.shape[:2]
    nl = Ultau.shape[0]
    if ftau.shape != (ntau, norb, norb):
        raise ValueError("ftau must be tensor")
    if Ultau.shape != (nl, ntau):
        raise ValueError("Ultau does not match")

    ftau = ftau.reshape(ntau, norb * norb).T
    fl = np.zeros((norb * norb, nl))
    res = 0
    for i in range(norb * norb):
        ftau_i = ftau[i]
        fl_i, res_i, _, _ = np.linalg.lstsq(Ultau.T, ftau_i, rcond=None)
        fl[i] = fl_i
        res += res_i.sum()

    fl = fl.T.reshape(nl, norb, norb)
    return fl, res

def fit_g0(f, beta, basis, nl=None):
    """Fit function of the form 1/(iw  - f[:])"""
    if nl is None:
        nl = basis.dim()
    fflat = np.ravel(f)
    wmax = basis._Lambda / beta
    y = fflat/wmax

    # compute the expansion coefficent of u_l(x) for a set of delta peaks in y
    vly = np.asarray([basis.vly(l, yi) for l in range(nl) for yi in y])
    vly = vly.reshape(nl, fflat.size)
    sl = np.array([basis.sl(l) for l in range(nl)])
    g0l = -np.sqrt(beta/2) * sl[:,None] * vly

    return g0l.reshape((nl,) + np.shape(f))

def dyson(Tlpq, fl, G0l, Sigmal):
    """Computes Gl by using the Dyson equation"""
    nl, norb = G0l.shape[:2]
    if Tlpq.shape != (nl, nl, nl):
        raise ValueError("Illegal size of Tql")
    if fl.shape != (nl,):
        raise ValueError("Illegal size of fl")
    if G0l.shape != (nl, norb, norb):
        raise ValueError("Illegal size of G0l")
    if Sigmal.shape != G0l.shape:
        raise ValueError("Illegal size of Sigmal")

    # Compute lhs, (1 - G0 * Sigma), regularized by f
    G0Sigma_contr = np.einsum('Pab,Qbc->PQac', G0l, Sigmal)
    G0Sigmal = np.einsum('SPQ,PQac->Sac', Tlpq, G0Sigma_contr)
    fG0Sigmal = np.einsum('LRS,R,Sac->Lac', Tlpq, fl, G0Sigmal)
    Al = fl[:,None,None] * np.eye(norb) - fG0Sigmal

    # Regularize rhs, G0 by f
    G0primel = np.einsum('LRS,R,Sac->Lac', Tlpq, fl, G0l)

    # Multiply A by convolution operator and group indices
    Aprimel = np.einsum('LPQ,Pab->LQab', Tlpq, Al)
    AprimePQ = Aprimel.transpose(0, 2, 1, 3).reshape((norb * nl,) * 2)
    G0primePR = G0primel.reshape(norb * nl, norb)

    # Linear solve - O(norb**3 * nl**3)
    GPR = np.linalg.solve(AprimePQ, G0primePR)

    # Ungroup indices
    Gl = GPR.reshape(nl, norb, norb)
    return Gl

def _start_guesses(n=1000):
    "Construct points on a logarithmically extended linear interval"
    x1 = np.arange(n)
    x2 = np.array(np.exp(np.linspace(np.log(n), np.log(1E+8), n)), dtype=int)
    x = np.unique(np.hstack((x1,x2)))
    return x

def _get_unl_real(basis, x):
    "Return highest-order basis function on the Matsubara axis"
    unl = basis.compute_unl(x)
    result = np.zeros(unl.shape, float)

    # Purely real functions
    real_loc = 1 if basis.statistics == 'F' else 0
    assert np.allclose(unl[:, real_loc::2].imag, 0)
    result[:, real_loc::2] = unl[:, real_loc::2].real

    # Purely imaginary functions
    imag_loc = 1 - real_loc
    assert np.allclose(unl[:, imag_loc::2].real, 0)
    result[:, imag_loc::2] = unl[:, imag_loc::2].imag
    return result

def _sampling_points(fn):
    "Given a discretized 1D function, return the location of the extrema"
    fn = np.asarray(fn)
    fn_abs = np.abs(fn)
    sign_flip = fn[1:] * fn[:-1] < 0
    sign_flip_bounds = np.hstack((0, sign_flip.nonzero()[0] + 1, fn.size))
    points = []
    for segment in map(slice, sign_flip_bounds[:-1], sign_flip_bounds[1:]):
        points.append(fn_abs[segment].argmax() + segment.start)
    return np.asarray(points)

def _full_interval(sample, stat):
    if stat == 'F':
        return np.hstack((-sample[::-1]-1, sample))
    else:
        # If we have a bosonic basis and even order (odd maximum), we have a
        # root at zero. We have to artifically add that zero back, otherwise
        # the condition number will blow up.
        if sample[0] == 0:
            sample = sample[1:]
        return np.hstack((-sample[::-1], 0, sample))

def get_mats_sampling(basis, lmax=None):
    "Generate Matsubara sampling points from extrema of basis functions"
    if lmax is None: lmax = basis.dim()-1

    x = _start_guesses()
    y = _get_unl_real(basis, x)[:,lmax]
    x_idx = _sampling_points(y)

    sample = x[x_idx]
    return _full_interval(sample, basis.statistics)
