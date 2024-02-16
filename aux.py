from __future__ import print_function, division
import numpy as np
import sys, time
from warnings import warn
import contextlib

class Stopwatch:
    def __init__(self):
        self.previous = time.time()
        self.initial = self.previous

    def click(self):
        elapsed = time.time() - self.previous
        self.previous += elapsed
        return elapsed

    def total(self):
        return time.time() - self.initial

@contextlib.contextmanager
def np_printoptions(*args, **kwds):
    """Context manager for numpy print options

    Compatibility code for numpy version < 1.15
    """
    opts = np.get_printoptions()
    np.set_printoptions(*args, **kwds)
    try:
        yield
    finally:
        np.set_printoptions(**opts)

def parse_tuples(entries, nvals):
    "Parse key/value pairs and turn them into an array and a set of axes"
    ndim = entries.shape[-1] - nvals
    keys = entries[:, :ndim].T
    vals = entries[:, ndim]

    labels = [None] * ndim
    indices = np.empty(keys.shape, int)
    for i in range(ndim):
        labels[i], indices[i] = np.unique(keys[i], return_inverse=True)

    arr = np.zeros(tuple(indices.max(1) + 1))
    arr[tuple(indices)] = vals
    return arr, labels

def mdivide(A, B):
    "Compute `A B^{-1}` or, equivalently, the solution to `X B = A`"
    return np.linalg.solve(B.swapaxes(-2,-1), A.swapaxes(-2,-1)).swapaxes(-2,-1)

def _check_recons(evals, ebasis):
    n = ebasis.shape[-1]
    if ebasis.shape != (n, n):
        raise ValueError("eigenbasis must be a square matrix")
    if evals.shape[-1] != n:
        raise ValueError("eigenvalues last dimension must match eigenbasis")

def recons(evals, ebasis):
    "Reconstruct matrix from eigenvalues and -vectors"
    _check_recons(evals, ebasis)
    return mdivide(ebasis * evals[..., None, :], ebasis)

def reconsh(evals, ebasis):
    "Reconstruct Hermitian matrix from eigenvalues and -vectors"
    _check_recons(evals, ebasis)
    return np.dot(ebasis * evals[..., None, :], ebasis.conj().T)

def decompose(a):
    "Decompose matrix into eigenvalus and eigenbasis"
    a = np.asarray(a)
    is_hermitian = np.allclose(a, a.conj().T)
    if is_hermitian:
        e, b = np.linalg.eigh(a)
    else:
        e, b = np.linalg.eig(a)
        e_perm = e.argsort()
        e = e[e_perm]
        b = b[:, e_perm]
    assert np.allclose(a, recons(e, b))
    return e, b, is_hermitian

def svd_trunc(a, rtol=None, nmax=None):
    "Singular value decomposition with truncation"
    u, s, vH = np.linalg.svd(a, full_matrices=False)
    # remove singular values smaller than rtol
    cut = s.size
    if rtol is not None:
        cut -= s[::-1].searchsorted(float(rtol) * s[0])
    if nmax is not None and cut > nmax:
        cut = nmax
    vprimeH = vH[:cut]
    uprime = u[:, :cut]
    sprime = s[:cut]
    return uprime, sprime, vprimeH

def checkwn(wn, beta, stat='fermi'):
    "Checks if the passed frequencies are Matsubara frequencies"
    wn = np.asarray(wn)
    count = (beta/np.pi * wn + {'fermi':1, 'bose':0}[stat])/2
    if not np.allclose(count, count.round()):
        raise ValueError("Invalid Matsubara frequency")
    return wn

def wraptau(tau, beta, stat='fermi'):
    "Wrap tau around 0, return restricted tau and sign"
    if (np.abs(tau) > beta).any():
        raise ValueError("tau must be in range [-beta,beta]")

    tau = tau.copy()
    tau_neg = tau < 0
    tau[tau_neg] += beta
    sign = np.ones_like(tau)
    if stat == 'fermi':
        sign -= 2 * tau_neg
    return tau, sign

def checkoverlap(overlap, tol=1e-12):
    "Check if the passed matrix is an overlap matrix"
    overlap = np.asarray(overlap)
    if not np.allclose(overlap, overlap.conj().T, tol):
        raise ValueError("Overlap must be a Hermitian matrix")
    if (overlap.diagonal() <= 0).any():
        raise ValueError("Diagonal of overlaps must be strictly positive")
    return overlap

def checkreal(a, tol=1e-12):
    "Checks if quantity is real"
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.complexfloating):
        imag_part = a.imag.max() - a.imag.min()
        if imag_part > tol:
            warn("Imaginary part of magnitude %g" % imag_part, UserWarning, 2)
    return a.real

def checksymm(a, tol=1e-8):
    "Checks if quantity is Hermitian in the last two indices"
    a = np.asarray(a)
    a_dag = a.conj().swapaxes(-2, -1)
    if not np.allclose(a_dag, a, atol=tol, rtol=tol):
        maxdiff = np.abs(a_dag - a).max()
        warn("Quantity is not symmetric. Difference: %g" % maxdiff, UserWarning, 2)
        a = a + a_dag
        a /= 2
    return a

def verifyexp(a, rtol=1e-4, atol=1e-10):
    """Check that the object passed is a basis expansion in zeroth index"""
    x = np.abs(a)
    thr = (rtol * x.max(0)).clip(atol, None)
    if (x[-1] > thr).any():
        warn("Basis expansion does not decay", UserWarning, 2)

def estimate_leakage(signal_iw):
    "Estimates signal leakage into the asymptotic region"
    signal_iw = signal_iw + signal_iw[::-1]
    if np.abs(signal_iw[-2:] < 1e-100).any():
        leakage = signal_iw[-2:].mean()
    else:
        leakage = 1/(1/signal_iw[-1] - 1/signal_iw[-2])
    return checkreal(leakage)
