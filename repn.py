from __future__ import print_function, division
import os.path
import numpy as np
import scipy.special as sp_spec
import h5py as hdf5

import cheby
import irutil
import aux
import irbasis

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def tau_for_x(x, beta):
    """Rescales tau axis to x -1 ... 1"""
    if x.min() < -1 or x.max() > 1:
        raise ValueError("domain of x")
    return .5 * beta * (x + 1)

def x_for_tau(tau, beta):
    """Rescales xaxis to tau in 0 ... beta"""
    if tau.min() < 0 or tau.max() > beta:
        raise ValueError("domain of tau")
    return 2/beta * (tau - beta/2)


class ChebyshevData:
    @classmethod
    def precompute_stat(cls, order, stat, prec=None):
        xroots = cheby.get_xroot(order)
        axl = cheby.get_evaltau_matrix(order)
        alx = cheby.get_fittau_matrix(order)
        wsample, awl = cheby.get_matstf(order, stat, prec)

        oorder = order + (-1 if stat == 'fermi' else +1)
        oroots = cheby.get_xroot(oorder)
        aoxl = np.polynomial.chebyshev.chebvander(oroots, order-1)

        return cls(order, stat, xroots, wsample, axl, alx, aoxl, awl)

    @classmethod
    def precompute(cls, order, prec=None):
        return (cls.precompute_stat(order, 'fermi'),
                cls.precompute_stat(order-1, 'bose'))

    @classmethod
    def from_archive(cls, ar, stat):
        if stat not in ('fermi', 'bose'):
            raise ValueError("statistics must be either 'fermi' or 'bose'")
        if ar.attrs["repr"] != "Chebyshev":
            raise ValueError("archive does not contain correct representation")

        order = ar.attrs["order"]
        group = ar[stat]
        xsample = group["xsample"][...]
        wsample = group["wsample"][...]
        uxl = group["uxl"][...]
        ulx = group["ulx"][...]
        uxl_other = group["uxl_other"][...]
        uwl = group["uwl"][...]
        return cls(order, stat, xsample, wsample, uxl, ulx, uxl_other, uwl)

    def to_archive(self, ar):
        ar.attrs["repr"] = "Chebyshev"
        ar.attrs["order"] = self.order
        group = ar.require_group(self.stat)
        group["wsample"] = self.wsample
        group["xsample"] = self.xsample
        group["uxl"] = self.axl
        group["ulx"] = self.alx
        group["uxl_other"] = self.aoxl
        group["uwl"] = self.awl

    def __init__(self, order, stat, xsample, wsample, axl, alx, aoxl, awl):
        self.order = order
        self.stat = stat
        self.xsample = xsample
        self.wsample = wsample
        self.axl = axl
        self.alx = alx
        self.aoxl = aoxl
        self.awl = awl


class ChebyshevRepr:
    "Chebyshev representation"
    @classmethod
    def _load_precomputed(cls, order, stat, datadir):
        if datadir is None:
            return None
        fname = os.path.join(datadir, 'cheby', '%d.hdf5' % order)
        try:
            f = hdf5.File(fname, 'r')
        except:
            return None
        else:
            return ChebyshevData.from_archive(f, stat)

    def __init__(self, order, beta, stat='fermi',
                 datadir=None, data=None):
        if data is None:
            data = self._load_precomputed(order, stat, datadir)
        if data is None:
            data = ChebyshevData.precompute_stat(order, stat)
        elif data.order != order or data.stat != stat:
            raise ValueError("Precomputed data does not match")

        # Bosons must have odd order, which is why we choose one order less.
        if stat == 'bose':
            order -= 1

        self.order = order
        self.beta = beta
        self.stat = stat

        # Things for tau axis
        self.tau = tau_for_x(data.xsample, self.beta)
        self.ntau = self.tau.size
        self.altau = data.alx
        self.ataul = data.axl

        self.aotaul = data.aoxl

        # Things for Matsubara axis
        self.wnn = data.wsample
        self.wn = 2.0 * np.pi/beta * (data.wsample + (0.5 if stat == 'fermi' else 0))
        self.nwn = data.wsample.size
        self.awl = beta/2 * data.awl
        self.alw = np.linalg.pinv(self.awl)

    def fit_tau(self, ftau):
        ftau = np.asarray(ftau)
        if ftau.shape[0] != self.ntau:
            raise ValueError("First index must be tau points")
        return np.einsum('ij,j...->i...', self.altau, ftau)

    def eval_tau(self, fl):
        fl = np.asarray(fl)
        aux.verifyexp(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be Chebyshev")
        return np.einsum('ij,j...->i...', self.ataul, fl)

    def eval_other_tau(self, fl):
        fl = np.asarray(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be Chebyshev")
        return np.einsum('ij,j...->i...', self.aotaul, fl)

    def eval_beta(self, fl):
        fl = np.asarray(fl)
        aux.verifyexp(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be Chebyshev")
        return fl.sum(0)

    def fit_iw(self, fwn):
        fwn = np.asarray(fwn)
        if fwn.shape[0] != self.nwn:
            raise ValueError("First index must be Matsubara")
        return np.einsum('ij,j...->i...', self.alw, fwn)

    def eval_iw(self, fl):
        fl = np.asarray(fl)
        aux.verifyexp(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be Chebyshev")
        return np.einsum('ij,j...->i...', self.awl, fl)

    def leakage(self, fl):
        fl = np.asarray(fl)
        return np.abs(fl[-1]).max() / np.abs(fl[0]).max()


class IntermediateData:
    @classmethod
    def _precompute_stat(cls, basis, ncoeff=None):
        """Precompute purely fermionic/bosonic part of a basis"""
        if ncoeff is None or ncoeff > basis.dim():
            ncoeff = basis.dim()

        stattext = {'F':'fermi', 'B':'bose'}[basis.statistics]
        lambda_ = basis.Lambda
        wsample = irutil.get_mats_sampling(basis, ncoeff-1)
        uwl = basis.compute_unl(wsample)[:,:ncoeff]

        xroots = irutil.find_roots(basis._ulx_data[ncoeff-1],
                                   basis._ulx_section_edges, 1e-8)
        xsample = irutil.get_x_sampling(xroots)
        uxl = np.asarray([basis.ulx_all_l(xi) for xi in xsample])[:,:ncoeff]
        ux1l = basis.ulx_all_l(1)[:ncoeff]
        return cls(lambda_, stattext, xsample, wsample, uxl, ux1l, uwl, None)

    @classmethod
    def precompute(cls, lambda_, ncoeff=None):
        """Precompute fermionic and bosonic basis and interface"""
        f_basis = irbasis.load('F', lambda_)
        f_data = cls._precompute_stat(f_basis, ncoeff=ncoeff)
        b_basis = irbasis.load('B', lambda_)
        b_data = cls._precompute_stat(b_basis, ncoeff=ncoeff)

        f_data.other_uxl = np.asarray([f_basis.ulx_all_l(xi)[:f_data.uxl.shape[1]]
                                       for xi in b_data.xsample])
        b_data.other_uxl = np.asarray([b_basis.ulx_all_l(xi)[:b_data.uxl.shape[1]]
                                       for xi in f_data.xsample])
        return (f_data, b_data)

    @classmethod
    def from_archive(cls, ar, stat):
        if stat not in ('fermi', 'bose'):
            raise ValueError("statistics must be either 'fermi' or 'bose'")
        if ar.attrs["repr"] != "IR":
            raise ValueError("archive does not contain correct representation")

        lambda_ = ar.attrs["lambda"]
        group = ar[stat]
        xsample = group["xsample"][...]
        wsample = group["wsample"][...]
        uxl = group["uxl"][...]
        ux1l = group["ux1l"][...]
        uwl = group["uwl"][...]
        other_uxl = group["other_uxl"][...]
        return cls(lambda_, stat, xsample, wsample, uxl, ux1l, uwl, other_uxl)

    def to_archive(self, ar):
        ar.attrs["repr"] = "IR"
        ar.attrs["lambda"] = self.lambda_
        group = ar.require_group(self.stat)
        group["xsample"] = self.xsample
        group["wsample"] = self.wsample
        group["uxl"] = self.uxl
        group["ux1l"] = self.ux1l
        group["uwl"] = self.uwl
        group["other_uxl"] = self.other_uxl

    def __init__(self, lambda_, stat, xsample, wsample, uxl, ux1l, uwl, other_uxl):
        xsize, order = uxl.shape
        wsize, _ = uwl.shape
        self.lambda_ = lambda_
        self.stat = stat
        self.xsample = xsample
        self.wsample = wsample
        self.uxl = uxl
        self.ux1l = ux1l
        self.uwl = uwl
        self.other_uxl = other_uxl

class IntermediateRepr:
    @classmethod
    def _load_precomputed(cls, lambda_, stat, datadir, ncoeff):
        if datadir is None:
            return None
        llambda = int(np.log10(lambda_))
        if 10 ** llambda != lambda_:
            return None
        if ncoeff is None:
            fname = os.path.join(datadir, 'ir', '1e%d.hdf5' % llambda)
        else:
            fname = os.path.join(datadir, 'ir', '1e%d_%d.hdf5' % (llambda, ncoeff))
        try:
            f = hdf5.File(fname, 'r')
        except:
            return None
        else:
            return IntermediateData.from_archive(f, stat)

    def __init__(self, lambda_, beta, stat='fermi',
                 datadir=None, data=None, ncoeff=None):
        if data is None:
            data = self._load_precomputed(lambda_, stat, datadir, ncoeff)
        if data is None:
            statletter = {'fermi':'F', 'bose':'B'}[stat]
            basis = irbasis.load(statletter, lambda_)
            data = IntermediateData._precompute_stat(basis, ncoeff)
        if data.lambda_ != lambda_ or data.stat != stat:
            raise ValueError("Precomputed data does not match")

        self.lambda_ = lambda_
        self.stat = stat
        self.beta = beta

        # Things for tau axis and Matsubara for fermions
        self.order = data.uxl.shape[-1]
        self.tau = tau_for_x(data.xsample, beta)
        self.ntau = self.tau.size
        self.wnn = data.wsample
        self.wn = 2.0*np.pi/beta * (self.wnn + (0.5 if stat == 'fermi' else 0))
#        print ("IR freq", self.wn)
        self.nwn = self.wn.size

        # Matrices
        self.utaul = np.sqrt(2/beta) * data.uxl
        self.utau1l = np.sqrt(2/beta) * data.ux1l
        if data.other_uxl is not None:
            self.other_utaul = np.sqrt(2/beta) * data.other_uxl
        self.ultau = np.linalg.pinv(self.utaul)
        self.uwl = np.sqrt(beta) * data.uwl
        self.ulw = np.linalg.pinv(self.uwl)

    def fit_tau(self, ftau):
        ftau = np.asarray(ftau)
        if ftau.shape[0] != self.ntau:
            raise ValueError("First index must be tau points")
        return np.einsum('ij,j...->i...', self.ultau, ftau)

    def eval_tau(self, fl):
        fl = np.asarray(fl)
        aux.verifyexp(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be IR")
        return np.einsum('ij,j...->i...', self.utaul, fl)

    def eval_other_tau(self, fl):
        fl = np.asarray(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be IR")
        return np.einsum('ij,j...->i...', self.other_utaul, fl)

    def eval_beta(self, fl):
        fl = np.asarray(fl)
        aux.verifyexp(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be IR")
        return np.einsum('i,i...->...', self.utau1l, fl)

    def fit_iw(self, fwn):
        fwn = np.asarray(fwn)
        if fwn.shape[0] != self.nwn:
            raise ValueError("First index must be Matsubara")
        return np.einsum('ij,j...->i...', self.ulw, fwn)

    def eval_iw(self, fl):
        fl = np.asarray(fl)
        aux.verifyexp(fl)
        if fl.shape[0] != self.order:
            raise ValueError("First index must be IR")
        return np.einsum('ij,j...->i...', self.uwl, fl)

    def leakage(self, fl):
        fl = np.asarray(fl)
        return np.abs(fl[-1]).max() / np.abs(fl[0]).max()
