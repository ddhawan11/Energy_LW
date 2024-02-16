import numpy as np
import sys

def _write(what, out=sys.stderr):
    out.write(what)
    out.flush()
    
class Interaction:
    """Class encapsulating the computation of Sigma given an interaction"""
    def __init__(self, uchem, frepn, brepn):
        uchem = np.asarray(uchem)
        norb = uchem.shape[0]
        if uchem.shape != (norb,) * 4:
            raise ValueError("U matrix is not a 4-cube")
        if frepn is not None and frepn.stat != 'fermi':
            raise ValueError("You must give a fermionic repn")
        if brepn is not None and brepn.stat != 'bose':
            raise ValueError("You must give a bosonic repn")
        self.norb = norb
        self.uchem = uchem
        self.frepn = frepn
        self.brepn = brepn

    def sigma1(self, rho):
        "Return Hartree-Fock self-energy"
        return (2 * np.einsum('ikjl,jl->ik', self.uchem, rho)
                - np.einsum('iljk,jl->ik', self.uchem, rho))

    def sigma2(self, gtau):
        "Compute second-order self-energy"
#        print ("shape", gtau.shape)
        ntau = gtau.shape[0]
        if gtau.shape != (ntau, self.norb, self.norb):
            raise ValueError("Invalid shape of gtau")

        result = np.nan * np.zeros_like(gtau)
        for itau in range(ntau):
            for j in range(self.norb):
                # This is a little bit of magic here: profiling shows that
                # einsum(...) is faster with Fortran order and whenever summing
                # over final indices
                gtau_inv = np.array(gtau[-1-itau].T, order='F')
                contr = self.uchem[:,:,j,:]
                contr = np.einsum('ikl,lJ->iJk', contr, gtau[itau])
                contr = np.einsum('iJk,kI->JIi', contr, gtau[itau])
                contr = np.einsum('JIi,iK->IKJ', contr, gtau_inv)
                resj = 2 * np.einsum('IKJ,IKJL->L', contr, self.uchem)
                contr = contr.transpose(2, 1, 0).copy()
                resj -= np.einsum('IKJ,IKJL->L', contr, self.uchem)
                result[itau, j, :] = resj
            self._draw_dot(itau, ntau)

        return result

    def _draw_dot(self, i, n):
        if not (i % (n // 10 + 1)):
            _write(".")
