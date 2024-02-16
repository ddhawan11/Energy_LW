import numpy as np
import scipy.linalg as LA
import math
import pickle

def import_density(n,fname):
    ell = open(fname,'r').readlines()
    rho = np.zeros((n,n))
    m   = 0
    for p in range(n):
        for q in range(n):
            rho[p,q] = float(ell[m].split()[2])
            m += 1
    return rho



def G(h1,S,omega,mu=0.0):
    return (1j*omega+mu)*S-h1

def get_Gnot(mol_data,rho,x_mesh,full_Ham=True):
    if (full_Ham):
        n          = mol_data['n']
        S          = np.eye(n)
        h1         = mol_data['h1']
        h2         = mol_data['h2']
        h1_eff     = h1
    else:
        t_file = open("T.obj",'rb')
        v_file = open("V.obj",'rb')
        h1_eff = pickle.load(t_file)
        h2     = pickle.load(v_file)
        h1     = mol_data['h1']
        n      = h1.shape[0]
        S      = np.eye(n)

        t_file.close()
        v_file.close()

    F          = h1_eff+(2*np.einsum('ikjl,jl->ik',h2,rho)-np.einsum('iljk,jl->ik',h2,rho))

    gf_nonint  = np.zeros((len(x_mesh),n,n),dtype=complex)

    for ix,x in enumerate(x_mesh):
        gf_nonint[ix,:,:] = G(h1,S,x,0.0)
    
    return gf_nonint
