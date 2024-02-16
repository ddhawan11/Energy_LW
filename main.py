import self_energy
import aux
import repn
import os
import argparse
import numpy as np
import h5py as hdf5
import gf_nonint
import pickle
def get_parser():
    parser = argparse.ArgumentParser(description='self-energy calculation')
    parser.add_argument('--beta', dest='beta', type=float, default=100,
                        help='inverse temperature')
    parser.add_argument('--ncoeff', dest='ncoeff', type=int, default=136,
                        help='number of coefficients in the representation')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=1e5,
                        help='Lambda for IR')

    return parser


parser = get_parser()
args = parser.parse_args()
scriptpath = os.path.dirname(os.path.realpath(__file__))

frepn = repn.IntermediateRepr(args.lambda_, args.beta, 'fermi',
                                      datadir=os.path.join(scriptpath, 'data'),
                                      ncoeff=args.ncoeff)
brepn = repn.IntermediateRepr(args.lambda_, args.beta, 'bose',
                                      datadir=os.path.join(scriptpath, 'data'),
                                      ncoeff=args.ncoeff)

a = 2
#giw = np.load("qse_green_function_freq_fci.npy", allow_pickle=True)
#f1 = open("gfQuantitiesSAO.obj", "rb")
#gf_and_se = pickle.load(f1)
#print(gf_and_se["giw"].shape)
#giw = gf_and_se["giw"]
#giw = giw/2
with hdf5.File("sim.h5", "r") as gf:
    gf_fci = gf["results/G_ij_omega/data"][...]
    omega =  gf["results/G_ij_omega/mesh/1/points"][...]

gf_fci_new = np.zeros((gf_fci.shape[0],a,a), dtype=np.complex128)
for x in range(gf_fci.shape[0]):
    for i in range(a):
        for k in range(a):
            gf_fci_new[x,i,k] = complex(gf_fci[x,i*a+k,0,0], gf_fci[x,i*a+k,0,1])
giw = gf_fci_new

gl = frepn.fit_iw(giw)
gtau = frepn.eval_tau(gl)

####Load U from h_dict.npy
mol_data = np.load('h_dict.npy',allow_pickle=True).item()
h0 = mol_data["h1"]
U  = mol_data["h2"]

interaction = self_energy.Interaction(U, frepn, brepn)
sigma2tau = interaction.sigma2(gtau)

sigma2l = frepn.fit_tau(sigma2tau)
sigma2iw = frepn.eval_iw(sigma2l)

np.save("sigma2iw_qse_fci.npy", sigma2iw)

nmo = 2
rho = -frepn.eval_beta(gl)
#rho = gf_nonint.import_density(nmo,'vqe_q_uccsd_1rdm.txt')/2.0

sigma1 = interaction.sigma1(rho)
e0 = 2 * aux.checkreal(rho.dot(h0).trace())
esigma1 = aux.checkreal(rho.dot(sigma1).trace())
trgsigmaiw = np.einsum('Nij,Nji->N', giw, sigma2iw)
trgsigmal = frepn.fit_iw(trgsigmaiw)
esigma2 = -aux.checkreal(frepn.eval_beta(trgsigmal))

print(e0, esigma1, esigma2, e0+esigma1+esigma2)
