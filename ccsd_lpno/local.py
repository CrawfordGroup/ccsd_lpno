
import numpy as np
import psi4
from opt_einsum import contract


class HelperLocal(object):
    def __init__(self, ccsd):
        # regular localization init
        # MO build
        self.no_fz = ccsd.no_fz
        self.no_occ = ccsd.no_occ
        basis = ccsd.wfn.basisset()
        # Localizing occupied orbitals using Boys localization procedure
        Local = psi4.core.Localizer.build("BOYS", basis, ccsd.C_occ)
        Local.localize()
        new_C_occ = Local.L
        nc_arr = ccsd.wfn.Ca().to_array()
        nco_arr = new_C_occ.to_array()
        print("Checking dimensions of localized active occupied:\nShape of local array: {}\nShape of C: {}\n".format(nco_arr.shape, nc_arr.shape))
        nc_arr[:, self.no_fz:(self.no_fz + self.no_occ)] = nco_arr[:,:]
        new_C = psi4.core.Matrix.from_array(nc_arr)
        print("Shape of new MO coeff matrix: {}".format(new_C.shape))
        self.MO_nfz = np.asarray(self.mints.mo_eri(new_C, new_C, new_C, new_C))
        self.MO = self.MO_nfz[self.no_fz:, self.no_fz:, self.no_fz:, self.no_fz:]
        print("Checking size of ERI tensor: {}".format(self.MO.shape))
        print("Checking size of ERI_nfz tensor: {}".format(self.MO_nfz.shape))
        
        # AO basis Fock matrix build
        De = contract('ui,vi->uv', nc_arr[:, :(self.no_fz+self.no_occ)], nc_arr[:, :(self.no_fz+self.no_occ)])
        self.F = ccsd.H + 2.0 * contract('pqrs,rs->pq', ccsd.I, De) - contract('prqs,rs->pq', ccsd.I, De)
        self.F_nfz = contract('uj, vi, uv', new_C, new_C, self.F)
        self.F = self.F_nfz[self.no_fz:, self.no_fz:]
        print("Checking size of Fock matrix: {}".format(self.F.shape))
        self.F_ao = self.H + 2.0 * contract('pqrs,rs->pq', ccsd.I, De) - contract('prqs,rs->pq', ccsd.I, De)
        hf_e = contract('pq,pq->', ccsd.H + self.F_ao, De)

    def init_PNOs(self, pno_cut, ccsd, pert):

        if pert == 'on':
            print('Pert switch on. Initializing pert PNOs')
            # Build guess Abar, guess Hbar
            # Abar_ijab = P_ij^ab (t_ij^eb A_ae - t_mj^ab A_mi)

            # Hbar_ii  = f_ii + t_inef ( 2 * <in|ef> - <in|fe> ) 
            # Hbar_aa = f_aa - t_mnfa (2 * <mn|fa> - <mn|af> )

            # Build guess X's
            # X_ijab = Abar_ijab / Hbar_ii + Hbar_jj - Hbar_aa _ Hbar_bb

            # Identify weak pairs using MP2 pseudoresponse
            # requires the building of the guess Abar matrix and guess X's


