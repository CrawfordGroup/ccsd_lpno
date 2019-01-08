'''
HelperHbar class definition and function definitions
For building spin-adapted CC Hbar matrix elements
'''

import numpy as np
import psi4
from ndot import ndot


class HelperHbar(object):
    def __init__(self, rhf_wfn, hcc, ccsd_e):

        # Get fock matrix, MOs(ERIs), t amplitudes from ccsd
        self.F = hcc.F
        self.F_occ = hcc.F_occ
        self.F_vir = hcc.F_vir
        self.MO = hcc.MO
        self.t_ia = hcc.t_ia
        self.t_ijab = hcc.t_ijab

        self.Ecc = ccsd_e

    # Functions to build 1-body Hbar
    # F_mi = f_mi + t_ie f_me + (t_inef + t_ie *t_nf) * (2<mn|ef> - <mn|fe>) + t_ne (2<mn|ie> - <mn|ei>)
    def make_Hoo(self, t_ia, t_ijab, Lmnef, Lmnie):
        H_oo = self.F_occ.copy()
        H_oo += ndot('ie,me->mi', t_ia, self.F[:no_occ, no_occ:])
        H_oo += ndot('inef,mnef->mi',t_ijab, Lmnef)
        H_oo += np.einsum('ie,nf,mnef->mi',t_ia, t_ia, Lmnef)
        H_oo += ndot('ne,mnie->mi', t_ia, self.Lmnie)
        return H_oo

    # F_me = f_me + t_nf (2 <mn|ef> - <mn|fe>)
    def make_Hov(self, t_ia, Lmnef):
        H_ov = self.F[:no_occ, no_occ:].copy()
        H_ov += ndot('nf,mnef->me', t_ia, Lmnef)
        return H_ov

    # F_ae = f_ae + t_ma f_me - (t_mnfa + t_mf *t_na) * (2<mn|fe> - <mn|ef>) + t_mf (2<am|ef> - <am|fe>)
    def make_Hvv():

    # F_ai = f_ai + t_ie f_ae + t_ma f_mi + t_me (2<am|ie> - <am|ei>) - (t_imea + t_ie t_ma) * f_me - (t_mnea + t_me t_na) * (2<mn|ei> - <mn|ie>) + (t_imef + t_ie t_mf) * (2<am|ef> - <am|fe>)
    # Fai += - 0.5 * (t_me t_niaf + t_na t_mief + t_if t_nmea) * (2<mn|ef> - <mn|fe>) + t_me t_nifa (2<mn|ef> - <mn|fe>) - t_me t_if t_na * (2<mn|ef> - <mn|fe>)
    def make_Hvo():

    def make_Lmnef(self):
        Lmnef = 2.0*self.MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
        Lmnef -= self.MO[:no_occ, :no_occ, no_occ:, no_occ:]
        return Lmnef

    def make_Lmnie(self):
        Lmnie = 2.0*self.MO[:no_occ, :no_occ, :no_occ, no_occ:].copy()
        Lmnie -= self.MO[:no_occ, :no_occ, no_occ:, :no_occ]
        return Lmnie

    def make_Lamef(self):
        Lamef = 2.0*self.MO[no_occ:,:no_occ, no_occ:, no_occ:].copy()
        Lamef -= self.MO[no_occ:, :no_occ, no_occ:, no_occ:]
        return Lamef

    def make_Lamie(self):
        Lamie = 2.0*self.MO[no_occ:, :no_occ, :no_occ, no_occ:].copy()
        Lamie -= self.MO[no_occ:, :no_occ, no_occ:, :no_occ]
        return Lamie
