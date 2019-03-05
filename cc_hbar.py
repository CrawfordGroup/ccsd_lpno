'''
HelperHbar class definition and function definitions
For building spin-adapted CC Hbar matrix elements
'''

import numpy as np
import psi4
from ndot import ndot


class HelperHbar(object):
    def __init__(self, hcc, ccsd_e):

        # Get fock matrix, MOs(ERIs), t amplitudes from ccsd
        self.F = hcc.F
        self.F_occ = hcc.F_occ
        self.F_vir = hcc.F_vir
        self.MO = hcc.MO
        self.t_ia = hcc.t_ia
        self.t_ijab = hcc.t_ijab
        self.no_occ = hcc.no_occ

        self.Ecc = ccsd_e

        # Setting up intermediates
        self.Lmnef = self.make_Lmnef()
        self.Lmnie = self.make_Lmnie()
        self.Lamef = self.make_Lamef()

    # Functions to build 1-body Hbar
    # F_mi = f_mi + t_ie f_me + (t_inef + t_ie *t_nf) * (2<mn|ef> - <mn|fe>) + t_ne (2<mn|ie> - <mn|ei>)
    def make_Hoo(self):
        H_oo = self.F_occ.copy()
        H_oo += ndot('ie,me->mi', self.t_ia, self.F[:self.no_occ, self.no_occ:])
        H_oo += ndot('inef,mnef->mi',self.t_ijab, self.Lmnef)
        H_oo += np.einsum('ie,nf,mnef->mi',self.t_ia, self.t_ia, self.Lmnef)
        H_oo += ndot('ne,mnie->mi', self.t_ia, self.Lmnie)
        return H_oo

    # F_me = f_me + t_nf (2 <mn|ef> - <mn|fe>)
    def make_Hov(self):
        H_ov = self.F[:self.no_occ, self.no_occ:].copy()
        H_ov += ndot('nf,mnef->me', self.t_ia, self.Lmnef)
        return H_ov

    # F_ae = f_ae + t_ma f_me - (t_mnfa + t_mf *t_na) * (2<mn|fe> - <mn|ef>) + t_mf (2<am|ef> - <am|fe>)
    def make_Hvv(self):
        H_vv = self.F[self.no_occ:, self.no_occ:].copy()
        H_vv += ndot('ma,me->ae', self.t_ia, self.F[:self.no_occ, self.no_occ:])
        H_vv -= ndot('mnfa,mnfe->ae', self.t_ijab, self.Lmnef)
        H_vv -= np.einsum('mf,na,mnfe->ae', self.t_ia, self.t_ia, self.Lmnef)
        H_vv += ndot('mf,amef->ae', self.t_ia, self.Lamef)
        return H_vv

    # F_ai = f_ai + t_ie f_ae + t_ma f_mi + t_me (2<am|ie> - <am|ei>) - (t_imea + t_ie t_ma) * f_me - (t_mnea + t_me t_na) * (2<mn|ei> - <mn|ie>) + (t_imef + t_ie t_mf) * (2<am|ef> - <am|fe>)
    # Fai += - 0.5 * (t_me t_niaf + t_na t_mief + t_if t_nmea) * (2<mn|ef> - <mn|fe>) + t_me t_nifa (2<mn|ef> - <mn|fe>) - t_me t_if t_na * (2<mn|ef> - <mn|fe>)


    # W_mnij = <mn|ij> + t_je <mn|ie> + t_ijef <mn|ef> + t_ie t_jf <mn|ef>
    def make_Hoooo(self):
        H_oooo = self.MO[:self.no_occ, :self.no_occ, :self.no_occ, :self.no_occ].copy()
        H_oooo += ndot('je,mnie->mnij', self.t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        H_oooo += ndot('ie,mnej->mnij', self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        H_oooo += ndot('ijef,mnef->mnij', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_oooo += np.einsum('ie,jf,mnef->mnij', self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return H_oooo

    # W_abef = <ab|ef> - 2 t_nb <an|ef> + t_mnab <mn|ef> + t_ma t_nb <mn|ef>
    def make_Hvvvv(self):
        H_vvvv = self.MO[self.no_occ:, self.no_occ:, self.no_occ:, self.no_occ:].copy()
        H_vvvv -= ndot('nb,anef->abef',self.t_ia, self.MO[self.no_occ:, :self.no_occ, self.no_occ:, self.no_occ:])
        H_vvvv -= ndot('na,nbef->abef',self.t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        H_vvvv += ndot('mnab,mnef->abef', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_vvvv += np.einsum('ma,nb,mnef->abef', self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return H_vvvv

    # W_amef = <am|ef> - t_na <nm|ef>
    def make_Hvovv(self):
        H_vovv = self.MO[self.no_occ:, :self.no_occ, self.no_occ:, self.no_occ:].copy()
        H_vovv -= ndot('na,nmef->amef', self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return H_vovv

    # W_mnie = <mn|ie> + t_if <mn|fe>
    def make_Hooov(self):
        H_ooov = self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:].copy()
        H_ooov += ndot('if,mnfe->mnie', self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return H_ooov

    # Wmbej = <mb|ej> + t_jf <mb|ef> - t_nb <mn|ej> - t_jnfb <mn|ef> - t_jf t_nb <mn|ef> + t_njfb self.Lmnef
    def make_Hovvo(self):
        H_ovvo = self.MO[:self.no_occ, self.no_occ:, self.no_occ:, :self.no_occ].copy()
        H_ovvo += ndot('jf,mbef->mbej', self.t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        H_ovvo -= ndot('nb,mnej->mbej', self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        H_ovvo -= ndot('jnfb,mnef->mbej', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_ovvo -= np.einsum('jf,nb,mnef->mbej', self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_ovvo += ndot('njfb,mnef->mbej', self.t_ijab, self.Lmnef)
        return H_ovvo

    # Wmbje = <mb|je> + t_jf <bm|ef> - t_nb <mn|je> - t_jnfb <nm|ef> - t_jf t_nb <nm|ef>
    def make_Hovov(self):
        H_ovov = self.MO[:self.no_occ, self.no_occ:, :self.no_occ, self.no_occ:].copy()
        H_ovov += ndot('jf,bmef->mbje', self.t_ia, self.MO[self.no_occ:, :self.no_occ, self.no_occ:, self.no_occ:])
        H_ovov -= ndot('nb,mnje->mbje', self.t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        H_ovov -= ndot('jnfb,nmef->mbje', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_ovov -= np.einsum('jf,nb,nmef->mbje', self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return H_ovov

    # Wabei = <ab|ei> + t_if <ab|ef> - t_mb <am|ei> - t_ma <bm|ie> - (t_imfb + t_if t_mb) <am|ef> - (t_imfa + t_if t_ma) <mb|ef> + (t_mnab + t_ma t_nb) <mn|ei> - t_miab f_me
    # Wabei += t_mifb self.Lamef + (t_if t_mnab + t_ma t_nibf + t_nb t_miaf) <mn|ef> - (t_mf t_niab + t_na t_mifb) Lmnfe + t_if t_ma t_nb <nm|fe>
    def make_Hvvvo(self):
        H_vvvo = self.MO[self.no_occ:, self.no_occ:, self.no_occ:, :self.no_occ].copy()
        H_vvvo += ndot('if,abef->abei', self.t_ia, self.MO[self.no_occ:, self.no_occ:, self.no_occ:, self.no_occ:])
        H_vvvo -= ndot('mb,amei->abei', self.t_ia, self.MO[self.no_occ:, :self.no_occ, self.no_occ:, :self.no_occ])
        H_vvvo -= ndot('ma,bmie->abei', self.t_ia, self.MO[self.no_occ:, :self.no_occ, :self.no_occ, self.no_occ:])
        H_vvvo -= ndot('imfb,amef->abei', self.t_ijab, self.MO[self.no_occ:, :self.no_occ, self.no_occ:, self.no_occ:])
        H_vvvo -= np.einsum('if,mb,amef->abei', self.t_ia, self.t_ia, self.MO[self.no_occ:, :self.no_occ, self.no_occ:, self.no_occ:])
        H_vvvo -= ndot('imfa,mbef->abei', self.t_ijab, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        H_vvvo -= np.einsum('if,ma,mbef->abei', self.t_ia, self.t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        H_vvvo += ndot('mnab,mnei->abei', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        H_vvvo += np.einsum('ma,nb,mnei->abei', self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        H_vvvo -= ndot('miab,me->abei', self.t_ijab, self.F[:self.no_occ, self.no_occ:])
        H_vvvo += ndot('mifb,amef->abei', self.t_ijab, self.Lamef)
        H_vvvo += np.einsum('if,mnab,mnef->abei', self.t_ia, self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_vvvo += np.einsum('ma,nibf,mnef->abei', self.t_ia, self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_vvvo += np.einsum('nb,miaf,mnef->abei', self.t_ia, self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_vvvo -= np.einsum('mf,niab,mnfe->abei', self.t_ia, self.t_ijab, self.Lmnef)
        H_vvvo -= np.einsum('na,mifb,mnfe->abei', self.t_ia, self.t_ijab, self.Lmnef)
        H_vvvo += np.einsum('if,ma,nb,nmfe->abei', self.t_ia, self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return H_vvvo

    # Wmbij = <mb|ij> + t_je <mb|ie> -t_nb <mn|ij> + t_ie <bm|je> - (t_ineb + t_ie t_nb) <nm|je> - (t_jneb + t_je t_nb) <mn|ie> + (t_ijef + t_ie t_jf) <mb|ef> + t_ijeb fme 
    # Wmbij += t_njeb Lmnie - (t_je t_infb + t_if t_jneb + t_nb t_jief) <mn|ef> + t_ie t_njfb Lmnef + t_nf t_ijeb Lmnef - t_je t_if t_nb <mn|ef>
    def make_Hovoo(self):
        H_ovoo = self.MO[:self.no_occ, self.no_occ:, :self.no_occ, :self.no_occ].copy()
        H_ovoo += ndot('je,mbie->mbij', self.t_ia, self.MO[:self.no_occ, self.no_occ:, :self.no_occ, self.no_occ:])
        H_ovoo -= ndot('nb,mnij->mbij', self.t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, :self.no_occ])
        H_ovoo += ndot('ie,mbej->mbij', self.t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, :self.no_occ])
        H_ovoo -= ndot('ineb,mnej->mbij', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        H_ovoo -= np.einsum('ie,nb,mnej->mbij', self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, :self.no_occ])
        H_ovoo -= ndot('jneb,mnie->mbij', self.t_ijab, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        H_ovoo -= np.einsum('je,nb,mnie->mbij', self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:])
        H_ovoo += ndot('ijef,mbef->mbij', self.t_ijab, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        H_ovoo += np.einsum('ie,jf,mbef->mbij', self.t_ia, self.t_ia, self.MO[:self.no_occ, self.no_occ:, self.no_occ:, self.no_occ:])
        H_ovoo += ndot('ijeb,me->mbij', self.t_ijab, self.F[:self.no_occ, self.no_occ:])
        H_ovoo += ndot('jnbe,mnie->mbij',self.t_ijab, self.Lmnie)
        H_ovoo -= np.einsum('je,infb,mnfe->mbij', self.t_ia, self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_ovoo -= np.einsum('if,jneb,mnfe->mbij', self.t_ia, self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_ovoo -= np.einsum('nb,ijef,mnef->mbij', self.t_ia, self.t_ijab, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        H_ovoo += np.einsum('ie,njfb,mnef->mbij', self.t_ia, self.t_ijab, self.Lmnef)
        H_ovoo += np.einsum('nf,ijeb,mnef->mbij', self.t_ia, self.t_ijab, self.Lmnef)
        H_ovoo -= np.einsum('ie,jf,nb,mnef->mbij', self.t_ia, self.t_ia, self.t_ia, self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:])
        return H_ovoo
       
    def make_Lmnef(self):
        tmp = self.MO[:self.no_occ, :self.no_occ, self.no_occ:, self.no_occ:].copy()
        Lmnef = 2.0 * tmp - tmp.swapaxes(2, 3) 
        return Lmnef

    def make_Lmnie(self):
        tmp = self.MO[:self.no_occ, :self.no_occ, :self.no_occ, self.no_occ:].copy()
        Lmnie = 2.0 * tmp - tmp.swapaxes(0, 1) 
        return Lmnie

    def make_Lamef(self):
        tmp = self.MO[self.no_occ:, :self.no_occ, self.no_occ:, self.no_occ:].copy()
        Lamef = 2.0 * tmp - tmp.swapaxes(2, 3) 
        return Lamef

    '''def make_Lamie(self):
        Lamie = 2.0*self.MO[self.no_occ:, :self.no_occ, :self.no_occ, self.no_occ:].copy()
        Lamie -= self.MO[self.no_occ:, :self.no_occ, self.no_occ:, :self.no_occ]
        return Lamie'''

