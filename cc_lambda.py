'''
HelperLambda class definition and function definitions
For iteratively solving for lambda lagrangian multipliers
After completing a CCSD calculation and building Hbar matrix elements
'''

import numpy as np
import psi4
from diis import *
from opt_einsum import contract

class HelperLambda(object):
    def __init__(self, hcc, hbar):

        # Get fock matrix, ERIs, T amplitudes from CCSD
        self.F = hcc.F
        self.F_occ = hcc.F_occ
        self.F_vir = hcc.F_vir
        self.MO = hcc.MO
        self.t_ia = hcc.t_ia
        self.t_ijab = hcc.t_ijab
        self.no_occ = hcc.no_occ
        self.no_mo = hcc.no_mo
        self.no_vir = self.no_mo - self.no_occ
        self.d_ia = hcc.d_ia
        self.d_ijab = hcc.d_ijab

        # Setting up intermediates
        self.Lmnef = hbar.Lmnef
        self.Lmnie = hbar.Lmnie
        self.Lamef = hbar.Lamef

        # Get 1-body Hbar elements
        self.Hoo = hbar.Hoo
        self.Hvv = hbar.Hvv
        self.Hov = hbar.Hov

        # Get 2-body Hbar elements
        self.Hoooo = hbar.Hoooo
        self.Hvvvv = hbar.Hvvvv
        self.Hvovv = hbar.Hvovv
        self.Hooov = hbar.Hooov
        self.Hovvo = hbar.Hovvo
        self.Hovov = hbar.Hovov
        self.Hvvvo = hbar.Hvvvo
        self.Hovoo = hbar.Hovoo

        # Init guesses, 2.0 * t1 -> l1, 4.0 * t2 - 2.0 t2.swap(2,3) -> l2
        self.l_ia = 2.0 * self.t_ia.copy()
        self.l_ijab = 4.0 * self.t_ijab.copy()
        self.l_ijab -= 2.0 * self.t_ijab.swapaxes(2,3)

    def make_Goo(self):
        Goo = contract('mjab,ijab->mi', self.t_ijab, self.l_ijab)
        return Goo

    def make_Gvv(self):
        Gvv = -1.0 * contract('ijab,ijeb->ae', self.l_ijab, self.t_ijab)
        return Gvv

    def update_ls(self, l_ia, l_ijab):

        Gvv = self.make_Gvv()
        Goo = self.make_Goo()

        # l_ia = 2 * Hov + l_ie H_ea - l_ma H_im + l_me (2 * H_ieam - H_iema) + l_imef H_efam - l_mnae Hiemn
        #       - G_ef (2 * H_eifa - H_eiaf) - G_mn (2 * Hmina - H_imna)
        Ria = 2.0 * self.Hov.copy()
        Ria += contract('ie,ea->ia', l_ia, self.Hvv)
        Ria -= contract('ma,im->ia', l_ia, self.Hoo)
        Ria += 2.0 * contract('me,ieam->ia', l_ia, self.Hovvo)
        Ria -= contract('me,iema->ia', l_ia, self.Hovov)
        Ria += contract('imef,efam->ia', l_ijab, self.Hvvvo)
        Ria -= contract('mnae,iemn->ia', l_ijab, self.Hovoo)
        Ria -= 2.0 * contract('eifa,ef->ia', self.Hvovv, Gvv)
        Ria += contract('eiaf,ef->ia', self.Hvovv, Gvv)
        Ria -= 2.0 * contract('mina,mn->ia', self.Hooov, Goo)
        Ria += contract('imna,mn->ia', self.Hooov, Goo)

        # l_ijab = 2 <ij|ab> - <ij|ba> + 2 * l_ia H_jb - l_ja H_ib + l_ijeb H_ea - l_mjab H_im + 0.5 * l_mnab H_ijmn
        #          + 0.5 l_ijef H_efab + l_ie (2 * H_ejab - H_ejba) - l_mb (2 * H_jima - H_ijma)
        #          + l_mjeb (2 * H_ieam - H_iema) - l_mibe H_jema - l_mieb H_jeam
        #          + G_ae (2 <ij|eb> - <ij|be>) - G_mi (2 <mj|ab> - <mj|ba>)
        # l_ijab = l_ijab + l_jiba

        Rijab = self.Lmnef.copy()
        Rijab += 2.0 * contract('ia,jb->ijab', l_ia, self.Hov)
        Rijab -= contract('ja,ib->ijab', l_ia, self.Hov)
        Rijab += contract('ijeb,ea->ijab', l_ijab, self.Hvv)
        Rijab -= contract('mjab,im->ijab', l_ijab, self.Hoo)
        Rijab += 0.5 * contract('mnab,ijmn->ijab', l_ijab, self.Hoooo)
        Rijab += 0.5 * contract('ijef,efab->ijab', l_ijab, self.Hvvvv)
        Rijab += 2.0 * contract('ie,ejab->ijab', l_ia, self.Hvovv)
        Rijab -= contract('ie,ejba->ijab', l_ia, self.Hvovv)
        Rijab -= 2.0 * contract('mb,jima->ijab', l_ia, self.Hooov)
        Rijab += contract('mb,ijma->ijab', l_ia, self.Hooov)
        Rijab += 2.0 * contract('mjeb,ieam->ijab', l_ijab, self.Hovvo)
        Rijab -= contract('mjeb,iema->ijab', l_ijab, self.Hovov)
        Rijab -= contract('mibe,jema->ijab', l_ijab, self.Hovov)
        Rijab -= contract('mieb,jeam->ijab', l_ijab, self.Hovvo)
        Rijab += contract('ae,ijeb->ijab', Gvv, self.Lmnef)
        Rijab -= contract('mi,mjab->ijab', Goo, self.Lmnef)

        Rijab += Rijab.swapaxes(0, 1).swapaxes(2, 3)

        new_lia = l_ia.copy()
        new_lia += Ria / self.d_ia
        new_lijab = l_ijab.copy()
        new_lijab += Rijab / self.d_ijab
        
        return new_lia, new_lijab

    def pseudo_energy(self, l_ijab):
        o = slice(0, self.no_occ)
        v = slice(self.no_occ, self.no_mo)
        # E = 1/2 <ab|ij> l_ijab
        E_pseudo = 0.5 * contract('abij,ijab->', self.MO[v, v, o, o], l_ijab)
        return E_pseudo

    def iterate(self, e_conv=1e-8, r_conv=1e-7, maxiter=40, max_diis=8, start_diis=0):
        self.old_pe = self.pseudo_energy(self.l_ijab)
        print('Iteration\t\t Pseudoenergy\t\tDifference\tRMS')
        # Set up DIIS
        diis = HelperDIIS(self.l_ia, self.l_ijab, max_diis)

        for i in range(maxiter):
            new_lia, new_lijab = self.update_ls(self.l_ia, self.l_ijab)
            new_pe = self.pseudo_energy(new_lijab)
            rms = np.linalg.norm(new_lia - self.l_ia)
            rms += np.linalg.norm(new_lijab - self.l_ijab)
            print('CC Iteration: {:3d}\t {:2.12f}\t{:1.12f} \t{:1.12f}\tDIIS size: {}'.format(i, new_pe, abs(new_pe - self.old_pe), rms, diis.diis_size))
            if(abs(new_pe - self.old_pe) < e_conv and abs(rms) < r_conv):
                print('Convergence reached.\n Pseudoenergy: {}\n'.format(new_pe))
                self.l_ia = new_lia
                self.l_ijab = new_lijab
                break
            # Update error vectors for DIIS
            diis.update_err_list(new_lia, new_lijab)

            # Extrapolate using DIIS
            if(i >= start_diis):
                new_lia, new_lijab = diis.extrapolate(new_lia, new_lijab)

            self.l_ia = new_lia
            self.l_ijab = new_lijab
            self.old_pe = new_pe

        return new_pe
