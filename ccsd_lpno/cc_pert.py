'''
HelperPert class definition and function definitions
For iteratively solving for first order T amplitudes 
and first order lagrangian multipliers, then solving
for the linear response function for two given 
perturbations, after completing a CCSD calculation and
building Hbar matrix elements, lambdas
'''

import numpy as np
import psi4
from opt_einsum import contract
from .diis import *

class HelperPert(object):
    def __init__(self, ccsd, hbar, lda, A, omega):

        # Get MOs from lda
        self.MO = ccsd.MO
        self.t_ia = ccsd.t_ia
        self.t_ijab = ccsd.t_ijab
        self.no_occ = ccsd.no_occ
        self.no_vir = ccsd.no_vir
        
        # L intermediates
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

        # Get lambdas
        self.l_ia = lda.l_ia
        self.l_ijab = lda.l_ijab
    
        # Get perturbation, wavelength
        self.A = A
        self.omega = omega
        #print('perturbation: \n{}\nomega(hartree): {}\n'.format(A, omega))

        self.D_ia = self.Hoo.diagonal().reshape(-1,1) - self.Hvv.diagonal()
        self.D_ia += omega
        self.D_ijab = self.Hoo.diagonal().reshape(-1,1,1,1) + self.Hoo.diagonal().reshape(-1,1,1) - self.Hvv.diagonal().reshape(-1,1) - self.Hvv.diagonal()
        self.D_ijab += omega

        # Guesses for X1 and X2 amplitudes (First order perturbed T amplitudes)
        self.x_ia = self.make_Avo().swapaxes(0,1)/self.D_ia
        self.pertbar_ijab = self.make_Avvoo().swapaxes(0,2).swapaxes(1,3)
        self.x_ijab = self.pertbar_ijab.copy()
        self.x_ijab += self.pertbar_ijab.swapaxes(0,1).swapaxes(2,3)
        self.x_ijab = self.x_ijab/self.D_ijab


        # Guesses for Y1 and Y2 amplitudes (First order perturbed Lambda amplitudes)
        self.y_ia =  2.0 * self.x_ia.copy()
        self.y_ijab =  4.0 * self.x_ijab.copy()
        self.y_ijab -= 2.0 * self.x_ijab.swapaxes(2,3)

    def make_Zvv(self):
        Zvv = 0
        Zvv += 2.0 * contract('amef,mf->ae', self.Hvovv, self.x_ia)
        Zvv -= contract('amfe,mf->ae', self.Hvovv, self.x_ia)
        Zvv -= contract('mnaf,mnef->ae', self.x_ijab, self.Lmnef)
        return Zvv

    def make_Zoo(self):
        Zoo = 0
        Zoo -= 2.0 * contract('mnie,ne->mi', self.Hooov, self.x_ia)
        Zoo += contract('nmie,ne->mi', self.Hooov, self.x_ia)
        Zoo -= contract('mnef,inef->mi', self.Lmnef, self.x_ijab)
        return Zoo

    def make_Goo(self, t_ijab, l_ijab):
        Goo = 0
        Goo += contract('mjab,ijab->mi', t_ijab, l_ijab)
        return Goo

    def make_Gvv(self, t_ijab, l_ijab):
        Gvv = 0
        Gvv -= contract('ijab,ijeb->ae', t_ijab, l_ijab)
        return Gvv
    

    # Matrix elements of the perturbation
    def make_Aoo(self):
        Aoo = self.A[:self.no_occ, :self.no_occ].copy()
        Aoo += contract('ia,ma->mi', self.t_ia, self.A[:self.no_occ, self.no_occ:])
        return Aoo

    def make_Aov(self):
        Aov = self.A[:self.no_occ, self.no_occ:].copy()
        return Aov

    def make_Avv(self):
        Avv = self.A[self.no_occ:,self.no_occ:].copy()
        Avv -= contract('ma,me->ae', self.t_ia, self.A[:self.no_occ,self.no_occ:])
        return Avv

    def make_Avo(self):
        Avo = self.A[self.no_occ:,:self.no_occ].copy()
        Avo += contract('ie,ae->ai', self.t_ia, self.A[self.no_occ:,self.no_occ:])
        Avo -= contract('ma,mi->ai', self.t_ia, self.A[:self.no_occ,:self.no_occ])
        #temp = 2.0 * self.t_ijab - self.t_ijab.swapaxes(0,1)
        #Avo += contract('miea,me->ai', temp, self.A[:self.no_occ,self.no_occ:])
        #temp = contract('ie,ma->imea', self.t_ia, self.t_ia)
        #Avo -= contract('miea,me->ai', temp, self.A[:self.no_occ,self.no_occ:])
        Avo += 2.0 * contract('miea,me->ai', self.t_ijab, self.A[:self.no_occ, self.no_occ:])
        Avo -= contract('imea,me->ai', self.t_ijab, self.A[:self.no_occ, self.no_occ:])
        temp = contract('ie,ma->imea', self.t_ia, self.t_ia)
        Avo -= contract('imea,me->ai', temp, self.A[:self.no_occ, self.no_occ:])
        return Avo

    def make_Aovoo(self):
        Aovoo = 0
        Aovoo += contract('ijeb,me->mbij', self.t_ijab, self.A[:self.no_occ,self.no_occ:])
        return Aovoo

    def make_Avvvo(self):
        Avvvo = 0
        Avvvo -= contract('miab,me->abei', self.t_ijab, self.A[:self.no_occ,self.no_occ:])
        return Avvvo

    def make_Avvoo(self):
        Avvoo = 0
        Avvoo += contract('ijeb,ae->abij', self.t_ijab, self.make_Avv())
        Avvoo -= contract('mjab,mi->abij', self.t_ijab, self.make_Aoo())
        return Avvoo

    def update_xs(self, x_ia, x_ijab):
    # X1 equations
        r_ia = self.make_Avo().swapaxes(0,1).copy()
        r_ia -= self.omega * x_ia.copy()
        r_ia += contract('ae,ie->ia', self.Hvv, x_ia)
        r_ia -= contract('mi,ma->ia', self.Hoo, x_ia)
        r_ia += 2.0 * contract('maei,me->ia', self.Hovvo, x_ia)
        r_ia -= contract('maie,me->ia', self.Hovov, x_ia)
        r_ia += 2.0 * contract('me,miea->ia', self.Hov, x_ijab)
        r_ia -= contract('me,imea->ia', self.Hov, x_ijab)
        r_ia += 2.0 * contract('amef,imef->ia', self.Hvovv, x_ijab)
        r_ia -= contract('amfe,imef->ia', self.Hvovv, x_ijab)
        r_ia -= 2.0 * contract('mnie,mnae->ia', self.Hooov, x_ijab)
        r_ia += contract('nmie,mnae->ia', self.Hooov, x_ijab)

    # X2 equations
        r_ijab = self.make_Avvoo().swapaxes(0,2).swapaxes(1,3).copy()
        r_ijab -= 0.5 * self.omega * self.x_ijab
        r_ijab += contract('abej,ie->ijab', self.Hvvvo, x_ia)
        r_ijab -= contract('mbij,ma->ijab', self.Hovoo, x_ia)
        r_ijab += contract('ae,ijeb->ijab', self.Hvv, x_ijab)
        r_ijab -= contract('mi,mjab->ijab', self.Hoo, x_ijab)
        r_ijab += 0.5 * contract('mnij,mnab->ijab', self.Hoooo, x_ijab)
        r_ijab += 0.5 * contract('abef,ijef->ijab', self.Hvvvv, x_ijab)
        r_ijab += 2.0 * contract('mbej,miea->ijab', self.Hovvo, x_ijab)
        r_ijab -= contract('mbje,miea->ijab', self.Hovov, x_ijab)
        r_ijab -= contract('maje,imeb->ijab', self.Hovov, x_ijab)
        r_ijab -= contract('mbej,imea->ijab', self.Hovvo, x_ijab)
        r_ijab += contract('mi,mjab->ijab', self.make_Zoo(), self.t_ijab)
        r_ijab += contract('ijeb,ae->ijab', self.t_ijab, self.make_Zvv())
        
        new_xia = x_ia.copy()
        new_xijab = x_ijab.copy()

        new_xia += r_ia/self.D_ia
        temp = r_ijab/self.D_ijab
        new_xijab += temp + temp.swapaxes(0,1).swapaxes(2,3)
        
        return new_xia, new_xijab

    def inhomogeneous_ys(self, x_ia, x_ijab):
    # Y1 equations, inhomogeneous terms
        r_ia = 2.0 * self.make_Aov().copy()
        r_ia -= contract('ma,im->ia', self.l_ia, self.make_Aoo())
#        r_ia += contract('ie,ae->ia', self.l_ia, self.make_Avv())
        r_ia += contract('ie,ea->ia', self.l_ia, self.make_Avv())
        r_ia += contract('imef,efam->ia', self.l_ijab, self.make_Avvvo())
        # above should be okay
        r_ia -= 0.5 * contract('mnea,ienm->ia', self.l_ijab, self.make_Aovoo())
        r_ia -= 0.5 * contract('mnae,iemn->ia', self.l_ijab, self.make_Aovoo())
        # <0|[Hbar, X1]|i a>
        r_ia += 2.0 * contract('imae,me->ia', self.Lmnef, x_ia)
        # <0|L1[Hbar, X1]|i a>
        temp = -1.0 * contract('ma,ie->miae', self.Hov, self.l_ia)
        temp -= contract('ie,ma->miae', self.Hov, self.l_ia)
        temp -= 2.0 * contract('mina,ne->miae', self.Hooov, self.l_ia)
        temp += contract('imna,ne->miae', self.Hooov, self.l_ia)
        temp -= 2.0 * contract('imne,na->miae', self.Hooov, self.l_ia)
        temp += contract('mine,na->miae', self.Hooov, self.l_ia)
        temp += 2.0 * contract('fmae,if->miae', self.Hvovv, self.l_ia)
        temp -= contract('fmea,if->miae', self.Hvovv, self.l_ia)
        temp += 2.0 * contract('fiea,mf->miae', self.Hvovv, self.l_ia)
        temp -= contract('fiae,mf->miae', self.Hvovv, self.l_ia)
        r_ia += contract('miae,me->ia', temp, x_ia)
        # <0|L1[Hbar, X2]|i a>
        r_ia += 2.0 * contract('imae,mnef,nf->ia', self.Lmnef, x_ijab, self.l_ia)
        r_ia -= contract('imae,mnfe,nf->ia', self.Lmnef, x_ijab, self.l_ia)
        r_ia -= contract('mi,ma->ia', self.make_Goo(x_ijab, self.Lmnef), self.l_ia)
        #r_ia += contract('ea,ie->ia', self.make_Gvv(x_ijab, self.Lmnef), self.l_ia)
        r_ia += contract('ie,ea->ia', self.l_ia, self.make_Gvv(x_ijab, self.Lmnef))

        # <0|L2[Hbar, X1]|i a>
        temp = -1.0 * contract('mfna,nief->iema', self.Hovov, self.l_ijab)
        temp -= contract('ifne,nmaf->iema', self.Hovov, self.l_ijab)
        temp -= contract('mfan,inef->iema', self.Hovvo, self.l_ijab)
        temp -= contract('ifen,nmfa->iema', self.Hovvo, self.l_ijab)
        temp += 0.5 * contract('fgae,imfg->iema', self.Hvvvv, self.l_ijab)
        temp += 0.5 * contract('fgea,imgf->iema', self.Hvvvv, self.l_ijab)
        temp += 0.5 * contract('imno,onea->iema', self.Hoooo, self.l_ijab)
        temp += 0.5 * contract('mino,noea->iema', self.Hoooo, self.l_ijab)
        r_ia += contract('iema,me->ia', temp, x_ia)
        r_ia += contract('imaf,fe,me->ia', self.Lmnef, self.make_Gvv(self.t_ijab, self.l_ijab), x_ia)
        r_ia += contract('mief,fa,me->ia', self.Lmnef, self.make_Gvv(self.t_ijab, self.l_ijab), x_ia)
        r_ia -= contract('mnea,ni,me->ia', self.Lmnef, self.make_Goo(self.t_ijab, self.l_ijab), x_ia)
        r_ia -= contract('inae,nm,me->ia', self.Lmnef, self.make_Goo(self.t_ijab, self.l_ijab), x_ia)
        # <0|L2[Hbar, X2]|i a>
        r_ia -= contract('ma,mi->ia', self.Hov, self.make_Goo(x_ijab, self.l_ijab))
        r_ia += contract('ie,ea->ia', self.Hov, self.make_Gvv(x_ijab, self.l_ijab))
        r_ia -= contract('gnea,mnef,imfg->ia', self.Hvovv, x_ijab, self.l_ijab)
        r_ia -= contract('gnae,mnef,mifg->ia', self.Hvovv, x_ijab, self.l_ijab)
        r_ia -= contract('gief,mnef,mnga->ia', self.Hvovv, x_ijab, self.l_ijab)
        r_ia += 2.0 * contract('gmae,nifg,mnef->ia', self.Hvovv, self.l_ijab, x_ijab)
        r_ia -= contract('gmea,nifg,mnef->ia', self.Hvovv, self.l_ijab, x_ijab)
        r_ia -= 2.0 * contract('fiea,fe->ia', self.Hvovv, self.make_Gvv(self.l_ijab, x_ijab))
        r_ia += contract('fiae,fe->ia', self.Hvovv, self.make_Gvv(self.l_ijab, x_ijab))
        r_ia += contract('mnoa,mnef,oief->ia', self.Hooov, x_ijab, self.l_ijab)
        r_ia += contract('inoe,mnef,mofa->ia', self.Hooov, x_ijab, self.l_ijab)
        r_ia += contract('miof,mnef,onea->ia', self.Hooov, x_ijab, self.l_ijab)
        r_ia -= 2.0 * contract('mioa,mo->ia', self.Hooov, self.make_Goo(x_ijab, self.l_ijab))
        r_ia += contract('imoa,mo->ia', self.Hooov, self.make_Goo(x_ijab, self.l_ijab))
        r_ia -= 2.0 * contract('imoe,nofa,mnef->ia', self.Hooov, self.l_ijab, x_ijab)
        r_ia += contract('mioe,nofa,mnef->ia', self.Hooov, self.l_ijab, x_ijab)
        
    # Y2 equations, inhomogeneous terms
        # <0|L1 Abar|ij ab>
        r_ijab = 2.0 * contract('jb,ia->ijab', self.make_Aov(), self.l_ia)
        r_ijab -= contract('ib,ja->ijab', self.make_Aov(), self.l_ia)
        # <0|L2 Abar|ij ab>
        r_ijab += contract('ijeb,ea->ijab', self.l_ijab, self.make_Avv())
        r_ijab -= contract('mjab,im->ijab', self.l_ijab, self.make_Aoo())
        # <0|L1[Hbar, X1]|ij ab>
        r_ijab -= contract('mieb,ja,me->ijab', self.Lmnef, self.l_ia, x_ia)
        r_ijab -= contract('ijae,mb,me->ijab', self.Lmnef, self.l_ia, x_ia)
        r_ijab -= contract('jmba,ie,me->ijab', self.Lmnef, self.l_ia, x_ia)
        r_ijab += 2.0 * contract('imae,jb,me->ijab', self.Lmnef, self.l_ia, x_ia)
        # Ashutosh's code has an extra term here, with the first term being a 2 * einsum
        # They should be the same without the 2 (but they're not!!)
        #r_ijab -= contract('miba,je,me->ijab', self.Lmnef, self.l_ia, x_ia)
        # <0|L2[Hbar, X1]|ij ab>
        r_ijab -= contract('ma,me,ijeb->ijab', self.Hov, x_ia, self.l_ijab)
        r_ijab -= contract('ie,me,jmba->ijab', self.Hov, x_ia, self.l_ijab)
        r_ijab -= contract('fmba,me,ijef->ijab', self.Hvovv, x_ia, self.l_ijab)
        r_ijab -= contract('fjea,me,mifb->ijab', self.Hvovv, x_ia, self.l_ijab)
        r_ijab -= contract('fibe,me,jmfa->ijab', self.Hvovv, x_ia, self.l_ijab)
        r_ijab += 2.0 * contract('fmae,me,ijfb->ijab', self.Hvovv, x_ia, self.l_ijab)
        r_ijab -= contract('fmea,me,ijfb->ijab', self.Hvovv, x_ia, self.l_ijab)
        #r_ijab += 2.0 * contract('fjeb,me,imaf->ijab', self.Hvovv, x_ia, self.l_ijab)
        #r_ijab -= contract('fjbe,me,imaf->ijab', self.Hvovv, x_ia, self.l_ijab)
        # why?
        r_ijab += 2.0 * contract('fiea,me,jmbf->ijab', self.Hvovv, x_ia, self.l_ijab)
        r_ijab -= contract('fiae,me,jmbf->ijab', self.Hvovv, x_ia, self.l_ijab)
        r_ijab += contract('jmna,me,ineb->ijab', self.Hooov, x_ia, self.l_ijab)
        r_ijab += contract('mjna,me,nieb->ijab', self.Hooov, x_ia, self.l_ijab)
        r_ijab += contract('jine,me,mnab->ijab', self.Hooov, x_ia, self.l_ijab)
        r_ijab -= 2.0 * contract('mina,me,njeb->ijab', self.Hooov, x_ia, self.l_ijab)
        r_ijab += contract('imna,me,njeb->ijab', self.Hooov, x_ia, self.l_ijab)
        r_ijab -= 2.0 * contract('imne,me,jnba->ijab', self.Hooov, x_ia, self.l_ijab)
        r_ijab += contract('mine,me,jnba->ijab', self.Hooov, x_ia, self.l_ijab)
        # <0|L2[Hbar, X2]|ij ab>
        r_ijab += 0.5 * contract('mnab,ijef,mnef->ijab', self.MO[:self.no_occ,:self.no_occ,self.no_occ:,self.no_occ:], self.l_ijab, x_ijab)
        r_ijab += 0.5 * contract('ijfe,mnef,mnba->ijab', self.MO[:self.no_occ,:self.no_occ,self.no_occ:,self.no_occ:], x_ijab, self.l_ijab)
        r_ijab += contract('jnae,mifb,mnef->ijab', self.MO[:self.no_occ,:self.no_occ,self.no_occ:,self.no_occ:], self.l_ijab, x_ijab)
        r_ijab += contract('njae,imfb,mnef->ijab', self.MO[:self.no_occ,:self.no_occ,self.no_occ:,self.no_occ:], self.l_ijab, x_ijab)
        r_ijab -= contract('inae,mjfb,mnef->ijab', self.Lmnef, self.l_ijab, x_ijab)
        r_ijab -= contract('jnba,in->ijab', self.l_ijab, self.make_Goo(self.Lmnef, x_ijab))
        r_ijab += contract('ijfb,af->ijab', self.l_ijab, self.make_Gvv(self.Lmnef, x_ijab))
        r_ijab += contract('ijae,be->ijab', self.Lmnef, self.make_Gvv(self.l_ijab, x_ijab))
        r_ijab -= contract('imab,jm->ijab', self.Lmnef, self.make_Goo(self.l_ijab, x_ijab))
        r_ijab -= contract('mjea,nifb,mnef->ijab', self.Lmnef, self.l_ijab, x_ijab)
        r_ijab += 2.0 * contract('imae,njfb,mnef->ijab', self.Lmnef, self.l_ijab, x_ijab)

        #r_y2 = np.load('ry2_iter1.npy')
        #print("Checking inhomogeneous terms: {}".format(np.allclose(r_y2, r_ijab, atol=1e-7)))

        return r_ia, r_ijab

    def update_ys(self, y_ia, y_ijab):
    # Y1 equations, homogeneous terms

        # y_ia = 2 * Hov + y_ie H_ea - y_ma H_im + y_me (2 * H_ieam - H_iema) + y_imef H_efam - y_mnae Hiemn
        #       - G_ef (2 * H_eifa - H_eiaf) - G_mn (2 * Hmina - H_imna)
        r_ia = self.inhmy_ia.copy()
        r_ia += self.omega * y_ia.copy()
        r_ia += contract('ie,ea->ia', y_ia, self.Hvv)
        r_ia -= contract('ma,im->ia', y_ia, self.Hoo)
        r_ia += 2.0 * contract('me,ieam->ia', y_ia, self.Hovvo)
        r_ia -= contract('me,iema->ia', y_ia, self.Hovov)
        r_ia += contract('imef,efam->ia', y_ijab, self.Hvvvo)
        r_ia -= contract('mnae,iemn->ia', y_ijab, self.Hovoo)
        r_ia -= 2.0 * contract('eifa,ef->ia', self.Hvovv, self.make_Gvv(self.y_ijab, self.t_ijab))
        r_ia += contract('eiaf,ef->ia', self.Hvovv, self.make_Gvv(self.y_ijab, self.t_ijab))
        r_ia -= 2.0 * contract('mina,mn->ia', self.Hooov, self.make_Goo(self.t_ijab, self.y_ijab))
        r_ia += contract('imna,mn->ia', self.Hooov, self.make_Goo(self.t_ijab, self.y_ijab))

    # Y2 equations, homogeneous terms

        # y_ijab = 2 <ij|ab> - <ij|ba> + 2 * y_ia H_jb - y_ja H_ib + y_ijeb H_ea - y_mjab H_im + 0.5 * y_mnab H_ijmn
        #          + 0.5 y_ijef H_efab + y_ie (2 * H_ejab - H_ejba) - y_mb (2 * H_jima - H_ijma)
        #          + y_mjeb (2 * H_ieam - H_iema) - y_mibe H_jema - y_mieb H_jeam
        #          + G_ae (2 <ij|eb> - <ij|be>) - G_mi (2 <mj|ab> - <mj|ba>)
        # y_ijab = y_ijab + y_jiba

        r_ijab = self.inhmy_ijab.copy()
        r_ijab += 0.5 * self.omega * y_ijab.copy()
        r_ijab += 2.0 * contract('ia,jb->ijab', y_ia, self.Hov)
        r_ijab -= contract('ja,ib->ijab', y_ia, self.Hov)
        r_ijab += contract('ijeb,ea->ijab', y_ijab, self.Hvv)
        r_ijab -= contract('mjab,im->ijab', y_ijab, self.Hoo)
        r_ijab += 0.5 * contract('mnab,ijmn->ijab', y_ijab, self.Hoooo)
        r_ijab += 0.5 * contract('ijef,efab->ijab', y_ijab, self.Hvvvv)
        r_ijab += 2.0 * contract('ie,ejab->ijab', y_ia, self.Hvovv)
        r_ijab -= contract('ie,ejba->ijab', y_ia, self.Hvovv)
        r_ijab -= 2.0 * contract('mb,jima->ijab', y_ia, self.Hooov)
        r_ijab += contract('mb,ijma->ijab', y_ia, self.Hooov)
        r_ijab += 2.0 * contract('mjeb,ieam->ijab', y_ijab, self.Hovvo)
        r_ijab -= contract('mjeb,iema->ijab', y_ijab, self.Hovov)
        r_ijab -= contract('mibe,jema->ijab', y_ijab, self.Hovov)
        r_ijab -= contract('mieb,jeam->ijab', y_ijab, self.Hovvo)
        r_ijab += contract('ae,ijeb->ijab', self.make_Gvv(y_ijab, self.t_ijab), self.Lmnef)
        r_ijab -= contract('mi,mjab->ijab', self.make_Goo(self.t_ijab, y_ijab), self.Lmnef)

        new_yia = y_ia.copy()
        new_yia += r_ia / self.D_ia
        new_yijab = y_ijab.copy()
        temp = r_ijab / self.D_ijab
        new_yijab += temp + temp.swapaxes(0,1).swapaxes(2,3)
        #print("Checking y2 here: \n{}".format(new_yijab[0]))

        return new_yia, new_yijab

    # compute pseudoresponse
    def pseudo_response(self, z_ia, z_ijab):
        polar1 = 0
        polar2 = 0
        polar1 = 2.0 * contract('ia,ai->', z_ia, self.make_Avo())
        temp = self.pertbar_ijab + self.pertbar_ijab.swapaxes(0,1).swapaxes(2,3)
        polar2 = 2.0 * contract('ijab,ijab->', z_ijab, temp)
        polar2 -= contract('ijba,ijab->', z_ijab, temp)

        return -2.0 * (polar1 + polar2)

    # iterate until convergence
    def iterate(self, hand, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=0): 
        print('Iteration\t\t Pseudoresponse\t\tRMS')
        if hand == 'right':
            new_presp = self.pseudo_response(self.x_ia, self.x_ijab)
            # Set up DIIS
            diis = HelperDIIS(self.x_ia, self.x_ijab, max_diis)
        else:
            new_presp = self.pseudo_response(self.y_ia, self.y_ijab)
            # Prep inhomogeneous terms before iterations start
            self.inhmy_ia, self.inhmy_ijab = self.inhomogeneous_ys(self.x_ia, self.x_ijab)
            # Set up DIIS
            diis = HelperDIIS(self.y_ia, self.y_ijab, max_diis)


        print('CCPert {} Iteration: 0\t {:2.12f}'.format(hand, new_presp))
        for i in range(maxiter):
            if hand == 'right':
                new_xia, new_xijab = self.update_xs(self.x_ia, self.x_ijab)
                new_presp = self.pseudo_response(new_xia, new_xijab)
                rms = np.linalg.norm(new_xia - self.x_ia)
                rms += np.linalg.norm(new_xijab - self.x_ijab)
            else:
                new_yia, new_yijab = self.update_ys(self.y_ia, self.y_ijab)
                new_presp = self.pseudo_response(new_yia, new_yijab)
                rms = np.linalg.norm(new_yia - self.y_ia)
                rms += np.linalg.norm(new_yijab - self.y_ijab)

            print('CCPert {} Iteration: {:3d}\t {:2.12f}\t{:1.12f}'.format(hand, i+1, new_presp, rms))
            if(abs(rms) < r_conv):
                print('{}-hand convergence reached.\n Pseudoresponse: {}\n'.format(hand, new_presp))
                if hand == 'right':
                    self.x_ia = new_xia
                    self.x_ijab = new_xijab
                else:
                    self.y_ia = new_yia
                    self.y_ijab = new_yijab
                break

            if hand == 'right':
                # Update error vectors for DIIS
                diis.update_err_list(new_xia, new_xijab)
                # Extrapolate using DIIS
                if(i >= start_diis):
                    new_xia, new_xijab = diis.extrapolate(new_xia, new_xijab)
                self.x_ia = new_xia
                self.x_ijab = new_xijab
            else:
                # Update error vectors for DIIS
                diis.update_err_list(new_yia, new_yijab)
                # Extrapolate using DIIS
                if(i >= start_diis):
                    new_yia, new_yijab = diis.extrapolate(new_yia, new_yijab)
                self.y_ia = new_yia
                self.y_ijab = new_yijab

        return new_presp

class HelperResp(object):
    def __init__(self, lda, B, pertA):

        # Get lambdas
        self.l_ia = lda.l_ia
        self.l_ijab = lda.l_ijab
        
        # Get Perturbation
        # Here A is the perturbation for which X's and Y's are calculated
        # And B is the operator whose expectation value is the linresp function
        # <B>^(1) = <0|Y(A) * B_bar|0> + <0| (1+L(0)) * [B_bar, X(A)]|0>
        self.B = B
        self.pertA = pertA
        self.x_ia = pertA.x_ia
        self.x_ijab = pertA.x_ijab
        self.y_ia = pertA.y_ia
        self.y_ijab = pertA.y_ijab

    def linear_resp(self):
        linresp = 0.0
        # <0| B_bar X1 |0>
        linresp += 2.0 * contract('ia,ia->', self.B.make_Aov(), self.x_ia)
        # <0| L1 B_bar X1 |0>
        linresp += contract('ca,ia,ic->', self.B.make_Avv(), self.x_ia, self.l_ia) #*
        linresp -= contract('ik,ia,ka->', self.B.make_Aoo(), self.x_ia, self.l_ia) #*
        # <0| L1 B_bar X2 |0>
        linresp += 2.0 * contract('jb,ijab,ia->', self.B.make_Aov(), self.x_ijab, self.l_ia)
        linresp -= contract('jb,ijba,ia->', self.B.make_Aov(), self.x_ijab, self.l_ia)
        # <0| L2 B_bar X1 |0>
        linresp -= 0.5 * contract('kbij,ka,ijab->', self.B.make_Aovoo(), self.x_ia, self.l_ijab)
        linresp += contract('bcaj,ia,ijbc->', self.B.make_Avvvo(), self.x_ia, self.l_ijab)
        linresp -= 0.5 * contract('kaji,kb,ijab->', self.B.make_Aovoo(), self.x_ia, self.l_ijab)
        # <0| L2 B_bar X2 |0>
        linresp -= 0.5 * contract('ki,kjab,ijab->', self.B.make_Aoo(), self.x_ijab, self.l_ijab)
        linresp -= 0.5 * contract('kj,kiba,ijab->', self.B.make_Aoo(), self.x_ijab, self.l_ijab)
        linresp += 0.5 * contract('ac,ijcb,ijab->', self.B.make_Avv(), self.x_ijab, self.l_ijab)
        linresp += 0.5 * contract('bc,ijac,ijab->', self.B.make_Avv(), self.x_ijab, self.l_ijab)
        #print("Polar2 : {}".format(linresp))
        # <0| Y1 B_bar |0>
        linresp += contract('ai,ia->', self.B.make_Avo(), self.y_ia)
        # <0| Y2 B_bar |0>
        linresp += 0.5 * contract('abij,ijab->', self.B.make_Avvoo(), self.y_ijab)
        linresp += 0.5 * contract('baji,ijab->', self.B.make_Avvoo(), self.y_ijab)

        linresp *= -1.0

        return linresp
