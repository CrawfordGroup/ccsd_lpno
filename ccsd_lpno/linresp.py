'''
Computing the CCSD-LR dipole polarizability  using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4
from .helper_cc import *
from .cc_hbar import *
from .cc_lambda import *
from .cc_pert import *
from psi4 import constants as pc 

# Bring in wfn from psi4
def do_linresp(local, pno_cut, wfn, omega_nm, method='polar'): 
    # Create Helper_CCenergy object
    hcc = HelperCCEnergy(local, pno_cut, wfn) 
    ccsd_e = hcc.do_CC(local=False, e_conv=1e-10, r_conv =1e-10, maxiter=40, start_diis=0)

    print('CCSD correlation energy: {}'.format(ccsd_e))
    # Create HelperCCHbar object
    hbar = HelperHbar(hcc, ccsd_e)

    # Create HelperLamdba object
    lda = HelperLambda(hcc, hbar)
    pseudo_e = lda.iterate(e_conv=1e-8, r_conv =1e-10, maxiter=30)

    # Set the frequency in hartrees
    omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)

    if method=='polar':
        # Get the perturbation A for Xs and Ys
        dipole_array = hcc.mints.ao_dipole()

        # Create HelperPert object and solve for xs and ys
        Mu = {}
        pert = {}
        hresp = {}
        polar = {}

        i=0
        for string in ['X', 'Y', 'Z']:
            Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
            pert[string] = HelperPert(hcc, hbar, lda, Mu[string], omega)

            i += 1
            for hand in ['right', 'left']:
                pseudoresponse = pert[string].iterate(hand, r_conv=1e-10)

        for string in ['X', 'Y', 'Z']:
            for string2 in ['X', 'Y', 'Z']:
                hresp[string+string2] = HelperResp(lda, pert[string], pert[string2])
                polar[string+string2] = hresp[string+string2].linear_resp()
            
        print('Polarizability tensor:')
        for string in ['X', 'Y', 'Z']:
            for string2 in ['X', 'Y', 'Z']:
                if string != string2:
                    polar[string+string2+'_new'] = 0.5 * (polar[string+string2] + polar[string2+string])
                else:
                    polar[string+string2+'_new'] = polar[string+string2]
                print("{} {}: {}\n".format(string, string2, polar[string+string2+'_new']))

        trace = polar['XX'] + polar['YY'] + polar['ZZ']
        isotropic_polar = trace / 3.0

        return isotropic_polar
