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
from .local import *
from psi4 import constants as pc 

# Bring in wfn from psi4
def plusplus_correction(wfn, omega_nm, mol, method='polar', gauge='length', localize=False, pert=None, pno_cut=0): 
    
    # Create Helper_local object
    if localize:
        no_occ = wfn.doccpi()[0]
        no_vir = wfn.nmo() - wfn.doccpi()[0] - wfn.frzcpi()[0]
        local = HelperLocal(wfn.doccpi()[0], no_vir)
    else:
        local=None

    # Create Helper_CCenergy object
    hcc = HelperCCEnergy(wfn, local=local, pert=pert, pno_cut=pno_cut) 
    ccsd_e = hcc.do_CC(local=local, e_conv=1e-10, r_conv =1e-10, maxiter=0, start_diis=0)

    print('CCSD correlation energy: {}'.format(ccsd_e))
    # Create HelperCCHbar object
    hbar = HelperHbar(hcc, ccsd_e)

    # Create HelperLamdba object
    lda = HelperLambda(hcc, hbar)

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
            pert[string] = HelperPert(hcc, hbar, lda, Mu[string], omega, local=local)

            i += 1

        for string in ['X', 'Y', 'Z']:
            for string2 in ['X', 'Y', 'Z']:
                x_ijab = np.reshape(pert[string].x_ijab, (no_occ * no_occ, no_vir, no_vir))
                y_ijab = np.reshape(pert[string].y_ijab, (no_occ * no_occ, no_vir, no_vir))

                for ij in range(self.no_occ * self.no_occ):
                    rm_pairs = no_vir - int(local.s_pairs[ij])
                    if rm_pairs == 0:
                        continue
                    i = ij // no_occ
                    j = ij % no_occ
                    if i == j:
                        Q_compute_s = local.Q[i*no_occ+i, :, :rm_pairs]
                        x_ia = contract('Ab,b->A', Q_compute_s.T, pert[string].x_ia[i])
                        y_ia = contract('Ab,b->A', Q_compute_s.T, pert[string].y_ia[i])

                    Q_compute_d = local.Q[ij, :, :rm_pairs]
                    new_x = contract('Aa,ab,bB->AB', Q_compute_d.T, x_ijab[ij], Q_compute_d)
                    new_y = contract('Aa,ab,bB->AB', Q_compute_d.T, y_ijab[ij], Q_compute_d)

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
    elif method=='optrot':
        if gauge=='both':

            ### Length gauge OR calculation
            ### Form of linear response function: <<mu;L>>

            # Get the perturbation mu
            dipole_array = hcc.mints.ao_dipole()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            Mu = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, Mu[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] - betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            # Have to multiply with omega for length gauge
            optrot_lg = prefactor * trace * omega

            ### Velocity gauge OR calculation
            ### Form of linear response function: <<p;L>>

            # Get the perturbation P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_vg = prefactor * trace
            # So velocity gauge is / omega

            ### Modified velocity gauge OR calculation
            ### Form of linear response function: <<p;L>> - <<p;L>>_0
            ### Using the velocity gauge OR value and subtracting the static value
            omega = 0.0

            # Get the perturbation P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_diff = prefactor * trace

            optrot_mvg = optrot_vg - optrot_diff
            
            return optrot_lg, optrot_mvg
            
        elif gauge=='length':

            ### Length gauge OR calculation
            ### Form of linear response function: <<mu;L>>

            # Get the perturbation mu
            dipole_array = hcc.mints.ao_dipole()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            Mu = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                Mu[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(dipole_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, Mu[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] - betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            # Have to multiply with omega for length gauge
            optrot_lg = prefactor * trace * omega
            
            return optrot_lg

        elif gauge=='velocity':
            ### Velocity gauge OR calculation
            ### Form of linear response function: <<p;L>>

            # Get the perturbation P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_vg = prefactor * trace
            # So velocity gauge is / omega

            ### Modified velocity gauge OR calculation
            ### Form of linear response function: <<p;L>> - <<p;L>>_0
            ### Using the velocity gauge OR value and subtracting the static value
            omega = 0.0

            # Get the perturbation P
            p_array = hcc.mints.ao_nabla()

            # Get the angular momentum L
            angular_momentum = hcc.mints.ao_angular_momentum()

            # Create HelperPert objects for both
            P = {}
            pert1 = {}
            L = {}
            pert2 = {}

            # Rosenfeld tensor
            beta = {}
            betap = {}
            beta_new = {}

            i=0
            for string in ['X', 'Y', 'Z']:
                P[string] = np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(p_array[i]))
                pert1[string] = HelperPert(hcc, hbar, lda, P[string], omega, local=local)
                L[string] = -0.5 * np.einsum('uj,vi,uv', hcc.C_arr, hcc.C_arr, np.asarray(angular_momentum[i]))
                pert2[string] = HelperPert(hcc, hbar, lda, L[string], omega, local=local)

                i+=1
                for hand in ['right', 'left']:
                    pseudoresponse1 = pert1[string].iterate(hand, r_conv=1e-10, local=local)
                for hand in ['right', 'left']:
                    pseudoresponse2 = pert2[string].iterate(hand, r_conv=1e-10, local=local)

            print('Rosenfeld tensor:')
            for string1 in ['X', 'Y', 'Z']:
                for string2 in ['X', 'Y', 'Z']:
                    beta[string1+string2] = HelperResp(lda, pert1[string1], pert2[string2]).linear_resp()
                    betap[string1+string2] = HelperResp(lda, pert2[string2], pert1[string1]).linear_resp()

                    beta_new[string1+string2] = 0.5 * (beta[string1+string2] + betap[string1+string2])
                    print(' {} {} : {}'.format(string1, string2, beta_new[string1+string2]))

            trace = 0.0
            for string in ['XX','YY','ZZ']:
                trace += beta_new[string]
            trace /= 3.0

            # Calculation of the specific rotation
            Mass = 0
            for atom in range(mol.natom()):
                Mass += mol.mass(atom)
            h_bar = pc.h / (2.0 * np.pi)
            prefactor = -72e6 * h_bar**2 * pc.na / (pc.c**2 * pc.me**2 * Mass)
            optrot_diff = prefactor * trace

            optrot_mvg = optrot_vg - optrot_diff

            return optrot_mvg

