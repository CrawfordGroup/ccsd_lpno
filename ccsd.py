'''
Computing the CCSD correlation energy using an RHF reference
References used:
    - http://github.com/CrawfordGroup/ProgrammingProjects
    - Stanton:1991:4334
    - https://github.com/psi4/psi4numpy
'''

import numpy as np
import psi4

import time

psi4.core.clean()

# Set memory
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
np.set_printoptions(precision=8, linewidth=400, suppress=True)
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk', 'mp2_type': 'conv',
                  'freeze_core': 'false', 'e_convergence': 1e-12,
                  'd_convergence': 1e-10, 'save_jk': 'true'})

# Set for CCSD
E_conv = 1e-6
maxiter = 30
print_amps = False
compare_psi4 = False

# Set for LPNO
local=True
#local=False
e_cut = 1e-6

# N dimensional dot
# Like a mini DPD library
# Using from helper_CC.py by dgasmith
def ndot(input_string, op1, op2, prefactor=None):
    """
    No checks, if you get weird errors its up to you to debug.

    ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    # Tensordot axes
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s), )
        right_pos += (input_right.find(s), )
    tdot_axes = (left_pos, right_pos)

    # Get result ordering
    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    # Matrix multiply
    # No transpose needed
    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_removed, dim_right))

    # If we have to transpose vector-matrix, einsum is faster
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2)
        used_einsum = True

    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    # Make sure the resulting shape is correct
    if (new_view.shape != shape_result) and not used_einsum:
        if (len(shape_result) > 0):
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    # In-place mult by prefactor if requested
    if prefactor is not None:
        new_view *= prefactor

    # Do final tranpose if needed
    if used_einsum:
        return new_view
    elif tdot_result == output_ind:
        return new_view
    else:
        return np.einsum(tdot_result + '->' + output_ind, new_view)

# Compute RHF energy with psi4
e_scf, wfn = psi4.energy('SCF', return_wfn=True)

print('SCF energy: {}\n'.format(e_scf))

# Get no_occ, no_mo, e_scf(?), eps
C = wfn.Ca()
no_occ = wfn.doccpi()[0]
no_mo = wfn.nmo()
eps = np.asarray(wfn.epsilon_a())
J = wfn.jk().J()[0].to_array()
K = wfn.jk().K()[0].to_array()

mints = psi4.core.MintsHelper(wfn.basisset())

H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

F = H + 2 * J - K

# Make spin-orbital MO integrals
MO = np.asarray(mints.mo_eri(C, C, C, C))

# Need to change ERIs to physicist notation
MO = MO.swapaxes(1, 2)

# change no. of MOs, no_occ, no_vir
#no_mo = no_mo * 2
#no_occ = no_occ * 2
no_vir = no_mo - no_occ
#eps = np.repeat(eps, 2)

print('no_occ: {} no_vir: {}'.format(no_occ, no_vir))

eps_occ = eps[:no_occ]
eps_vir = eps[no_occ:]
# note that occ.transpose(col) - vir(row) gives occ x vir matrix of differences
# needs F_occ and F_vir separate (will need F_vir for semi-canonical basis later)

# Make F MO basis
F = np.einsum('uj, vi, uv', C, C, F)
#F = np.repeat(F, 2, axis=0)
#F = np.repeat(F, 2, axis=1)

# Make F block diagonal
spin_ind = np.arange(F.shape[0], dtype=np.int) % 2
F *= (spin_ind.reshape(-1, 1) == spin_ind)

F_occ = F[:no_occ, :no_occ]
F_vir = F[no_occ:, no_occ:]

# init T1s
t_ia = np.zeros((no_occ, no_vir))

# init T2s
d_ia = eps_occ.reshape(-1, 1) - eps_vir
d_ijab = eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1, 1) - eps_vir.reshape(-1, 1) - eps_vir
t_ijab = MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
t_ijab /= d_ijab

# print('D_ia\'s : \n {} \n'.format(d_ia))
# print('Initial T2s: \n {}\n'.format(t_ijab))

# Make intermediates, Staunton:1991 eqns 3-11
# Spin-adapted, every TEI term is modified to include
# antisymmetrized term
def make_taut(t_ia, t_ijab):
    tau_t = t_ijab + 0.5 * (np.einsum('ia,jb->ijab', t_ia, t_ia))
    return tau_t


def make_tau(t_ia, t_ijab):
    tau = t_ijab + (np.einsum('ia,jb->ijab', t_ia, t_ia))
    return tau


def make_Fae(taut, t_ia, t_ijab):
    Fae = F_vir.copy()
    #Fae[np.diag_indices_from(Fae)] = 0
    Fae -= ndot('me,ma->ae', F[:no_occ, no_occ:], t_ia, prefactor=0.5)
    Fae += ndot('mf,mafe->ae', t_ia, MO[:no_occ, no_occ:, no_occ:, no_occ:], prefactor=2.0)
    Fae -= ndot('mf,maef->ae', t_ia, MO[:no_occ, no_occ:, no_occ:, no_occ:])
    Fae -= ndot('mnaf,mnef->ae', taut, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=2.0)
    Fae += ndot('mnaf,mnfe->ae', taut, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    return Fae


def make_Fmi(taut, t_ia, t_ijab):
    Fmi = F_occ.copy()
    #Fmi[np.diag_indices_from(Fmi)] = 0
    Fmi += ndot('ie,me->mi', t_ia, F[:no_occ, no_occ:], prefactor=0.5)
    Fmi += ndot('ne,mnie->mi', t_ia, MO[:no_occ, :no_occ, :no_occ, no_occ:], prefactor=2.0)
    Fmi -= ndot('ne,mnei->mi', t_ia, MO[:no_occ, :no_occ, no_occ:, :no_occ])
    Fmi += ndot('inef,mnef->mi', taut, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=2.0)
    Fmi -= ndot('inef,mnfe->mi', taut, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    return Fmi


def make_Fme(t_ia, t_ijab):
    Fme = F[:no_occ, no_occ:].copy()
    Fme += ndot('nf,mnef->me', t_ia, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor = 2.0)
    Fme -= ndot('nf,mnfe->me', t_ia, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    return Fme


def make_Wmnij(tau, t_ia, t_ijab):
    Wmnij = MO[:no_occ, :no_occ, :no_occ, :no_occ].copy()
    Wmnij += ndot('je,mnie->mnij', t_ia, MO[:no_occ, :no_occ, :no_occ, no_occ:])
    Wmnij += ndot('ie,mnej->mnij', t_ia, MO[:no_occ, :no_occ, no_occ:, :no_occ])
    Wmnij += ndot('ijef,mnef->mnij', tau, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    return Wmnij


#def make_Wabef(tau, t_ia, t_ijab):
#    Wabef = MO[no_occ:, no_occ:, no_occ:, no_occ:].copy()
#    Wabef -= ndot('mb,amef->abef', t_ia, MO[no_occ:, :no_occ, no_occ:, no_occ:])
#    Wabef += ndot('ma,bmef->abef', t_ia, MO[no_occ:, :no_occ, no_occ:, no_occ:])
#    Wabef += ndot('mnab,mnef->abef', tau, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=0.25)
#    return Wabef


def make_Wmbej(t_ia, t_ijab):
    Wmbej = MO[:no_occ, no_occ:, no_occ:, :no_occ].copy()
    Wmbej += ndot('jf,mbef->mbej', t_ia, MO[:no_occ, no_occ:, no_occ:, no_occ:])
    Wmbej -= ndot('nb,mnej->mbej', t_ia, MO[:no_occ, :no_occ, no_occ:, :no_occ])
    tmp = 0.5 * t_ijab.copy() + np.einsum('jf,nb->jnfb', t_ia, t_ia)
    Wmbej -= ndot('jnfb,mnef->mbej', tmp, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    Wmbej += ndot('njfb,mnef->mbej', t_ijab, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    Wmbej -= ndot('njfb,mnfe->mbej', t_ijab, MO[:no_occ, :no_occ, no_occ:, no_occ:], prefactor=0.5)
    return Wmbej


def make_Wmbje(t_ia, t_ijab):
    Wmbje = -1.0 * MO[:no_occ, no_occ:, :no_occ, no_occ:].copy()
    Wmbje -= ndot('jf,mbfe->mbje', t_ia, MO[:no_occ, no_occ:, no_occ:, no_occ:])
    Wmbje += ndot('nb,mnje->mbje', t_ia, MO[:no_occ, :no_occ, :no_occ, no_occ:])
    tmp = 0.5 * t_ijab.copy() + np.einsum('jf,nb->jnfb', t_ia, t_ia)
    Wmbje += ndot('jnfb,mnfe->mbje', tmp, MO[:no_occ, :no_occ, no_occ:, no_occ:])
    return Wmbje


def make_Zmbij(tau):
    Zmbij = 0
    Zmbij += ndot('mbef,ijef->mbij', MO[:no_occ, no_occ:, no_occ:, no_occ:], tau)
    return Zmbij


# Update T1 and T2 amplitudes
def update_ts(tau, tau_t, t_ia, t_ijab):

    # Build intermediates
    Fae = make_Fae(tau_t, t_ia, t_ijab)
    Fmi = make_Fmi(tau_t, t_ia, t_ijab)
    Fme = make_Fme(t_ia, t_ijab)

    Wmnij = make_Wmnij(tau, t_ia, t_ijab)
    #Wabef = make_Wabef(tau, t_ia, t_ijab)
    Wmbej = make_Wmbej(t_ia, t_ijab)
    Wmbje = make_Wmbje(t_ia, t_ijab)
    Zmbij = make_Zmbij(tau)

    # Create residual T1s
    Ria = F[:no_occ, no_occ:].copy()
    Ria += ndot('ie,ae->ia', t_ia, Fae)
    Ria -= ndot('ma,mi->ia', t_ia, Fmi)
    Ria += ndot('imae,me->ia', t_ijab, Fme, prefactor=2.0)
    Ria -= ndot('imea,me->ia', t_ijab, Fme)
    Ria -= ndot('nf,naif->ia', t_ia, MO[:no_occ, no_occ:, :no_occ, no_occ:])
    Ria += ndot('nf,nafi->ia', t_ia, MO[:no_occ, no_occ:, no_occ:, :no_occ], prefactor=2.0)
    Ria += ndot('mief,maef->ia', t_ijab, MO[:no_occ, no_occ:, no_occ:, no_occ:], prefactor=2.0)
    Ria -= ndot('mife,maef->ia', t_ijab, MO[:no_occ, no_occ:, no_occ:, no_occ:])
    Ria -= ndot('mnae,nmei->ia', t_ijab, MO[:no_occ, :no_occ, no_occ:, :no_occ], prefactor=2.0)
    Ria += ndot('mnae,nmie->ia', t_ijab, MO[:no_occ, :no_occ, :no_occ, no_occ:])

    # Create residual T2s
    Rijab = MO[:no_occ, :no_occ, no_occ:, no_occ:].copy()
    # Term 2
    tmp = ndot('ijae,be->ijab', t_ijab, Fae)
    Rijab += tmp 
    Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = 0.5 * np.einsum('ijae,mb,me->ijab', t_ijab, t_ia, Fme)
    Rijab -= tmp
    Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
    # Term 3
    tmp = ndot('imab,mj->ijab', t_ijab, Fmi)
    Rijab -= tmp
    Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = 0.5 * np.einsum('imab,je,me->ijab', t_ijab, t_ia, Fme)
    Rijab -= tmp 
    Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
    # Term 4
    Rijab += ndot('mnab,mnij->ijab', tau, Wmnij)
    # Term 5
    Rijab += ndot('ijef,abef->ijab', tau, MO[no_occ:, no_occ:, no_occ:, no_occ:])
    # Extra term since Wabef is not formed
    tmp = ndot('ma,mbij->ijab', t_ia, Zmbij)
    Rijab -= tmp
    Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)
    # Term 6 # 1
    tmp = ndot('imae,mbej->ijab', t_ijab, Wmbej)
    tmp -= ndot('imea,mbej->ijab', t_ijab, Wmbej)
    Rijab += tmp
    Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
    tmp1 = np.einsum('ie,ma,mbej->ijab', t_ia, t_ia, MO[:no_occ, no_occ:, no_occ:, :no_occ])
    Rijab -= tmp1
    Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
    # Term 6 # 2
    tmp = ndot('imae,mbej->ijab', t_ijab, Wmbej)
    tmp += ndot('imae,mbje->ijab', t_ijab, Wmbje)
    Rijab += tmp
    Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
    # Term 6 # 3
    tmp = ndot('mjae,mbie->ijab', t_ijab, Wmbje)
    Rijab += tmp
    Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
    tmp1 = np.einsum('ie,mb,maje->ijab', t_ia, t_ia, MO[:no_occ, no_occ:, :no_occ, no_occ:])
    Rijab -= tmp1
    Rijab -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

    # Term 7
    tmp = ndot('ie,abej->ijab', t_ia, MO[no_occ:, no_occ:, no_occ:, :no_occ])
    Rijab += tmp
    Rijab += tmp.swapaxes(0, 1).swapaxes(2, 3)
    # Term 8
    tmp = ndot('ma,mbij->ijab', t_ia, MO[:no_occ, no_occ:, :no_occ, :no_occ])
    Rijab -= tmp
    Rijab -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    if local:
        # Transform Rs using Q
        R1Q = np.einsum('iiba,ib,iiba->ia', Q, Ria, Q)
        R2Q = np.einsum('ijac,ijab,ijbd->ijcd', Q, Rijab, Q)
        # Transform RQs using L
        R1QL = np.einsum('iiba,ib,iiba->ia', L, R1Q, L)
        R2QL = np.einsum('ijac,ijab,ijbd->ijcd', L, R2Q, L)
        # Use vir orb. energies from semicanonical
        d1_QL = np.zeros_like(t_ia)
        for i in range(no_occ):
            for a in range(no_vir):
                d1_QL[i, a] = F_occ[i, i] - eps_pno[i, i, a]
        d2_QL = np.zeros_like(t_ijab)
        for i in range(no_occ):
            for j in range(no_occ):
                for a in range(no_vir):
                    for b in range(no_vir):
                        d2_QL[i, j, a, b] = F_occ[i, i] + F_occ[j, j] - eps_pno[i, j, a] - eps_pno[i, j, b]
        #print('denom in semi-canonical PNO basis:\n{}\n'.format(d_QL.shape))
        T1QL = R1QL / d1_QL
        T2QL = R2QL / d2_QL
        # Back transform to TQs
        T1Q = np.einsum('iiab,ib,iiab->ia', L, T1QL, L)
        T2Q = np.einsum('ijca,ijab,ijdb->ijcd', L, T2QL, L)
        # Back transform to Ts
        new_tia = t_ia.copy()
        new_tia += np.einsum('iiab,ib,iiab->ia', Q, T1Q, Q)
        #new_tia = Ria / d_ia
        new_tijab = t_ijab.copy()
        new_tijab += np.einsum('ijca,ijab,ijdb->ijcd', Q, T2Q, Q)
    else:
        # Apply denominators
        new_tia =  t_ia + Ria / d_ia
        new_tijab = t_ijab + Rijab / d_ijab

    return new_tia, new_tijab


# Compute CCSD correlation energy
def corr_energy(t_ia, t_ijab):
    E_corr = ndot('ia,ia->', F[:no_occ, no_occ:], t_ia, prefactor=2.0)
    tmp_tau = make_tau(t_ia, t_ijab)
    E_corr += ndot('ijab,ijab->', MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau, prefactor=2.0)
    E_corr -= ndot('ijba,ijab->', MO[:no_occ, :no_occ, no_occ:, no_occ:], tmp_tau, prefactor=1.0)
    return E_corr

if local:
    # Initialize PNOs
    print('Local switch on. Initializing PNOs.')

    # Identify weak pairs using MP2 pair corr energy
    e_ij = 0.5 * np.einsum('ijab,ijab->ij', MO[:no_occ, :no_occ, no_occ:, no_occ:], t_ijab)
    mp2_e  = corr_energy(t_ia, t_ijab)
    print('MP2 correlation energy: {}\n'.format(mp2_e))
    #print('Pair corr energy matrix: {}'.format(np.diag(e_ij)))
    str_pair_list = abs(e_ij) > e_cut
    print('Strong pair matrix is symmetric: {}'.format(np.allclose(str_pair_list, str_pair_list.T)))
    # Create Tij and Ttij
    T_ij = t_ijab.copy()
    Tt_ij = 2 * T_ij - T_ij.transpose(0, 1, 3, 2)

    #print('T_ij for i = 0, j = 1:\n{}\nT_ij.T for i = 0, j = 1:\n{}\nTt_ij for i = 0, j = 1:\n{}'.format(T_ij[0, 1, :, :], T_ij[0, 1, :, :].T, Tt_ij[0, 1, :, :]))

    # Form pair densities
    D = np.einsum('ijab,ijcb->ijac', T_ij, Tt_ij) + np.einsum('ijba,ijbc->ijac', T_ij, Tt_ij)
    for i in range(no_occ):
        for j in range(no_occ):
            if i == j:
                continue
            else:
                D[i, j] *= 2.0
    #print("Density matrix [0, 0]: \n{}".format(D[0,0]))

    # Diagonalize pair densities to get PNOs (Q) and occ_nos
    occ_nos = np.zeros((no_occ, no_occ, no_vir))
    Q = np.zeros((no_occ, no_occ, no_vir, no_vir))
    for i in range(no_occ):
        for j in range(no_occ):
            occ_nos[i, j], Q[i, j] = np.linalg.eigh(D[i, j])

    #print('Occupation numbers [0, 1]:\n {}'.format(occ_nos[0, 1]))
    #print('Transformation matrix [0, 0] :\n{}\n'.format(Q[0, 0]))

    # Get semicanonical transforms
        # transform F_vir to PNO basis
        # Diagonalize F_pno, get L
        # save virtual orb. energies
    F_pno = np.einsum('ijpa,ab,ijbq->ijpq', Q.transpose(0, 1, 3, 2), F_vir, Q)
    eps_pno = np.zeros((no_occ, no_occ, no_vir))
    L = np.zeros_like(t_ijab)
    for i in range(no_occ):
        for j in range(no_occ):
            eps_pno[i, j], L[i, j] = np.linalg.eigh(F_pno[i, j])
    print('Q x L:\t\t {}\n'.format(Q[1, 0] @ L[1, 0]))

old_e = corr_energy(t_ia, t_ijab)
print('Iteration\t\t CCSD Correlation energy\t\tDifference\nMP2\t\t\t {}'.format(old_e))

# Iterate until convergence
for i in range(maxiter):
    tau_t = make_taut(t_ia, t_ijab)
    tau = make_tau(t_ia, t_ijab)
    new_tia, new_tijab = update_ts(tau, tau_t, t_ia, t_ijab)
    new_e = corr_energy(new_tia, new_tijab)
    print('{}\t\t\t {}\t\t\t{}'.format(i, new_e, abs(new_e - old_e)))
    if(abs(new_e - old_e) < E_conv):
        print('Convergence reached.\n CCSD Correlation energy: {}\n'.format(new_e))
        break
    t_ia = new_tia
    t_ijab = new_tijab
    old_e = new_e
