'''
DIIS Helper class definitions and function definitions
For DIIS extrapolation in CCSD
'''

import numpy as np

class HelperDIIS(object):
    def __init__(self, t_ia, t_ijab, max_diis):
        self.oldt1 = t_ia.copy()
        self.oldt2 = t_ijab.copy()
        self.x_t1 = [t_ia.copy()]
        self.x_t2 = [t_ijab.copy()]
        self.x_errs = []
        self.diis_size = 0
        self.max_diis = max_diis


    def update_err_list(self, t_ia, t_ijab):
        self.x_t1.append(t_ia.copy())
        self.x_t2.append(t_ijab.copy())

        x_err1 = (self.x_t1[-1] - self.oldt1).ravel()
        x_err2 = (self.x_t2[-1] - self.oldt2).ravel()
        x_err = np.concatenate((x_err1, x_err2))

        self.x_errs.append(x_err)
        #print("Size of x_t1: {} Size of x_t2: {} Size of x_errs: {}".format(len(self.x_t1),  len(self.x_t2), len(self.x_errs)))
        #print("Norm of x_err: {}".format(np.linalg.norm(x_err)))
        self.oldt1 = t_ia.copy()
        self.oldt2 = t_ijab.copy()

    def extrapolate(self, t_ia, t_ijab):
        print("Extrapolating...")
        
        # Check if diis size is too large, then delete first entry
        if len(self.x_t1) > self.max_diis:
            del self.x_t1[0]
            del self.x_t2[0]
            del self.x_errs[0]

        # Must set this after removing elements
        self.diis_size = len(self.x_t1) - 1

        B = np.ones((self.diis_size + 1, self.diis_size +1)) * -1
        B[-1, -1] = 0
        
        for i, xi in enumerate(self.x_errs):
            B[i, i] = np.dot(xi, xi)
            for j, xj in enumerate(self.x_errs):
                if i >= j: continue
                B[i, j] = np.dot(xi, xj)
                B[j, i] = B[i, j] 

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        C = np.zeros(self.diis_size + 1)
        C[-1] = -1

        X = np.linalg.solve(B, C)

        new_tia = np.zeros_like(self.oldt1)
        new_tijab = np.zeros_like(self.oldt2)
        for i in range(self.diis_size):
            new_tia += X[i] * self.x_t1[i+1]
            new_tijab += X[i] * self.x_t2[i+1]

        self.oldt1 = new_tia.copy()
        self.oldt2 = new_tijab.copy()

        return new_tia, new_tijab
