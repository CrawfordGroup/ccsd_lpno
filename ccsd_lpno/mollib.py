'''
Molecule library to enable easier testing
Works with current Psi4
'''

molecule_hof = """
          O          -0.947809457408    -0.132934425181     0.000000000000
          H          -1.513924046286     1.610489987673     0.000000000000
          F           0.878279174340     0.026485523618     0.000000000000
unit bohr
noreorient
"""

molecule_h2o2 = """
     O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
     O      0.028962160801     0.694396279686    -0.049338350190                                                                  
     H      0.350498145881    -0.910645626300     0.783035421467                                                                  
     H     -0.350498145881     0.910645626300     0.783035421467                                                                  
    symmetry c1        
    """

mollib = {}
mollib['hof'] = molecule_hof
mollib['h2o2'] = molecule_h2o2
