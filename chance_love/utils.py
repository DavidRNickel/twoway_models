# -*- coding: utf-8 -*-
#
# Utility functions for CL scheme.
#

import numpy as np
from scipy.linalg import toeplitz
from sympy import Symbol, Poly, expand
import matplotlib.pyplot as plt
import matplotlib
from math import pi


#
# Helper function used for calc_..._opt
def _get_min_pos_root(coefs):
    roots = np.roots(coefs)
    opt = roots[np.isreal(roots)]
    opt = opt[opt>0]
    if opt.size>1:
       return opt.min()
   
    elif opt.size==0:
        return 0
    
    else:
        return opt.item()

#
# Lemma 5
def calc_beta_opt(s_2,n,g,r):
    b = Symbol('b')
    coefs = Poly(expand(b**(2*n) \
                        - (n+(1+s_2)*n*g*r)*(b**2) \
                        + n-1,b)).all_coeffs()

    return _get_min_pos_root(coefs).real
    
#
# Lemma 6
def calc_gamma_opt(s_2,r,n):
    a = s_2
    b = r*(1+s_2)

    if n < (1 + 1/b):
        return 0
    
    g = Symbol('g')
    coefs = Poly(expand(a*((1+b*g)**n) - n*b*(1-g) + b+1,g)).all_coeffs()
    
    return _get_min_pos_root(coefs).real
    

def calc_q(g, F, s_2, n):
    C_inv = np.linalg.inv(((np.eye(n)+F)@(np.eye(n)+F).T + s_2*F@F.T))
    return (g.T@C_inv).T / (g.T@C_inv@g) # throw in the transpose on the top for consistent vector shaping

#   
# Lemma 5
def calc_F(s_2, b, n):
    lead_coef = -(1-b**2)/(b*(1+s_2))
    vec = [0]
    vec += [b**nn for nn in range(n-1)]
    return lead_coef*toeplitz(vec,np.zeros(n))

#
# convert linear to decibel
def to_db(x):
    return 10*np.log10(x)


# #
# # Lemma 5 (This one wasn't unbiased, so I have no clue why it's in the paper)
# def calc_q(b,n):
#     return np.sqrt((1-b**2)/(1-b**(2*n)))*np.array([b**nn for nn in range(n)]).reshape(-1,1)