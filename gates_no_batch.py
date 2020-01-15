import numpy as np
from scipy.linalg import expm
import math

def destroy_operator(n_photons):
    matrix = np.zeros((n_photons, n_photons))
    for i in range(1,n_photons):
        matrix[i-1,i] = np.sqrt(i)
    return matrix

def create_operator(n_photons):
    return destroy_operator(n_photons).conj().transpose(1,0)

def S_gate(r, phi, n_photons):
    """
    Squeezing gate
    """
    z = r*np.exp(phi*1j)
    eq = z.conj()*np.linalg.matrix_power(destroy_operator(n_photons),2) -\
        z*np.linalg.matrix_power(create_operator(n_photons),2)
    return expm(0.5*eq)

def D_gate(r, phi, n_photons):
    """
    Displacement gate
    """
    alpha = r*np.exp(phi*1.j)
    eq = alpha*create_operator(n_photons) -\
        alpha.conj()*destroy_operator(n_photons)
    return expm(eq)

def X_gate(x, n_photons):
    """
    Displacement gate on X axis
    """
    eq = (create_operator(n_photons) - destroy_operator(n_photons))/np.sqrt(2.)
    return expm(x*eq)

def Z_gate(p, n_photons):
    """
    momentum displacement gate
    """
    eq = (destroy_operator(n_photons) +
             create_operator(n_photons))/np.sqrt(2.)
    return expm(1.j*p*eq)

def R_gate(theta, n_photons):
    """
    Rotation gate
    """
    number_operator = np.einsum('jk, kl-> jl', create_operator(n_photons), 
                      destroy_operator(n_photons))
    eq = theta*number_operator
    return expm(1.j*eq)

def P_gate(n_photons):
    """
    Quadratic phase gate
    args:
    """
    eq = destroy_operator(n_photons) + create_operator(n_photons)
    eq = np.linalg.matrix_power(eq, 2)/4.
    return expm(1.j*eq)

def V_gate(gamma, n_photons):
    """
    Qubic phase gate
    """
    eq = gamma*np.linalg.matrix_power(destroy_operator(n_photons) +
                create_operator(n_photons), 3)
    return expm(1.j*eq/np.sqrt(8.)/3.)

def N_phase_gate(gamma, degree, n_photons):
    """
    N-phase gate
    """
    eq = gamma*np.linalg.matrix_power(destroy_operator(n_photons) +
          create_operator(n_photons), degree)
    return expm(1.j/float(degree)*eq)

def F_gate(n_photons):
    """
    Fourier gate
    args:
    """
    return R_gate(np.pi/2., n_photons)
