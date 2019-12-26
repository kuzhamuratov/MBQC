import numpy as np
from scipy.linalg import expm
import math

def destroy_operator(n_photons):
    matrix = np.zeros((n_photons, n_photons))
    for i in range(1,n_photons):
        matrix[i-1,i] = np.sqrt(i)
    return matrix

def create_operator(n_photons):
    return destroy_operator(n_photons).conj().T


def S_gate(r, phi, n_photons):
    """
    Squeezing gate
    """
    z = r*np.exp(phi*1j)
    eq = z.conj()*np.linalg.matrix_power(destroy_operator(n_photons),2) - z*np.linalg.matrix_power(create_operator(n_photons),2)
    return expm(0.5*eq)

def D_gate(r, phi, n_photons):
    """
    Displacement gate
    """
    alpha = r*np.exp(phi*1.j)
    eq = alpha*create_operator(n_photons) - alpha.conj()*destroy_operator(n_photons)
    return expm(eq)

def X_gate(x, n_photons):
    """
    Displacement gate
    """
    return expm((x*create_operator(n_photons) - x*destroy_operator(n_photons))/np.sqrt(2.))

def Z_gate(p, n_photons):
    """
    momentum displacement gate
    """
    return expm(1.j*p*(destroy_operator(n_photons) + create_operator(n_photons))/np.sqrt(2.))

def R_gate(theta, n_photons):
    """
    Rotation gate
    """
    return expm(1.j*theta*create_operator(n_photons)*destroy_operator(n_photons))

def P_gate(s, n_photons):
    """
    Quadratic phase gate
    """
    return expm(1.j*(destroy_operator(n_photons) + create_operator(n_photons))/4.)

def V_gate(gamma, n_photons):
    """
    Qubic phase gate
    """
    return expm(1.j*gamma/3.*(destroy_operator(n_photons) + create_operator(n_photons))**3)

def N_phase_gate(gamma, degree, n_photons):
    """
    N-phase gate
    """
    return expm(1.j*gamma/float(degree)*(destroy_operator(n_photons) + create_operator(n_photons))**degree)

def F_gate(n_photons):
    """
    Fourier gate
    """
    return R_gate(np.pi/2., n_photons)
