import numpy as np
from scipy.linalg import expm
import math

def destroy_operator(batch_size, n_photons):
    matrix = np.zeros((batch_size, n_photons, n_photons))
    for i in range(1,n_photons):
        matrix[:,i-1,i] = np.sqrt(i)
    return matrix

def create_operator(batch_size, n_photons):
    return destroy_operator(batch_size, n_photons).conj().transpose(0,2,1)


def S_gate(r, phi, n_photons):
    """
    Squeezing gate
    """
    z = r*np.exp(phi*1j)
    batch_size = 1 if len(z.shape)==0 else z.shape[0]
    z = z.reshape(batch_size, 1, 1)
    eq = np.multiply(z.conj(), np.linalg.matrix_power(destroy_operator(batch_size, n_photons),2)) -\
         np.multiply(z, np.linalg.matrix_power(create_operator(batch_size, n_photons),2))
    return np.array([expm(0.5*eq[i]) for i in range(batch_size)])

def D_gate(r, phi, n_photons):
    """
    Displacement gate
    """
    alpha = r*np.exp(phi*1.j)
    batch_size = 1 if len(alpha.shape)==0 else alpha.shape[0]
    alpha = alpha.reshape(batch_size, 1, 1)
    eq = np.multiply(alpha, create_operator(batch_size, n_photons)) -\
         np.multiply(alpha.conj(), destroy_operator(batch_size, n_photons))
    return np.array([expm(eq[i]) for i in range(batch_size)])

def X_gate(x, n_photons):
    """
    Displacement gate on X axis
    """
    batch_size = 1 if len(x.shape)==0 else x.shape[0]
    x = x.reshape(batch_size, 1, 1)
    eq = np.multiply(x, create_operator(batch_size,n_photons) 
           - destroy_operator(batch_size, n_photons))/np.sqrt(2.)
    return np.array([expm(eq[i]) for i in range(batch_size)])

def Z_gate(p, n_photons):
    """
    momentum displacement gate
    """
    batch_size = 1 if len(p.shape)==0 else p.shape[0]
    p = p.reshape(batch_size, 1, 1)
    eq = np.multiply(p, destroy_operator(batch_size, n_photons) +
             create_operator(batch_size, n_photons))/np.sqrt(2.)
    return np.array([expm(1.j*eq[i]) for i in range(batch_size)])

def R_gate(theta, n_photons):
    """
    Rotation gate
    """
    batch_size = 1 if len(theta.shape)==0 else theta.shape[0]
    theta = theta.reshape(batch_size, 1, 1)
    number_operator = np.einsum('ijk, ikl-> ijl', create_operator(batch_size, n_photons), 
                      destroy_operator(batch_size, n_photons))
    eq = np.multiply(theta, number_operator)
    return np.array([expm(1.j*eq[i]) for i in range(batch_size)])

def P_gate(batch_size, n_photons):
    """
    Quadratic phase gate
    args:
    first batch_size !!!!!
    """
    eq = (destroy_operator(batch_size, n_photons) + create_operator(batch_size, n_photons))
    eq = np.linalg.matrix_power(eq, 2)/4.
    return np.array([expm(1.j*eq[i]) for i in range(batch_size)])

def V_gate(gamma, n_photons):
    """
    Qubic phase gate
    """
    batch_size = 1 if len(gamma.shape)==0 else gamma.shape[0]
    gamma = gamma.reshape(batch_size, 1, 1)
    eq = np.multiply(gamma, np.linalg.matrix_power(destroy_operator(batch_size, n_photons) +
                create_operator(batch_size, n_photons), 3))
    return np.array([expm(1.j*eq[i]/np.sqrt(8.)/3.) for i in range(batch_size)])

def N_phase_gate(gamma, degree, n_photons):
    """
    N-phase gate
    """
    batch_size = 1 if len(gamma.shape)==0 else gamma.shape[0]
    gamma = gamma.reshape(batch_size, 1, 1)
    eq = np.multiply(gamma, np.linalg.matrix_power(destroy_operator(batch_size, n_photons) +
          create_operator(batch_size, n_photons), degree))
    return np.array([expm(1.j/float(degree)*eq[i]) for i in range(batch_size)])

def F_gate(batch_size, n_photons):
    """
    Fourier gate
    args:
    first batch_size !!!!!
    """
    angles = np.zeros(batch_size)
    angles[:] = np.pi/2.
    angles = angles.reshape(batch_size, 1, 1)
    return R_gate(angles, n_photons)
