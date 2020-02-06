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

def tensor_product(operation, numb_mode, n_modes, n_photons):
    for i in range(1,n_modes+1):
        if i==1:
            if i==numb_mode:
                matrix = operation
            else:
                matrix = np.eye(n_photons)
        elif i==numb_mode:
            matrix = np.kron(operation, matrix)
        else:
            matrix = np.kron(np.eye(n_photons), matrix)
    return matrix


# two mode gates 

def BS_gate(theta, phi, mode1, mode2, n_modes, n_photons):
    """
     beamsplitter from libs
    """
    op1 = tensor_product(destroy_operator(n_photons), mode1, n_modes, n_photons)
    op2 = tensor_product(destroy_operator(n_photons), mode2, n_modes, n_photons)
    return expm(theta * (np.exp(1j * phi) * op1 @ op2.conj().T - np.exp(-1j * phi) * op1.conj().T @ op2))

def CZ_gate(s, mode1, mode2, n_modes, n_photons):
    """
    Controlled-Z gate
    """
    operator = (destroy_operator(n_photons) + create_operator(n_photons))/np.sqrt(2.)
    op1 = tensor_product(operator, mode1, n_modes, n_photons)
    op2 = tensor_product(operator, mode2, n_modes, n_photons)
    return expm(1j*s*op1 @ op2)

def CX_gate(s, mode1, mode2, n_modes, n_photons):
    """
    Controlled-Z gate
    """
    operator_x = (destroy_operator(n_photons) + create_operator(n_photons))/np.sqrt(2.)
    operator_p = (destroy_operator(n_photons) - create_operator(n_photons))/np.sqrt(2.)
    op1 = tensor_product(operator_x, mode1, n_modes, n_photons)
    op2 = tensor_product(operator_p, mode2, n_modes, n_photons)
    return expm(s*op1 @ op2)

def S_gate(r, phi, numb_mode, n_modes, n_photons):
    """
    Squeezing gate
    """
    z = r*np.exp(phi*1j)
    eq = z.conj()*np.linalg.matrix_power(destroy_operator(n_photons),2) -\
        z*np.linalg.matrix_power(create_operator(n_photons),2)
    return tensor_product(expm(0.5*eq), numb_mode, n_modes, n_photons)

def D_gate(r, phi, numb_mode, n_modes, n_photons):
    """
    Displacement gate
    """
    alpha = r*np.exp(phi*1.j)
    eq = alpha*create_operator(n_photons) -\
        alpha.conj()*destroy_operator(n_photons)
    return tensor_product(expm(eq), numb_mode, n_modes, n_photons)

def X_gate(x, numb_mode, n_modes, n_photons):
    """
    Displacement gate on X axis
    """
    eq = (create_operator(n_photons) - destroy_operator(n_photons))/np.sqrt(2.)
    return tensor_product(expm(x*eq), numb_mode, n_modes, n_photons)

def Z_gate(p, numb_mode, n_modes, n_photons):
    """
    momentum displacement gate
    """
    eq = (destroy_operator(n_photons) +
             create_operator(n_photons))/np.sqrt(2.)
    return tensor_product(expm(1.j*p*eq), numb_mode, n_modes, n_photons)

def R_gate(theta, numb_mode, n_modes, n_photons):
    """
    Rotation gate
    """
    number_operator = np.einsum('jk, kl-> jl', create_operator(n_photons), 
                      destroy_operator(n_photons))
    eq = theta*number_operator
    return tensor_product(expm(1.j*eq), numb_mode, n_modes, n_photons)

def P_gate(numb_mode, n_modes, n_photons):
    """
    Quadratic phase gate
    args:
    """
    eq = destroy_operator(n_photons) + create_operator(n_photons)
    eq = np.linalg.matrix_power(eq, 2)/4.
    return tensor_product(expm(1.j*eq), numb_mode, n_modes, n_photons)

def V_gate(gamma, numb_mode, n_modes, n_photons):
    """
    Qubic phase gate
    """
    eq = gamma*np.linalg.matrix_power(destroy_operator(n_photons) +
                create_operator(n_photons), 3)
    return tensor_product(expm(1.j*eq/np.sqrt(8.)/3.), numb_mode, n_modes, n_photons)

def N_phase_gate(gamma, degree, numb_mode, n_modes, n_photons):
    """
    N-phase gate
    """
    eq = gamma*np.linalg.matrix_power(destroy_operator(n_photons) +
          create_operator(n_photons), degree)
    return tensor_product(expm(1.j/float(degree)*eq), numb_mode, n_modes, n_photons)

def F_gate(numb_mode, n_modes, n_photons):
    """
    Fourier gate
    args:
    """
    return R_gate(np.pi/2., numb_mode, n_modes, n_photons)
