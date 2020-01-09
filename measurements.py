import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial
from gates import R_gate
def calc_coef(state, n_photons):
    for i in range(n_photons):
        state[i] = state[i]*1./np.sqrt(factorial(i))*2**(-i/2.)
    return state

def meas_X(state, n_photons):
    """
    Measurement on X quadrature
    """
    return meas_X_theta(state, 0., n_photons)

def meas_P(state, n_photons):
    """
    Measurement on P quadrature
    """
    return meas_X_theta(state, np.pi/2., n_photons)

def meas_X_theta(state, theta, n_photons, x_max=20, 
                 x_min=-20, num_bins=10000, instances_size=100000):
    """
    Measurement on X_theta quadrature
    """
    rotation = R_gate(-theta, n_photons)
    state = rotation.dot(state)
    x = [x_min + (x_max - x_min)/(num_bins + 1)*i for i in range(num_bins+1)]
    probs = []
    new_state = calc_coef(state, n_photons)
    for x_inst in x:
        prob_x_inst = 1./np.pi**0.25*np.exp(-x_inst**2/2.)*hermval(x_inst, new_state)
        probs.append(np.abs(prob_x_inst*np.conj(prob_x_inst)))
    probs = probs/np.sum(probs)
    return np.random.choice(x, size=instances_size, p=probs)
