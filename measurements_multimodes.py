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

def meas_X_theta(state, theta, n_photons, x_max=5, 
                 x_min=-5, num_bins=10_000, instances_size=10_000):
    """
    Measurement on X_theta quadrature
    """
    rotation = R_gate(-theta, n_photons)
    state_ = np.einsum('jk, k-> j', rotation, state)
    x = np.linspace(x_min, x_max, num_bins+1)
    new_state = calc_coef(state_, n_photons)
    ampl = 1./np.pi**0.25*np.exp(-x**2/2.)*hermval(x, new_state)
    probs = np.abs(ampl*np.conj(ampl))
    probs /= np.sum(probs)
    return np.random.choice(x, size=instances_size, p=probs)