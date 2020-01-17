import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial
from gates_batch import R_gate
def calc_coef(state, n_photons):
    for i in range(n_photons):
        state[:, i] = state[:, i]*1./np.sqrt(factorial(i))*2**(-i/2.)
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
                 x_min=-20, num_bins=10_000, instances_size=200_000):
    """
    Measurement on X_theta quadrature
    """
    batch_size = state.shape[0]
    theta = theta*np.ones(batch_size)
    rotation = R_gate(-theta, n_photons)
    state = np.einsum('ijk, ik -> ij', rotation, state)
    x = np.linspace(x_min, x_max, num_bins+1)
    new_state = calc_coef(state, n_photons)
    new_state = new_state.transpose(1,0)
    ampl = 1./np.pi**0.25*np.exp(-x**2/2.)*hermval(x, new_state)
    probs = np.abs(ampl*np.conj(ampl))
    probs /= np.sum(probs, axis=1).reshape(batch_size,1)
    data = np.zeros((batch_size, instances_size))
    for i in range(batch_size):
        data[i] = np.random.choice(x, size=instances_size, p=probs[i,:])
    return data
