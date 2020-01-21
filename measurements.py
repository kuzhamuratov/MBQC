import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial
from gates import R_gate



def proba_1d(state, x_pret, n_photons):
    res = 0
    matrix = np.eye(n_photons)
    for i in range(n_photons):
        part = state[i]*hermval( x_pret, matrix[i])*1./np.sqrt(factorial(i))*2**(-i/2.)/np.pi**0.25*np.exp(-x_pret**2/2.)
        res += np.abs(part*part.conj())
    return res


def proba_2d(state, x_pret, y_pret, n_photons):
    matrix_factorial = np.zeros((n_photons, n_photons))
    for i in range(n_photons):
        for j in range(n_photons):
                matrix_factorial[i, j] = 1./np.sqrt(factorial(i)*factorial(j))*2**(-(i+j)/2.)
    res = 0
    matrix = np.eye(n_photons)
    for i in range(n_photons):
        for j in range(n_photons):
            part = (state[i*n_photons+j]*hermval( x_pret, matrix[i])*hermval(y_pret, matrix[j])*matrix_factorial[i, j])
            res += np.abs(part*part.conj())
    return res*1./np.pi*np.exp(-(x_pret**2+y_pret**2))

def proba_3d(state, x_pret, y_pret, z_pret, n_photons):
    matrix_factorial = np.zeros((n_photons, n_photons, n_photons))
    for i in range(n_photons):
        for j in range(n_photons):
            for k in range(n_photons):
                matrix_factorial[i, j, k] = 1./np.sqrt(factorial(i)*factorial(j)*factorial(k))*2**(-(i+j+k)/2.)
    res = 0
    matrix = np.eye(n_photons)
    for i in range(n_photons):
        for j in range(n_photons):
            for k in range(n_photons):
                part = (state[i*n_photons**2+j*n_photons+k]*hermval( x_pret, matrix[i])*hermval(y_pret, matrix[j])*
                hermval(z_pret, matrix[k])*matrix_factorial[i, j, k])
                res += np.abs(part*part.conj())
    return res*1./np.pi**1.5*np.exp(-(x_pret**2+y_pret**2+z_pret**2))

def bootstrap(state, n_modes, n_photons, instances_size):
    step = 0
    cur_length = 0
    while cur_length <= instances_size:
        x_pret = np.random.uniform(-5, 5, size=10000)
        p = np.random.uniform(size=10000)
        if n_modes==1: 
            probabilities = proba_1d(state, x_pret, n_photons)
        elif n_modes==2: 
            y_pret = np.random.uniform(-5, 5, size=10000)
            probabilities = proba_2d(state, x_pret, y_pret, n_photons)
        elif n_modes==3: 
            y_pret = np.random.uniform(-5, 5, size=10000)
            z_pret = np.random.uniform(-5, 5, size=10000)
            probabilities = proba_3d(state, x_pret, y_pret, z_pret, n_photons)
        eq = probabilities>=p
        length = x_pret[eq].shape[0]
        cur_length += length
        if n_modes==1: arr = np.asarray(x_pret[eq])
        elif n_modes==2: arr = np.concatenate([x_pret[eq].reshape(length,1),y_pret[probabilities>=p].reshape(length,1)],axis=1)
        elif n_modes==3: arr = np.concatenate([x_pret[eq].reshape(length,1),y_pret[probabilities>=p].reshape(length,1),
                         z_pret[eq].reshape(length,1)],axis=1)
        if step==0:
            data = arr
            step = 1
        else:
            data = np.concatenate([data,arr], axis=0)
    return data[:instances_size]



def meas_X(state, n_modes, n_photons):
    """
    Measurement on X quadrature
    """
    return meas_X_theta(state, 0., n_modes, n_photons)

def meas_P(state, n_modes, n_photons):
    """
    Measurement on P quadrature
    """
    return meas_X_theta(state, np.pi/2., n_modes, n_photons)

def meas_X_theta(state, theta, n_modes, n_photons, x_max=5, 
                 x_min=-5, instances_size=10_000):
    """
    Measurement on X_theta quadrature
    state is pure state
    """

    # rotate to theta angle
    for i in range(n_modes):
        rotation = R_gate(theta, i, n_modes, n_photons)
        state_ = rotation @ state @ rotation.conj().T
    
    if n_modes>3:
        raise NotImplementedError
    else:
        return bootstrap(state_, n_modes, n_photons, instances_size)
   