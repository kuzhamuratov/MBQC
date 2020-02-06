import numpy as np
from scipy.special import factorial

def infinite_squeezing(phi, n_photons):
    state = np.array([(-0.5 * np.exp(1j * phi)) ** (n // 2) * np.sqrt(factorial(n)) / factorial(n // 2) if n % 2 == 0 else 0.
                  for n in range(n_photons)])
    return state/np.sqrt(np.dot(state, state.conj().T))

def vacuum(n_photons):
    return np.eye(n_photons)[0]

def fock_state(number, n_photons):
    return np.eye(n_photons)[number]

def finite_squeezing(r, phi, n_photons):
    state = np.array([1/np.sqrt(np.cosh(r))*(-np.exp(1j*phi)*np.tanh(r))**(n//2)*np.sqrt(factorial(n))/2**(n//2)/factorial(n//2) if n%2==0 else 0.
            for n in range(n_photons)])
    return state/np.sqrt(np.dot(state, state.conj().T))

def random_state(n_photons):
    state = np.random.uniform(size=n_photons) + 1j*np.random.uniform(size=n_photons)
    return state/np.sqrt(state.dot(state.conj().T))
