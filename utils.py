import scipy as sp
import numpy as np
import math

def wigner_iterative(rho, xvec, yvec, g=np.sqrt(2)):

    """

      Wigner function for a state vector or density matrix at points
    `xvec + i * yvec`.
 
    Parameters
    ----------
 
    rho : np.array
        A state vector or density matrix.
 
    xvec : array_like
        x-coordinates at which to calculate the Wigner function.
 
    yvec : array_like
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.
 
    g : float
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.
 
    Returns
    -------
 
    w : array
        Values representing the Wigner function calculated over the specified
        range [xvec,yvec].
 
    Notes
    -----
    Using an iterative method to evaluate the wigner functions for the Fock
    state :math:`|m><n|`.
 
    The Wigner function is calculated as
    :math:`w = \sum_{mn} \\rho_{mn} w_{mn}` where :math:`w_{mn}` is the Wigner
    function for the density matrix :math:`|m><n|`.
 
    In this implementation, for each row m, w_list contains the Wigner functions
    w_list = [0, ..., w_mm, ..., w_mN]. As soon as one w_mn Wigner function is
    calculated, the corresponding contribution is added to the total Wigner
    function, weighted by the corresponding element in the density matrix
    :math:`rho_{mn}`.
    """
 
    m_max = int(np.prod(rho.shape[0]))
    x, y = sp.meshgrid(xvec, yvec)
    a = 0.5 * g * (x + 1.0j * y)
 
    w_list = np.array([np.zeros(np.shape(a), dtype=complex) for k in range(m_max)])
    w_list[0] = np.exp(-2.0 * abs(a) ** 2) / np.pi
 
    w = np.real(rho[0, 0]) * np.real(w_list[0])
    for n in range(1, m_max):
        w_list[n] = (2.0 * a * w_list[n - 1]) / np.sqrt(n)
        w += 2 * np.real(rho[0, n] * w_list[n])
 
    for m in range(1, m_max):
        temp = sp.copy(w_list[m])
        w_list[m] = (2 * np.conj(a) * temp - np.sqrt(m) * w_list[m - 1]) / np.sqrt(m)
 
        # w_list[m] = Wigner function for |m><m|
        w += sp.real(rho[m, m] * w_list[m])
 
        for n in range(m + 1, m_max):
            temp2 = (2 * a * w_list[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = sp.copy(w_list[n])
            w_list[n] = temp2
 
            # w_list[n] = Wigner function for |m><n|
            w += 2 * sp.real(rho[m, n] * w_list[n])
 
    return 0.5 * w * g ** 2


def fidelity(rho,rho_target):
    """Computes fidelity between two states represented by density matrices.

    Parameters
    ----------
    rho, rho_target : np.array
        Density matrices.

    Returns
    -------
    fidelity : float
    """
    
    sqrt_target = sp.linalg.sqrtm(rho_target)
    
    return np.abs(np.einsum('ii->',sp.linalg.sqrtm(sqrt_target @ rho @ sqrt_target))**2)

def ptrace(state, sel, cutoff_n):
    """Partial trace of the density matrix with selected components remaining.
    Parameters
    ----------
    state : Density matrix of a multimode quantum state.
    
    sel : int/list
        An ``int`` or ``list`` of components to keep after partial trace.
    
    cutoff_n : int
        Photon cutoff number
        
    Returns
    -------
    rhomat : Density matrix representing partial trace with selected components
        remaining.
    """
    
    rd = np.asarray([cutoff_n+1]*int(math.log(state.shape[0],cutoff_n+1)+0.1))
    nd = len(rd)
    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)
    sel = list(np.sort(sel))
    dkeep = (rd[sel]).tolist()
    qtrace = list(set(np.arange(nd)) - set(sel))
    dtrace = (rd[qtrace]).tolist()
    rd = list(rd)
    
    rhomat = np.trace(state.reshape(rd + rd) \
                          .transpose(qtrace + [nd + q for q in qtrace] +
                                     sel + [nd + q for q in sel]) \
                          .reshape([np.prod(dtrace),
                                    np.prod(dtrace),
                                    np.prod(dkeep),
                                    np.prod(dkeep)]))
    
    return rhomat

def calc_squeezing_after_loss(eta, last_sq=np.inf):
    return -0.5*np.log(eta*np.exp(-2*last_sq) + 0.5*(1-eta))
