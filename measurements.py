import numpy as np
import itertools
from numpy.polynomial.hermite import hermval
from scipy.special import factorial
from .gates import R_gate, BS_gate, D_gate, tensor_product
from .states import infinite_squeezing, fock_state, vacuum
from .utils import ptrace

# For rho matrix
## helping functions

def reshaping_rho(rho, n_modes, n_photons, reverse=True):
    '''
    from shape i (n_photons**n_modes, n_photons**n_modes)
    to (n_photons, n_photons,....,n_photons, n_photons...) shape j
    args: rho - density matrix in shape i
          n_modes - number of modes in input_state
          n_photons - dimension of truncuation
          reverse if True: i = 1 j = 2
                  if False: i = 2 j = 1
    output: rho - density matrix in shape j
    '''
    size = tuple([n_photons]*2*n_modes) if reverse else (n_photons**n_modes, n_photons**n_modes)
    rho_new = np.zeros(size, dtype=np.complex128)
    for permutation in itertools.product([i for i in range(n_photons)], repeat=2*n_modes):
        val1 = sum(map(lambda x: x[1]*n_photons**(n_modes-x[0]-1), enumerate(permutation[:n_modes])))
        val2 = sum(map(lambda x: x[1]*n_photons**(n_modes-x[0]-1), enumerate(permutation[n_modes:])))
        if reverse:
            rho_new[permutation] = rho[val1, val2]
        else:
            rho_new[val1, val2] = rho[permutation]
    return rho_new


def proba_rho(rho, x_pret, n_modes, n_photons):
    """
    gives probabilities to sample x_pret
    args: rho - density matrix in (n_photons, n_photons,....,n_photons, n_photons...) shape
          x_pret - quadrature sample (size, n_modes)
          n_modes - number of modes
          n_photons - photon number
    output: probabilities to sample x_pret
    """
    matrix = np.eye(n_photons)
    size = tuple([n_photons]*2*n_modes)
    matrix_indexex  = np.zeros(size)
    
    for permutation in itertools.product([i for i in range(n_photons)], repeat=2*n_modes):
        val = np.prod(list(map(lambda x: 1./np.sqrt(factorial(x))*2**(-x/2.), permutation))) 
        matrix_indexex[permutation] = val
        
    if n_modes==1:
        part = np.einsum('ij, il, jl, ij->l', rho, 
                         hermval(x_pret[:,0], matrix),hermval(x_pret[:,0], matrix), 
                         matrix_indexex)
    elif n_modes==2:
        part = np.einsum('imjn, il, jl, ml, nl, imjn->l', rho, 
                         hermval(x_pret[:,0], matrix), hermval(x_pret[:,0], matrix), 
                         hermval(x_pret[:,1], matrix), hermval(x_pret[:,1], matrix),
                         matrix_indexex)
    elif n_modes==3:
        part = np.einsum('impjnq, il, jl, ml, nl, pl, ql, impjnq->l', rho, 
                         hermval(x_pret[:,0], matrix), hermval(x_pret[:,0], matrix), 
                         hermval(x_pret[:,1], matrix), hermval(x_pret[:,1], matrix),
                         hermval(x_pret[:,2], matrix), hermval(x_pret[:,2], matrix),
                         matrix_indexex)
    elif n_modes==4:
        part = np.einsum('impdjnqe, il, jl, ml, nl, pl, ql, dl, el, impdjnqe -> l', rho, 
                         hermval(x_pret[:,0], matrix), hermval(x_pret[:,0], matrix), 
                         hermval(x_pret[:,1], matrix), hermval(x_pret[:,1], matrix),
                         hermval(x_pret[:,2], matrix), hermval(x_pret[:,2], matrix),
                         hermval(x_pret[:,3], matrix), hermval(x_pret[:,3], matrix),
                         matrix_indexex)
    elif n_modes==5:
        part = np.einsum('impdbjnqec, il, jl, ml, nl, pl, ql, dl, el, bl, cl, impdbjnqec -> l', rho, 
                         hermval(x_pret[:,0], matrix), hermval(x_pret[:,0], matrix), 
                         hermval(x_pret[:,1], matrix), hermval(x_pret[:,1], matrix),
                         hermval(x_pret[:,2], matrix), hermval(x_pret[:,2], matrix),
                         hermval(x_pret[:,3], matrix), hermval(x_pret[:,3], matrix),
                         hermval(x_pret[:,4], matrix), hermval(x_pret[:,4], matrix),
                         matrix_indexex)
    elif n_modes==6:
        part = np.einsum('impdbfjnqecz, il, jl, ml, nl, pl, ql, dl, el, bl, cl, fl, zl, impdbfjnqecz -> l', rho, 
                         hermval(x_pret[:,0], matrix), hermval(x_pret[:,0], matrix), 
                         hermval(x_pret[:,1], matrix), hermval(x_pret[:,1], matrix),
                         hermval(x_pret[:,2], matrix), hermval(x_pret[:,2], matrix),
                         hermval(x_pret[:,3], matrix), hermval(x_pret[:,3], matrix),
                         hermval(x_pret[:,4], matrix), hermval(x_pret[:,4], matrix),
                         hermval(x_pret[:,5], matrix), hermval(x_pret[:,5], matrix),
                         matrix_indexex)

    elif n_modes==7:
        part = np.einsum('impdbfxjnqeczy, il, jl, ml, nl, pl, ql, dl, el, bl, cl, fl, zl, xl, yl, impdbfxjnqeczy -> l', rho, 
                         hermval(x_pret[:,0], matrix), hermval(x_pret[:,0], matrix), 
                         hermval(x_pret[:,1], matrix), hermval(x_pret[:,1], matrix),
                         hermval(x_pret[:,2], matrix), hermval(x_pret[:,2], matrix),
                         hermval(x_pret[:,3], matrix), hermval(x_pret[:,3], matrix),
                         hermval(x_pret[:,4], matrix), hermval(x_pret[:,4], matrix),
                         hermval(x_pret[:,5], matrix), hermval(x_pret[:,5], matrix),
                         hermval(x_pret[:,6], matrix), hermval(x_pret[:,6], matrix),
                         matrix_indexex)
    elif n_modes==8:
        part = np.einsum('impdbfxgjnqeczyh, il, jl, ml, nl, pl, ql, dl, el, bl, cl, fl, zl, xl, yl, gl, hl, impdbfxgjnqeczyh -> l', rho, 
                         hermval(x_pret[:,0], matrix), hermval(x_pret[:,0], matrix), 
                         hermval(x_pret[:,1], matrix), hermval(x_pret[:,1], matrix),
                         hermval(x_pret[:,2], matrix), hermval(x_pret[:,2], matrix),
                         hermval(x_pret[:,3], matrix), hermval(x_pret[:,3], matrix),
                         hermval(x_pret[:,4], matrix), hermval(x_pret[:,4], matrix),
                         hermval(x_pret[:,5], matrix), hermval(x_pret[:,5], matrix),
                         hermval(x_pret[:,6], matrix), hermval(x_pret[:,6], matrix),
                         hermval(x_pret[:,7], matrix), hermval(x_pret[:,7], matrix),
                         matrix_indexex)                     
    else:
        raise NotImplementedError('Not implemented for n_modes>8')
        
    return np.abs(part*part.conj())*np.pi**n_modes*np.prod(np.exp(-2*x_pret**2),axis=1)


def bootstrap_rho(rho, n_modes, n_photons, instances_size=10_000, random_size=5_000):
    """
    returns bootsprapped data format [size_data, n_modes]
    args: rho - density matrix of shape (n_photons**n_modes, n_photons**n_modes)
          n_modes - number of modes in rho
          n_photons - dimension of trunc state
    output: data[size_data, n_modes]
    """
    step = 0
    cur_length = 0
    rho_reshaped = reshaping_rho(rho, n_modes, n_photons)

    while cur_length <= instances_size:
        x_pret = np.random.uniform(-5, 5, size=(random_size, n_modes))
        p = np.random.uniform(size=random_size)
        probabilities = proba_rho(rho_reshaped, x_pret, n_modes, n_photons)
        eq = probabilities>=p
        length = x_pret[eq,:].shape[0]
        cur_length += length
        arr = np.asarray(x_pret[eq,:])
        if step==0:
            data = arr
            step = 1
        else:
            data = np.concatenate([data,arr], axis=0)
    return data[:instances_size]

def tensor_product_many_modes(operations, mode_list, n_modes, n_photons):
    """
    args: operations : List
          mode_list : sorted List 
    output:
          projection matrix
    
    """
    j = 0
    for i in range(n_modes,0,-1):
        if i==n_modes:
            if i==mode_list[::-1][j]:
                matrix = operations[::-1][j]
                if j<len(mode_list)-1:
                    j += 1 
            else:
                matrix = np.eye(n_photons)
        elif i==mode_list[::-1][j]:
            matrix = np.kron(operations[j], matrix)
            if j<len(mode_list)-1:
                j += 1 
        else:
            matrix = np.kron(matrix, np.eye(n_photons))
    return matrix


def project_homodyne(rho, modes_to_project, n_modes, n_photons, x=[], phi=None):
    """
    projection on modes_to_project x*cos(phi/2.) + p*sin(phi/2.) = x_i for i in modes_to_project;
    args: rho - density matrix 
          phi - phase of projection 
          modes_to_project - modes on which to project
          n_photons - photon number
          x - quadrature on which to proj according to modes_to_project
    output: rho_put density matrix of remaining 
    """
    disp = [0.]*len(modes_to_project) if not list(x) else x/np.sqrt(2.)

    # TO DO phi
    phi = [0.]*len(modes_to_project) 

    ## projection
    operations = []
    for ind, _ in enumerate(modes_to_project):
        proj_state = infinite_squeezing(phi[ind], n_photons)
        if disp[ind]!=0.:
            gate = D_gate(disp[ind], 0, 1, 1, n_photons)
            proj_state = gate @ proj_state
        proj_state = proj_state.reshape(proj_state.shape[0], 1)
        rho_proj = np.kron(proj_state, proj_state.conj().T)
        operations.append(rho_proj)

    projector = tensor_product_many_modes(operations, modes_to_project, n_modes, n_photons)
    rho_projected = projector @ rho @ projector.conj().T

    ## ptrace on remaining modes
    sel = []
    for j in range(1,n_modes+1):
        if j not in modes_to_project:
            sel.append(j-1)
    
    rho_out = ptrace(rho_projected, sel, n_photons-1)
    normalization = np.trace(rho_out)
    
    return rho_out/normalization


## main functions
def meas_X_rho(rho, modes_to_measure, n_modes, n_photons, proj='projection', flag='full'):
    return meas_X_theta_rho(rho, 0., modes_to_measure, n_modes, n_photons, proj=proj, flag=flag)

def meas_P_rho(rho, modes_to_measure, n_modes, n_photons, proj='projection', flag='full'):
    return meas_X_theta_rho(rho, np.pi/2., modes_to_measure, n_modes, n_photons, proj=proj, flag=flag)


def meas_X_theta_rho(rho, theta, modes_to_measure, n_modes, n_photons, proj='projection', flag = 'full'):
    """
    Measurement on X_theta (+probably P_theta simultaneous) quadrature
    args: rho - density matrix in shape (n_photons**n_modes, n_photons**n_modes)
          theta - list of measurement angles len = len(modes_to_measure)
          modes_to_measure - list of measurement modes
          n_photons - photon number
          proj and flag seems boolean
          output if proj=='projection':
                    if flag=="full": data[:, n_modes]
                    if flag=="part": sample_of_x [0, n_modes], result_state (n_modes-len(modes_to_measure))
                 if proj=="simXP": 
                    if flag=="full": data[:, 2*n_modes]   (x1, p1, x2, p2, .....)
                    if flag=="part": sample_of_x_p (x1, p1, x2, p2, .....), result_state (n_modes-len(modes_to_measure))
    """
    # rotate to theta angle
    for ind, mode in enumerate(modes_to_measure):
        if isinstance(theta, float):
            rotation = R_gate(theta, mode, n_modes, n_photons)
        else:
            rotation = R_gate(theta[ind], mode, n_modes, n_photons)
        rho_rotate = rotation @ rho @ rotation.conj().T

    # Get measurement values and output density matrix    
    if proj=='projection' and flag=='full':
        return bootstrap_rho(rho_rotate, n_modes, n_photons)

    if proj=='projection' and flag=='part':
       reduced_rho = ptrace(rho_rotate, list(map(lambda x: x - 1, modes_to_measure)), n_photons-1)
       sample_x = bootstrap_rho(reduced_rho, len(modes_to_measure), n_photons, 1)
       ### projection on sample_x
       rho_out = project_homodyne(rho, modes_to_measure, n_modes, n_photons, x=sample_x[0])
       return sample_x, rho_out
    
    if proj=='simXP' and flag=='full':
        # preparing vacuum additions
        for i in range(1, n_modes+1):
            idler_state = np.kron(vacuum(n_photons), idler_state) if i!=1 else vacuum(n_photons)
            
        idler_state = idler_state.reshape(idler_state.shape[0], 1)
        idler_rho = np.kron(idler_state, idler_state.conj().T)
        rho_sim = np.kron(rho, idler_rho) # 2*n_modes

        # homodyne rotations
        phi_ = np.pi/2.
        theta_ = np.pi/4.
        for i in range(1, n_modes+1):
            gate0 = BS_gate(theta_, phi_, i, n_modes+i, 2*n_modes, n_photons)
            rho_sim_ = gate0 @ rho_sim @ gate0.conj().T
            gate1 = R_gate(-np.pi/2., i, 2*n_modes, n_photons)
            rho_sim_ = gate1 @ rho_sim_ @ gate1.conj().T
        return bootstrap_rho(rho_sim_, 2*n_modes, n_photons)
    
    if proj=='simXP' and flag=='part':
        reduced_rho = ptrace(rho_rotate, list(map(lambda x: x - 1, modes_to_measure)), n_photons-1)
        # preparing vacuum additions
        active_modes = len(modes_to_measure)
        for i in range(active_modes):
            idler_state = np.kron(vacuum(n_photons), idler_state) if i!=0 else vacuum(n_photons)
        
        idler_state = idler_state.reshape(idler_state.shape[0], 1)
        idler_rho = np.kron(idler_state, idler_state.conj().T)
        rho_sim = np.kron(reduced_rho, idler_rho) # 2*modes_to_measure
        # homodyne rotations
        phi_ = np.pi/2.
        theta_ = np.pi/4.
        for i in range(1, active_modes+1):
            gate = BS_gate(theta_, phi_, i, n_modes+i, 2*active_modes, n_photons)
            rho_sim_ = gate @ rho_sim @ gate.conj().T
            gate = R_gate(-np.pi/2, i, 2*active_modes, n_photons)
            rho_sim_ = gate @ rho_sim_ @ gate.conj().T
        sample_xp = bootstrap_rho(rho_sim_, 2*active_modes, n_photons, 1)
        sample_x = sample_xp[:,:active_modes]
        rho_out = project_homodyne(rho, modes_to_measure, n_modes, n_photons, x=sample_x[0])
        return sample_xp, rho_out


def project_fock(rho, photon_number, modes_to_measure, n_modes, n_photons):
    """
    projection on fock basis.
    args: rho matrix is in (n_photons**n_modes, n_photons**n_modes) shape
         photon_number list of photon state on which to project
         modes_to_measure list of modes on which to project
         n_modes munber of modes 1, 2,...., n_modes
         n_photons number of photons
    output: rho_reduced or probability if n_modes==len(modes_to_measure) + 1
    """
    rho_reshaped = reshaping_rho(rho, n_modes, n_photons)
    final_list = []
    for i in range(1, n_modes+1):
        if i <= len(modes_to_measure):
            if i==modes_to_measure[i-1]:
                final_list.append(photon_number[i-1])
        else:
            final_list.append(slice(0, n_photons))
    final_rho = rho_reshaped[tuple(final_list*2)]
    return final_rho/np.trace(final_rho), np.real(np.trace(final_rho))