import numpy as np
import qiskit as qsk
import circuits
import scipy.optimize as opt
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from datetime import datetime
#from google.colab import files


optimizer_data = []
optimizer_len = 0

def initialize_parameters(blocks, params_block):
    '''
    Initialize the parameters for variational circuit
    blocks: number of unitary blocks for most circuits, number of qubits for random circuit
    params_block: number of parameters per unitary block for most circuit, number of layers for random circuit
    '''
    #params = np.random.uniform(low=0.0, high=1.0, size=((blocks, params_block)))
    params = np.random.uniform(low=0.0, high=2*np.pi, size=((blocks, params_block)))
    return params

def initialize_theta(circ_type, n_qubits, n_ancillas, depth, unitary):
    if depth < 1:
            raise Exception("Sorry, the depth of any circuit must always be 1 or more") 
    if circ_type == 'MPS':
        if unitary:
            blocks, params_block = (n_qubits*depth), 2*(n_ancillas+1)*3
        else:
            blocks, params_block = (n_qubits*depth), 2*(n_ancillas+1)
        initial_theta = initialize_parameters(blocks, params_block)

    elif circ_type == 'TTN':
        if n_qubits != 8:
            raise Exception("Sorry, for TTN the number of qubits currently has to equal 8") 
        if unitary:
            blocks, params_block = (n_qubits-1)*depth, (2*(n_ancillas+1))*3
        else:
            blocks, params_block = (n_qubits-1)*depth, (2*(n_ancillas+1))
        initial_theta = initialize_parameters(blocks, params_block)

    elif circ_type == 'TTN_reuse':
        if n_qubits % 4 != 0:
            raise Exception("Sorry, qubit count can only be a multiple of 4. Also, ancilla count doesn't matter for this one. Check function details")
        n_ancillas = n_qubits//4-1
        if unitary:
            blocks, params_block = 7*depth, ((n_ancillas+1)*2)*3
        else:
            blocks, params_block = 7*depth, ((n_ancillas+1)*2)
        initial_theta = initialize_parameters(blocks, params_block)

    elif circ_type == 'MERA':
        if n_qubits != 8:
            raise Exception("Sorry, for MERA the number of qubits currently has to equal 8") 
        if unitary:
            blocks, params_block = 11*depth, (2*(n_ancillas+1))*3
        else:
            blocks, params_block = 11*depth, (2*(n_ancillas+1))
        initial_theta = initialize_parameters(blocks, params_block)

    elif circ_type == 'rand':
        blocks, params_block = n_qubits, depth
        initial_theta = initialize_parameters(n_qubits, depth)

    return initial_theta, [blocks, params_block]

def optimizer_callback(current_theta):
    """
    Helps store the theta funtion for each iteration as the variational circuit gets optimized
    """

    global optimizer_data
    global optimizer_len

    optimizer_data.append(current_theta)
    optimizer_len += 1
    return False

def reset():
    """
    Manually reset the global variables
    """

    global optimizer_data
    global optimizer_len

    optimizer_data = []
    optimizer_len = 0
    return

########################## LOSS FUNCTIONS #################################

def compute_loss_(theta_vector, *args):
    '''
    Compute the loss for a pass of the optimizer. Needed for scipy minimize function
    theta_vector: flattened vector of parameters for variational circuit, needed for minimize function
    args: arguments passed by minimize function, needed for minimize function. follows pattern:
        target = args[0]
        blocks = args[1]
        theta_block = args[2]
        n_qubit = args[3]
        n_ancilla = args[4]
        depth = args[5]
        circ_type = args[6]
        fid_type = args[7]
    '''
    target = args[0]
    blocks = args[1]
    theta_block = args[2]
    n_qubit = args[3]
    n_ancilla = args[4]
    depth = args[5]
    circ_type = args[6]
    fid_type = args[7]
    unitary = args[8]
    theta = np.reshape(theta_vector, (blocks, theta_block))
    fidelity = get_fidelity_(theta, target, circ_type, fid_type, n_qubit, depth, unitary)
    loss = get_loss_(fidelity)
    return loss

def get_loss_(fidelity):
    '''
    Returns 1-sqrt(fidelity), which is loss function for variational circuit
    '''
    #print("fid:", fidelity)
    loss = 1 - np.sqrt(fidelity)
    #print("loss:", loss)
    return loss

############################### Fidelity functions ######################################

def get_fidelity_(theta, target, circ_type, fid_type, n_qubits, depth, unitary):
    '''
    Returns fidelity for any variational circuit type. Accepts multiple ways of calculating fidelity
    theta: unflattened array of parameters for variational circuit. needed for constructing circuit
    target: target state or circuit
    circ_type: specify the variational ciruit type
    fid_type: specify how fidelity is calculated
    n_qubit: number of qubits for target
    '''
    if circ_type == "MPS":
        circ, measure = circuits.MPS(theta, depth, unitary)

    elif circ_type == 'TTN':
        circ, measure = circuits.TTN(theta, depth, unitary)

    elif circ_type == 'TTN_reuse':
        circ, measure = circuits.TTN_reuse(theta, depth)

    elif circ_type == 'MERA':
        circ, measure = circuits.MERA(theta, depth, unitary)

    elif circ_type == 'rand':
        circ, measure = circuits.random_circ(theta)

    if fid_type == 'sv':
        vc_state = simulate_circ_(circ)
        trace = [i for i in range(measure[0]) if i not in measure]
        #print(trace)
        traced_state = qsk.quantum_info.partial_trace(vc_state, trace)
        #print(traced_state)
        fidelity = compute_fidelity_(target, traced_state)

    elif fid_type == 'swap_sv':
        swap, n_wires = swap_test_sv(target, circ, measure)
        fidelity = swap_test_fidelity_sv(swap, n_wires)

    elif fid_type == 'swap':
        swap = swap_test(target, circ, measure)
        fidelity = swap_test_fidelity(swap)

    if fidelity <= 0:
        return 1e-6
    else:  
        return fidelity    

def compute_fidelity_(target, vc_state):
    '''
    Computes ideal fidelity in state vector form
    target: target state
    vc_state: variational circuit output state
    '''
    fidelity = qsk.quantum_info.state_fidelity(target, vc_state)
    return fidelity

######################### SWAP TEST FUNCTIONS #######################################

# builds swap test circuit with measurement for non-state vector methods
def swap_test(target, phi, meas):
    n_wires = target.num_qubits+phi.num_qubits+1
    swap_test = qsk.QuantumCircuit(n_wires, 1)
    swap_test.compose(target, qubits=range(1, 1+target.num_qubits), inplace=True)
    swap_test.compose(phi, qubits=range(1+target.num_qubits, 1+target.num_qubits+phi.num_qubits), inplace=True)
    swap_test.barrier()
    swap_test.h(0)
    for i in range(target.num_qubits):
        swap_test.cswap(0, i+1, target.num_qubits+1+meas[-(i+1)])
    swap_test.h(0)
    swap_test.measure(0,0)
    return swap_test

# returns fidelity between 2 circuits using measurements
def swap_test_fidelity(swap_test_circ):
    backend = AerSimulator() 
    compiled = qsk.transpile(swap_test_circ, backend)
    job = backend.run(compiled, shots=8192)
    results = job.result()
    counts = results.get_counts(compiled)
    if '1' in counts:
        b = counts['1']
    else:
        b = 0
        
    s = 1-(2/8192)*(b)
    
    return s

#build swap test circuit for use with state vector methods
def swap_test_sv(target, phi, meas):
    n_wires = target.num_qubits+phi.num_qubits+1
    swap_test = qsk.QuantumCircuit(n_wires)
    swap_test.compose(target, qubits=range(1, 1+target.num_qubits), inplace=True)
    swap_test.compose(phi, qubits=range(1+target.num_qubits, 1+target.num_qubits+phi.num_qubits), inplace=True)
    swap_test.barrier()
    swap_test.h(0)
    for i in range(target.num_qubits):
        swap_test.cswap(0, i+1, target.num_qubits+1+meas[-(i+1)])
    swap_test.h(0)
    return swap_test, n_wires
    
# returns fidelity between 2 circuits using state vector methods
def swap_test_fidelity_sv(swap_test_circ, n_wires):
    backend = qsk.Aer.get_backend('statevector_simulator')   # get simulator
    job = qsk.execute(swap_test_circ, backend)
    result = job.result()
    circ_statevect = qsk.quantum_info.Statevector(
        result.get_statevector(swap_test_circ))
    zero_vector = qsk.quantum_info.partial_trace(circ_statevect, range(1, n_wires))
    return 1-2*zero_vector.data[1][1].real

###############################################################

def simulate_circ_(circ):
    '''
    Simulates circuit and returns ideal statevector output
    circ: any quantum circuit
    '''
    backend = qsk.Aer.get_backend('statevector_simulator')   # get simulator
    job = qsk.execute(circ, backend)
    result = job.result()
    circ_statevect = qsk.quantum_info.Statevector(
        result.get_statevector(circ))
    # use partial trace if using double qubits for purification
    #circ_statevect = qsk.quantum_info.partial_trace(circ_statevect, [0])
    return circ_statevect

def compute_loss_gradient_(theta_flat_vector, *args):
    '''
    Computes loss gradient for variational circuit

    '''
    target = args[0]
    blocks = args[1]
    theta_block = args[2]
    n_qubit = args[3]
    n_ancilla = args[4]
    depth = args[5]
    circ_type = args[6]
    fid_type = args[7]
    unitary = args[8]
    theta = np.reshape(theta_flat_vector, (blocks, theta_block))  # reshapes the flat theta vector
    fidelity = get_fidelity_(theta, target, circ_type, fid_type, n_qubit, depth, unitary)

    # the derivative of the loss wrt fidelity
    dl_df = -0.5 * fidelity ** (-0.5)

    df_dtheta = []  # a list of partial derivatives of the fidelity wrt the theta parameters

    for index in range(len(theta_flat_vector)):
        layer_index = index // theta_block
        qbit_index = index % theta_block

        theta_plus = np.copy(theta)
        theta_plus[layer_index][qbit_index] += np.pi / 2  # added pi/2 to the ith theta parameter

        theta_minus = np.copy(theta)
        theta_minus[layer_index][qbit_index] -= np.pi / 2  # subtracted pi/2 to the ith theta parameter

        df_dtheta_i = 0.5 * (get_fidelity_(theta_plus, target, circ_type, fid_type, n_qubit, depth, unitary) - get_fidelity_(theta_minus, target, circ_type, fid_type, n_qubit, depth, unitary))  # ith derivative
        df_dtheta.append(df_dtheta_i)

    df_dtheta = np.array(df_dtheta)
    dl_dtheta = dl_df * df_dtheta  # chain rule to get partial derivative of loss wrt theta parameters
    return dl_dtheta

def optimize_theta_(theta, n_qubits, n_ancillas, depth, target, circ_type, fid_type, unitary):
    '''
    target will be either state or circuit, depending on type of fidelity

    args follows the pattern:
    target = args[0]
    blocks = args[1]
    theta_block = args[2]
    n_qubit = args[3]
    n_ancilla = args[4]
    depth = args[5]
    circ_type = args[6]
    fid_type = args[7]
    unitary = args[8]
    '''
    n_blocks, theta_block = theta.shape
    theta_flat_vector = np.reshape(theta, theta.size)
    global optimizer_data

    results = opt.minimize(compute_loss_, theta_flat_vector, args=(target, n_blocks, theta_block, n_qubits, n_ancillas, depth, circ_type, fid_type, unitary), method='BFGS',
                           jac=compute_loss_gradient_, callback=optimizer_callback, options={'maxiter': 100})

    return results, optimizer_data

################## RUNNING FUCNTIONS ########################

def run(n_qubit, n_ancilla, depth, circ_type, unitary, fid_type, target_state_type, phase, connections, seed=0):
    global optimizer_data
    global optimizer_len
    optimizer_data = []
    optimizer_len = 0

    target = get_target_state(target_state_type, n_qubit, phase, connections)

    if seed == -1:
        np.random.seed()
    else:
        np.random.seed(seed)
    initial_theta, block_info = initialize_theta(circ_type, n_qubit, n_ancilla, depth, unitary)

    optimizer_data.append(initial_theta)
    optimizer_len += 1
    results, optimizer_data = optimize_theta_(initial_theta, n_qubit, n_ancilla, depth, target, circ_type, fid_type, unitary)  # Final result 

    print("Done, final loss: {}".format(results.fun))
    return target, block_info

def run_and_plot(n_qubits, n_ancillas, depth, circ_type, unitary, fid_type, target_state_type, phase, connections, seed):
    target, block_info = run(n_qubits, n_ancillas, depth, circ_type, unitary, fid_type, target_state_type, phase, connections, seed)

    loss_data = []
    fidelity_data = []

    for theta_vect in optimizer_data:
        theta = np.reshape(theta_vect, (block_info[0], block_info[1]))
        fidelity = get_fidelity_(theta, target, circ_type, fid_type, n_qubits, depth, unitary)
        loss = get_loss_(fidelity)
        
        loss_data.append(loss)
        fidelity_data.append(fidelity)

    #optimized_theta = np.reshape(optimizer_data[-1], (block_info[0], block_info[1]))
    #phi = get_state_(optimized_theta, circ_type, depth, unitary)   
        
    plt.plot(range(len(loss_data)), loss_data, label='loss', marker='o', linestyle='--')
    plt.title('Loss vs Iterations for %s with %i Qubits. Using %s Fidelity calculator, %s Target State type, and %i for random seed'%(circ_type, n_qubits, fid_type, target_state_type, seed))
    plt.xlabel('Num Iterations')
    plt.ylabel('Loss = 1 - Sqrt(Fidelity)')
    now = datetime.now()
    plt.savefig('%s.png' %(now.strftime("%H_%M_%S")), bbox_inches="tight")
    #files.download('%s.png' %(now.strftime("%H_%M_%S")))
    plt.show()


def run_n_ancillas(n_qubits, ancilla_list, depth, circ_type, unitary, fid_type, target_state_type, phase, connections, seed):
    losses = []

    for ancilla in ancilla_list:
        target, block_info = run(n_qubits, ancilla, depth, circ_type, unitary, fid_type, target_state_type, phase, connections, seed)

        loss_data = []
        fidelity_data = []

        for theta_vect in optimizer_data:
            theta = np.reshape(theta_vect, (block_info[0], block_info[1]))
            fidelity = get_fidelity_(theta, target, circ_type, fid_type, n_qubits, depth, unitary)
            loss = get_loss_(fidelity)
            
            loss_data.append(loss)
            fidelity_data.append(fidelity)

        losses.append(loss_data)
    
    for i in range(len(losses)):
        plt.plot(range(len(losses[i])), losses[i], label='Ancillas = %i' %ancilla_list[i], marker='o', linestyle='--')
    plt.title('Loss vs Iterations for %s with %i Qubits and Depth = %i. Using %s Fidelity calculator, %s Target State type, and %i for random seed'%(circ_type, n_qubits, depth, fid_type, target_state_type, seed))
    plt.xlabel('Num Iterations')
    plt.ylabel('Loss = 1 - Sqrt(Fidelity)')
    now = datetime.now()
    plt.savefig('%s.png' %(now.strftime("%H_%M_%S")), bbox_inches="tight")
    #files.download('%s.png' %(now.strftime("%H_%M_%S")))
    plt.legend()
    plt.show()

def run_n_depths(n_qubits, n_ancillas, depth_list, circ_type, unitary, fid_type, target_state_type, phase, connections, seed):
    losses = []

    for d in depth_list:
        target, block_info = run(n_qubits, n_ancillas, d, circ_type, unitary, fid_type, target_state_type, phase, connections, seed)

        loss_data = []
        fidelity_data = []

        for theta_vect in optimizer_data:
            theta = np.reshape(theta_vect, (block_info[0], block_info[1]))
            fidelity = get_fidelity_(theta, target, circ_type, fid_type, n_qubits, d, unitary)
            loss = get_loss_(fidelity)
            
            loss_data.append(loss)
            fidelity_data.append(fidelity)

        losses.append(loss_data)
    
    for i in range(len(losses)):
        plt.plot(range(len(losses[i])), losses[i], label='Depth = %i' %depth_list[i], marker='o', linestyle='--')
    plt.title('Loss vs Iterations for %s with %i Qubits and %i Ancillas. Using %s Fidelity calculator, %s Target State type, and %i for random seed' %(circ_type, n_qubits, n_ancillas, fid_type, target_state_type, seed))
    plt.xlabel('Num Iterations')
    plt.ylabel('Loss = 1 - Sqrt(Fidelity)')
    plt.legend()
    now = datetime.now()
    plt.savefig('%s.png' %(now.strftime("%H_%M_%S")), bbox_inches="tight")
    #files.download('%s.png' %(now.strftime("%H_%M_%S")))
    plt.show()

def run_n_ancillas_depths(n_qubits, ancilla_list, depth_list, circ_type, unitary, fid_type, target_state_type, phase, connections, seed):
    losses = []
    ancilla_count = []
    depth_count = []

    for ancilla in ancilla_list:
        for d in depth_list:
            psi, block_info = run(n_qubits, ancilla, d, circ_type, unitary, fid_type, target_state_type, phase, connections, seed)

            loss_data = []
            fidelity_data = []

            for theta_vect in optimizer_data:
                theta = np.reshape(theta_vect, (block_info[0], block_info[1]))
                fidelity = get_fidelity_(theta, psi, circ_type, fid_type, n_qubits, d, unitary)
                loss = get_loss_(fidelity)
                
                loss_data.append(loss)
                fidelity_data.append(fidelity)

            losses.append(loss_data)
            ancilla_count.append(ancilla)
            depth_count.append(d)
    
    for i in range(len(losses)):
        plt.plot(range(len(losses[i])), losses[i], label='Ancilla = %i,Depth = %i' %(ancilla_count[i], depth_count[i]), marker='o', linestyle='--')
    plt.title('Loss vs Iterations for %s with %i Qubits. Using %s Fidelity calculator, %s Target State type, and %i for random seed' %(circ_type, n_qubits, fid_type, target_state_type, seed))
    plt.xlabel('Num Iterations')
    plt.ylabel('Loss = 1 - Sqrt(Fidelity)')
    plt.legend()
    now = datetime.now()
    plt.savefig('%s.png' %(now.strftime("%H_%M_%S")), bbox_inches="tight")
    #files.download('%s.png' %(now.strftime("%H_%M_%S")))
    plt.show()

############################# STATE FUNCTIONS ###################################

def get_state_(theta, circ_type, depth, unitary):

    if circ_type == "MPS":
        circ, measure = circuits.MPS(theta, depth, unitary)
    elif circ_type == 'TTN':
        circ, measure = circuits.TTN(theta, depth, unitary)
    elif circ_type == 'TTN_reuse':
        circ, measure = circuits.TTN_reuse(theta, depth)
    elif circ_type == 'MERA':
        circ, measure = circuits.MERA(theta, depth, unitary)
    elif circ_type == 'rand':
        circ, measure = circuits.random_circ(theta)
    else:
        raise Exception("Unknown circ type. Options are: MPS, TTN, TTN_reuse, MERA, rand")

    #vc_state = simulate_circ_(circ)
    vc_state = qsk.quantum_info.DensityMatrix(circ)
    traced_state = qsk.quantum_info.partial_trace(vc_state, [i for i in range(measure[0]) if i not in measure])
    return traced_state

def get_target_state(state_type, n_qubits, phase=0, connections=None):
    if state_type == 'GHZ_sv':
        state = circuits.GHZ_sv(n_qubits)
    elif state_type == 'GHZ_circ':
        state = circuits.GHZ_circ(n_qubits)
    elif state_type == 'GHZ_decohere_circ':
        state = circuits.GHZ_decohere_circ(n_qubits, phase)
    elif state_type == 'GHZ_decohere_sv':
        state = circuits.GHZ_decohere_sv(n_qubits, phase)
    elif state_type == 'GHZ_mixed':
        state = circuits.GHZ_mixed(n_qubits, phase)
        print('Purity of selected mixed state is', state.purity())
    elif state_type == 'cluster_sv':
        state = circuits.cluster_state_sv(n_qubits, connections, phase)
    elif state_type == 'cluster_circ':
        state = circuits.cluster_state_circ(n_qubits, connections, phase)
    #elif state_type == 'cluster_mixed':
    #    state = circuits.cluster_state_mixed(n_qubits, connections1, connections2, phase=0)
    elif state_type == 'W_state':
        state = circuits.W_state_sv(n_qubits)
    else:
        raise Exception('Target state type not recognized. Options are: GHZ_sv, GHZ_circ, GHZ_decoherence, GHZ_mixed, cluster_sv, cluster_circ, W_state')
    return state

def main():
    return

if __name__ == "__main__":
    main()