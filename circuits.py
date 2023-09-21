import qiskit as qsk
import numpy as np
from qiskit.quantum_info import Statevector
import helper_functions

####################### UNITARY BLOCKS ##############################

def basic_rx_block_4(params):
    sub_circ = qsk.QuantumCircuit(4)
    sub_circ.rx(params[0], 0)
    sub_circ.rx(params[1], 1)
    sub_circ.rx(params[2], 2)
    sub_circ.rx(params[3], 3)
    sub_circ.cx(0, 1)
    sub_circ.cx(1, 2)
    sub_circ.cx(2, 3)
    sub_circ.cx(3, 0)
    return sub_circ.to_gate()

def basic_unitary_block_4(params):
    sub_circ = qsk.QuantumCircuit(4)
    sub_circ.u(params[0], params[1], params[2], 0)
    sub_circ.u(params[3], params[4], params[5], 1)
    sub_circ.u(params[6], params[7], params[8], 2)
    sub_circ.u(params[9], params[10], params[11], 3)
    sub_circ.cx(0, 1)
    sub_circ.cx(1, 2)
    sub_circ.cx(2, 3)
    sub_circ.cx(3, 0)
    return sub_circ.to_gate()

# takes n_wires*3 params
def basic_unitary_block(params, n_qubits):
    sub_circ = qsk.QuantumCircuit(n_qubits)
    for i in range(0, n_qubits):
        sub_circ.u(params[i*(n_qubits-1)], params[1+(i*(n_qubits-1))], params[2+(i*(n_qubits-1))], i)
    for i in range(n_qubits - 1):
        sub_circ.cx(i, i+1)
    sub_circ.cx(n_qubits-1, 0)
    return sub_circ.to_gate()

def basic_rotation_block(params, n_qubits, parity):
    sub_circ = qsk.QuantumCircuit(n_qubits)
    for i in range(0, n_qubits):
        if parity:
            sub_circ.rx(params[i], i)
        else:
            sub_circ.ry(params[i], i)
    for i in range(n_qubits - 1):
        sub_circ.cx(i, i+1)
    sub_circ.cx(n_qubits-1, 0)
    return sub_circ.to_gate()

############################# CIRCUITS #############################

def random_circ(theta):
    '''
    initializa_parameters(n_qubits, depth)
    '''
    num_qbits, circ_depth = theta.shape
    var_circ = qsk.QuantumCircuit(num_qbits)

    odd_layer = 0

    for layer in range(circ_depth - 1):
        odd_layer = (layer+1)%2 == 0
        for qbit in range(num_qbits):
            
            if odd_layer:
                var_circ.ry(theta[qbit][layer], qbit)
            else:
                var_circ.rx(theta[qbit][layer], qbit)
            
            
        for qbit in range(0+odd_layer, num_qbits, 2):
            if (qbit + 1) < num_qbits:
                var_circ.cx(qbit, qbit+1)
        var_circ.barrier()

    for qbit in range(num_qbits):  # bonus layer at the end only has rx gates and no cx
        var_circ.rx(theta[qbit][circ_depth - 1], qbit)

    return var_circ, [*range(num_qbits-1, -1, -1)]


def TTN_reuse(params, depth, unitary):
    '''
    use: initialize_parameters(7*depth, ((n_ancilla+1)*2)*3))
    Umm be careful with 
    '''
    n_blocks, params_block = params.shape
    if unitary:
        n_ancilla = (params_block//3)//2-1
    else:
        n_ancilla = (params_block)//2-1
    wires_block = 2*(n_ancilla+1)
    n_wires = (n_ancilla+1)*4
    circ = qsk.QuantumCircuit(n_wires)

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[0*depth+i], wires_block), range(0, wires_block))
        else:
            circ.append(basic_rotation_block(params[0*depth+i], wires_block, i%2), range(0, wires_block))

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[1*depth+i], wires_block), range(1+n_ancilla, 1+n_ancilla+wires_block))
        else:
            circ.append(basic_rotation_block(params[1*depth+i], wires_block, i%2), range(1+n_ancilla, 1+n_ancilla+wires_block))

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[2*depth+i], wires_block), range(2*(1+n_ancilla), 2*(1+n_ancilla)+wires_block))
        else:
            circ.append(basic_rotation_block(params[2*depth+i], wires_block, i%2), range(2*(1+n_ancilla), 2*(1+n_ancilla)+wires_block))

    for i in range(2*(1+n_ancilla), 2*(1+n_ancilla)+1+n_ancilla):
        circ.reset(i)
    
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[3*depth+i], wires_block), range(1+n_ancilla, 1+n_ancilla+wires_block))
        else:
            circ.append(basic_rotation_block(params[3*depth+i], wires_block, i%2), range(1+n_ancilla, 1+n_ancilla+wires_block))

    for i in range(1+n_ancilla, 1+n_ancilla+wires_block):
       circ.reset(i) 

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[4*depth+i], wires_block), range(0, 0+wires_block))
        else:
            circ.append(basic_rotation_block(params[4*depth+i], wires_block, i%2), range(0, 0+wires_block))

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[5*depth+i], wires_block), range(1+n_ancilla, 1+n_ancilla+wires_block))
        else:
            circ.append(basic_rotation_block(params[5*depth+i], wires_block, i%2), range(1+n_ancilla, 1+n_ancilla+wires_block))

    for i in range(1+n_ancilla, 1+n_ancilla+1+n_ancilla):
        circ.reset(i)
    
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[6*depth+i],wires_block), range(0, wires_block))
        else:
            circ.append(basic_rotation_block(params[6*depth+i],wires_block), range(0, wires_block))

    return circ, [*range(n_wires-1, -1, -1)]

def MPS(params, depth, unitary):
    '''
    Use: initialize_theta(n_qubit*depth, 2*(n_ancilla+1)*3)
    Number of qubits equals number of blocks
    Number of ancillas is ((parameters per block // 3) // 2) - 1
    Number of total wires is (number of qubits + 1) * (number of ancillas + 1)
    Number of wires per block is 2 * (number of ancillas + 1)
    '''
    n_blocks, param_block = params.shape
    n_qubits = n_blocks//depth
    if unitary:
        n_ancilla = (param_block//3)//2-1
    else:
        n_ancilla = (param_block)//2-1
    n_wires = (n_qubits+1)*(n_ancilla+1)
    wires_block = 2*(n_ancilla+1)
    circ = qsk.QuantumCircuit(n_wires)
    count = 0
    for i in range(0, n_blocks, depth):
        for j in range(depth):
            if unitary:
                circ.append(basic_unitary_block(params[i+j], wires_block), range(count, count+wires_block))
            else:
                circ.append(basic_rotation_block(params[i+j], wires_block, j%2), range(count, count+wires_block))
        count += 1+n_ancilla
    measure = [i for i in range(n_wires-1, 0+2*(n_ancilla), -(n_ancilla+1))]
    return circ, measure

def TTN(params, depth, unitary):
    '''
    Only works for 8 qubits right now

    Use: initialize_parameters((n_qubit*depth)-1, (2*(n_ancilla+1))*3)
    Number of qubits equals number of blocks + 1
    Number of ancillas is ((parameters per block // 3) // 2) - 1
    Number of total wires is (number of qubits) * (number of ancillas + 1)
    Number of wires per block is 2 * (number of ancillas + 1)
    '''
    blocks, param_block = params.shape
    n_qubits = (blocks//depth) + 1
    if unitary:
        n_ancilla = ((param_block//3)//2)-1
    else:
        n_ancilla = ((param_block)//2)-1
    n_wires = (n_ancilla+1)*n_qubits
    wires_block = 2*(n_ancilla+1)

    circ = qsk.QuantumCircuit(n_wires)

    block_wires = []
    for i in range(0, n_wires, wires_block):
        block_wires.append([*range(i, i+wires_block)])
    
    temp1 = []
    for i in range(len(block_wires)//2):
        temp1.append(block_wires[i][-len(block_wires[1])//2:])

    temp2 = []
    for i in range(len(block_wires)//2, len(block_wires)):
        temp2.append(block_wires[i][:len(block_wires[1])//2])

    temp3 = []
    temp3.append(block_wires[1][-len(block_wires[1])//2:])
    temp3.append(block_wires[2][:len(block_wires[1])//2])


    block_wires.append([item for items in temp1 for item in items])
    block_wires.append([item for items in temp2 for item in items])
    block_wires.append([item for items in temp3 for item in items])

    for i in range(len(block_wires)):
        for j in range(depth):
            if unitary:
                circ.append(basic_unitary_block(params[i*depth+j], wires_block), block_wires[len(block_wires)-1-i])
            else:
                circ.append(basic_rotation_block(params[i*depth+j], wires_block, j%2), block_wires[len(block_wires)-1-i])

    #measure = [i for i in range(n_wires-1, -1, -(n_ancilla+1))]
    meas1 = [i for i in range((n_wires-1), (n_wires-1)//2, -(n_ancilla+1))]
    meas2 = [i for i in range(0, (n_wires-1)//2+1, n_ancilla+1)]
    measure = meas1 + meas2[::-1] 
    return circ, measure

def MERA(params, depth, unitary):
    '''
    Only works for 8 qubits right now

    Use: initialize_parameters(11, (2*(n_ancilla+1))*3)
    Number of qubits equals 8
    Number of ancillas is ((parameters per block // 3) // 2) - 1
    Number of total wires is (number of qubits) * (number of ancillas + 1)
    Number of wires per block is 2 * (number of ancillas + 1)
    '''
    n_blocks, params_block = params.shape
    n_qubits = 8
    if unitary:
        n_ancilla = ((params_block//3)//2)-1
    else:
        n_ancilla = ((params_block)//2)-1
    n_wires = (n_ancilla+1)*n_qubits
    wires_block = 2*(n_ancilla+1)

    circ = qsk.QuantumCircuit(n_wires)

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[0*depth+i], wires_block), range(n_wires//2-(1+n_ancilla), n_wires//2+(1+n_ancilla)))
        else:
            circ.append(basic_rotation_block(params[0*depth+i], wires_block, i%2), range(n_wires//2-(1+n_ancilla), n_wires//2+(1+n_ancilla)))

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[1*depth+i], wires_block), [*range(n_wires//2, n_wires//2+1+n_ancilla), *range(n_wires//2+n_wires//4, n_wires//2+n_wires//4+n_ancilla+1)])
        else:
            circ.append(basic_rotation_block(params[1*depth+i], wires_block, i%2), [*range(n_wires//2, n_wires//2+1+n_ancilla), *range(n_wires//2+n_wires//4, n_wires//2+n_wires//4+n_ancilla+1)])

    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[2*depth+i], wires_block), [*range(n_wires//2-1, n_wires//2-1-(1+n_ancilla), -1), *range(n_wires//2-n_wires//4-1, n_wires//2-n_wires//4-1-(n_ancilla+1), -1)])
        else:
            circ.append(basic_rotation_block(params[2*depth+i], wires_block, i%2), [*range(n_wires//2-1, n_wires//2-1-(1+n_ancilla), -1), *range(n_wires//2-n_wires//4-1, n_wires//2-n_wires//4-1-(n_ancilla+1), -1)])


    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[3*depth+i], wires_block), range(n_wires//2-(1+n_ancilla), n_wires//2+(1+n_ancilla)))
        else:
            circ.append(basic_rotation_block(params[3*depth+i], wires_block, i%2), range(n_wires//2-(1+n_ancilla), n_wires//2+(1+n_ancilla)))

    circ.barrier()
    count = 0
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[4*depth+i], wires_block), range(count, count+(2*(n_ancilla+1))))
        else:
            circ.append(basic_rotation_block(params[4*depth+i], wires_block, i%2), range(count, count+(2*(n_ancilla+1))))

    count += 2*(n_ancilla+1)
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[5*depth+i], wires_block), range(count, count+(2*(n_ancilla+1))))
        else:
            circ.append(basic_rotation_block(params[5*depth+i], wires_block, i%2), range(count, count+(2*(n_ancilla+1))))

    count += 2*(n_ancilla+1)
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[6*depth+i], wires_block), range(count, count+(2*(n_ancilla+1))))
        else:
            circ.append(basic_rotation_block(params[6*depth+i], wires_block, i%2), range(count, count+(2*(n_ancilla+1))))

    count += 2*(n_ancilla+1)
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[7*depth+i], wires_block), range(count, count+(2*(n_ancilla+1))))
        else:
            circ.append(basic_rotation_block(params[7*depth+i], wires_block, i%2), range(count, count+(2*(n_ancilla+1))))

    count = n_ancilla+1
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[8*depth+i], wires_block), range(count, count+(2*(n_ancilla+1))))
        else:
            circ.append(basic_rotation_block(params[8*depth+i], wires_block, i%2), range(count, count+(2*(n_ancilla+1))))

    count += 2*(n_ancilla+1)
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[9*depth+i], wires_block), range(count, count+(2*(n_ancilla+1))))
        else:
            circ.append(basic_rotation_block(params[9*depth+i], wires_block, i%2), range(count, count+(2*(n_ancilla+1))))

    count += 2*(n_ancilla+1)
    for i in range(depth):
        if unitary:
            circ.append(basic_unitary_block(params[10*depth+i], wires_block), range(count, count+(2*(n_ancilla+1))))  
        else:
            circ.append(basic_rotation_block(params[10*depth+i], wires_block, i%2), range(count, count+(2*(n_ancilla+1)))) 

    meas1 = [i for i in range((n_wires-1), (n_wires-1)//2, -(n_ancilla+1))]
    meas2 = [i for i in range(0, (n_wires-1)//2+1, n_ancilla+1)]
    measure = meas1 + meas2[::-1] 
    return circ, measure

#################################### STATES #################################################

def GHZ_sv(n_qubits):
    '''
    Returns the statevector representation of a GHZ states for a given number of qubits
    '''
    #return Statevector(helper_functions.simulate_circ_(GHZ_circ(n_qubits)))
    circ = GHZ_circ(n_qubits)
    return qsk.quantum_info.DensityMatrix(circ)


def GHZ_circ(n_qubits):
    '''
    Returns the circuit for a GHZ state for a given number of qubits
    '''
    circ = qsk.QuantumCircuit(n_qubits)
    circ.h(0)
    for i in range(n_qubits-1):
        circ.cx(i, i+1)
    return circ

def GHZ_decohere_circ(n_qubits, phase):
    '''
    Returns the circuit for a decohered GHZ state for a given number of qubits
    '''
    params = np.random.uniform(low=0.0, high=1.0, size=((n_qubits)))
    circ = qsk.QuantumCircuit(n_qubits)
    circ.h(0)
    for i in range(n_qubits-1):
        circ.cx(i, i+1)
    circ.barrier()
    for i in range(n_qubits):
        if phase == 'random':
            circ.p(params[i], i)
        else:
            circ.p(phase, i)
        #circ.p(0.1, i)
        circ.x(i)
    #circ.barrier()
    #for i in range(n_qubits-1, 0, -1):
    #    circ.cx(i-1, i)
    #circ.h(0)
    return circ

def GHZ_decohere_sv(n_qubits, phase):
    #return qsk.quantum_info.Statevector(GHZ_decohere_circ(n_qubits, phase))
    circ = GHZ_decohere_circ(n_qubits, phase)
    return qsk.quantum_info.DensityMatrix(circ)

def GHZ_mixed(n_qubits, phase):
    decohere = GHZ_decohere_sv(n_qubits, phase)
    cohere = GHZ_sv(n_qubits)
    return (1/2)*decohere+(1/2)*cohere

def cluster_state_circ(n_qubits, connections, phase=0):
    '''
    Returns the circuit for a cluster state for a given number of qubits
    connections: list of tuples that represent connections of graph that forms the cluster state
    phase: phase of controlled phase gates in circuit. Usual choise being 0 or pi
    '''
    circ = qsk.QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circ.h(i)
    for i in range(len(connections)):
        circ.cp(phase, connections[i][0], connections[i][1])
        #circ.h(connections[i][0])
    return circ

def cluster_state_sv(n_qubits, connections, phase=0):
    '''
    Returns the state vector representation of a cluster state for a given number of qubits
    connections: list of tuples that represent connections of graph that forms the cluster state
    phase: phase of controlled phase gates in circuit. Usual choise being 0 or pi
    '''
    #return helper_functions.simulate_circ_(cluster_state_circ(n_qubits, phase, connections))
    circ = cluster_state_circ(n_qubits, connections, phase)
    return qsk.quantum_info.DensityMatrix(circ)

def cluster_state_mixed(n_qubits, connections1, connections2, phase=0):
    '''
    Returns the state vector representation of a cluster state for a given number of qubits
    connections: list of tuples that represent connections of graph that forms the cluster state
    phase: phase of controlled phase gates in circuit. Usual choise being 0 or pi
    '''
    #return helper_functions.simulate_circ_(cluster_state_circ(n_qubits, phase, connections))
    cluster1 = cluster_state_sv(n_qubits, phase, connections1)
    cluster2 = cluster_state_sv(n_qubits, phase, connections2)
    return (1/2)*cluster1 + (1/2)*cluster2

def W_state_sv(n_qubits):
    arr = np.zeros(2**n_qubits)
    for i in range(n_qubits):
        arr[2**i] = 1/np.sqrt(n_qubits)
    w = qsk.quantum_info.Statevector(arr)
    return w

def main():
    return

if __name__ == "__main__":
    main()