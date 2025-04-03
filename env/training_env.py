import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from .abstractenv import AbstractEnv
from training_utils import sparse_to_tuple

import tensorflow as tf


class OnlineDistributedEnv(AbstractEnv):
    def __init__(self, S, A, B, Pt, R, PT):
        """ Initialize the environment. Here, we represent the joint system
        state internally by concatenating the state of each agent.
        """
        self.S, self.A, self.B, self.Pt, self.R, self.PT = S, A, B, Pt, R, PT
        self.N = self.S.shape[0]
        self.n = int((self.A).shape[0] / (self.S).shape[0])    # Each state's size (since each subsystem may not be one dimension)
        self.m = int((self.B).shape[1] / (self.S).shape[0])    # Each input's size 
        self.p = self.m
        # self.P, self.K, self.X = _lqr_solve(A, B, Q, R)  # Solve a LQR problem to obtain P, K and X
        # self.QT = self.P

    def random_x0(self, batch_size, num_nodes, mean, std, seed=None):
        """ Generate a random initial condition 
        """
        if seed is not None:
            tf.random.set_seed(seed)
        x0s = tf.random.normal((batch_size, num_nodes, self.n, 1), mean=mean, stddev=std)   # we have in total num_samples of \mathbb{R}^{N\times p}
        return x0s
    
    def dynamic_update(self, xt, ut, num_nodes, noise_std):
        """ Takes in a state and control input, output the next state.
        Note that for the distributed training, we need to update over the total number of batches and nodes.
        """

        batch_size = xt.shape[0] // num_nodes
        system_dim = xt.shape[1]
        xt_reshape, ut_reshape = tf.reshape(xt, (-1, 1)), tf.reshape(ut, (-1, 1))   # compute the size of the new dimension automatically based on the size of the original tensor.
        
        # Transform A and B matrices into their block diagonal forms and save them by coordinates, non-zero values and shape
        list_A = [self.A]*batch_size
        list_B = [self.B]*batch_size
        block_diag_A = sp.block_diag(list_A, format='csr')
        block_diag_B = sp.block_diag(list_B, format='csr')
        block_diag_A_shape = sparse_to_tuple(block_diag_A)
        block_diag_B_shape = sparse_to_tuple(block_diag_B)
        coords_A, values_A, shape_A = block_diag_A_shape
        coords_A = tf.cast(coords_A, dtype = tf.int64)
        coords_B, values_B, shape_B = block_diag_B_shape
        coords_B = tf.cast(coords_B, dtype = tf.int64)

        # Create and save the block diagonal matrices A and B as sparse tensors by a tensor with (indices, non-zero values and shape)
        block_diag_A = tf.sparse.SparseTensor(indices=coords_A, values=values_A, dense_shape=shape_A)
        block_diag_B = tf.sparse.SparseTensor(indices=coords_B, values=values_B, dense_shape=shape_B)
        
 

        ### Now, we compute the states in the next time step distributedly, using the stacked networked system dynamics: \mbox{vec}(X^\T(t+1)) = \block_diag{A}\mbox{vec}(X^\T(t))+\block_diag{B}\mbox{vec}(U^\T(t))
        # noise_var = noise_std**2
        # print("Noise std: {:.3f}, noise_variance {:.3f}".format(noise_std, noise_var))
        sample_size = xt_reshape.shape[0]
        noise = np.random.normal(0, noise_std, size=(sample_size, 1))    # Add some noise in the dynamics
        x_next = tf.sparse.sparse_dense_matmul(block_diag_A, xt_reshape) + tf.sparse.sparse_dense_matmul(block_diag_B, ut_reshape) + noise    # Compute the next states in batches <2000, 1>
        
        # Now, we need to change the x_next back to the shape <1000, 2>
        x_next = tf.reshape(x_next, (xt.shape[0], -1))
        return x_next      


    def get_trajectory(self, list_adj_matrix, model, x0s, z_tminus1, T, noise_std, training_weights, activation_func_name):
        # Their architecture takes in an additional time dimension
        # num_nodes = x0s.shape[1]
        # list_x0s = [x0 for x0 in x0s]   # transform the np.array into a list
        # x0_in = tf.concat(list_x0s, 0)  # Now, we combine the first dimension with the second one, and the initial states x0s are now with the shape (batch_size*num_nodes, n, 1).
        #                                 # Here, each elements in list_x0s are stacked one after another along axis 0, i.e., along the batch_size. For example, the first batch_size is to store the nodes from 0 to N-1 and then we stack the second batch_size, which involves the second batch of nodes from 0 to N-1
        
        adj_matrix = sp.block_diag(list_adj_matrix, format='csr')     # Copy the same adjacency matrix S by 100 times, and we will have a 
        adj_matrix_shape = sparse_to_tuple(adj_matrix)

        batch_size_times_num_nodes = x0s.shape[0]
        num_nodes = batch_size_times_num_nodes // len(list_adj_matrix)

        xt = x0s
        xt = tf.reshape(xt, (batch_size_times_num_nodes, self.n))
        xt = tf.cast(xt, tf.float64)

        zt = z_tminus1
        zt = tf.reshape(zt, (batch_size_times_num_nodes, self.p))
        zt = tf.cast(zt, tf.float64)
        
        # Here, we set some 0 batches to store the data
        x = tf.zeros(shape = (batch_size_times_num_nodes, T+1, self.n), dtype=tf.float64)   # Set initial states X[0]
        z = tf.zeros(shape = (batch_size_times_num_nodes, T+1, self.p), dtype=tf.float64)   # Set initial hidden states Z[-1]
        u = tf.zeros(shape = (batch_size_times_num_nodes, T, self.m), dtype=tf.float64)     # Create a set to store control inputs U[t]
        
        indices = tf.constant([[i, 0] for i in range(batch_size_times_num_nodes)], dtype=tf.int64)
        x = tf.tensor_scatter_nd_update(x, indices, xt)    # x[:, 0]. The initial states are filled in the the initial time step, i.e., time t = 0, each is with \mathbb{R}^{n\times 1}
        z = tf.tensor_scatter_nd_update(z, indices, zt)    # z[:, 0]. 

        ### Compute for multiple step (use to compute the multi-step loss) and generate the control input and the next states
        for t in range(T):
            ut, zt = model(adj_matrix_shape, t, z[:, t], x[:, t], training_weights, num_nodes, activation_func_name)  # Compute the control input u(t) and the hidden state z(t) at this time step. Note that z(t) starts at time -1. This will generate a tensor with the shape = (batch_size*num_node, m)    
            indices = tf.constant([[i, t] for i in range(batch_size_times_num_nodes)], dtype=tf.int64)
            u = tf.tensor_scatter_nd_update(u, indices, ut)     # Collect the control input at this step u[:, t]
            x_next = self.dynamic_update(x[:, t], u[:, t], num_nodes, noise_std)  
            idx = tf.constant([[i, t+1] for i in range(batch_size_times_num_nodes)], dtype=tf.int64)
            z = tf.tensor_scatter_nd_update(z, idx, zt)      # Collect the hidden at this step z[:, t], but since it start at z(-1), we collect it at the one step shifted position here. 
            x = tf.tensor_scatter_nd_update(x, idx, x_next)  # x[:, t+1]

        return x, u, z
    

    def gcnn_cost(self, x_traj, u_traj, T, batch_size):
        # batch_size, T, _, _ = u_traj.size()
        # x_, u_ = x_traj.flatten(2), u_traj.flatten(2)
        state_cost = 0
        control_cost = 0

        # Stack up Q, QT and R matrices for all batch_size
        list_Pt = [self.Pt]*batch_size
        list_R = [self.R]*batch_size
        block_diag_Pt = sp.block_diag(list_Pt, format='csr')
        block_diag_R = sp.block_diag(list_R, format='csr')
        block_diag_Pt_shape = sparse_to_tuple(block_diag_Pt)
        block_diag_R_shape = sparse_to_tuple(block_diag_R)
        coords_Pt, values_Pt, shape_Pt = block_diag_Pt_shape
        coords_Pt = tf.cast(coords_Pt, dtype = tf.int64)
        coords_R, values_R, shape_R = block_diag_R_shape
        coords_R = tf.cast(coords_R, dtype = tf.int64)

        # Create and save the block diagonal matrices A and B as sparse tensors by a tensor with (indices, non-zero values and shape)
        block_diag_Pt = tf.sparse.SparseTensor(indices=coords_Pt, values=values_Pt, dense_shape=shape_Pt)
        block_diag_R = tf.sparse.SparseTensor(indices=coords_R, values=values_R, dense_shape=shape_R)
        block_diag_PT = block_diag_Pt
        batch_size_times_num_nodes = x_traj[:,0].shape[0]
        for t in range(T):
            x_t = tf.reshape(x_traj[:,t], (-1, 1))
            u_t = tf.reshape(u_traj[:,t], (-1, 1))
            state_cost += (x_t * tf.sparse.sparse_dense_matmul(block_diag_Pt, x_t))     # State costs in the LQR problem
            control_cost += (u_t * tf.sparse.sparse_dense_matmul(block_diag_R, u_t))   # Control cost in the LQR problem
        x_T = tf.reshape(x_traj[:,T], (-1, 1))
        terminal_cost = (x_T * tf.sparse.sparse_dense_matmul(block_diag_PT, x_T))         # Terminal cost in the LQR problem

        state_cost = tf.reduce_sum(tf.reshape(state_cost, (batch_size_times_num_nodes, self.n)), axis=1, keepdims=True)
        control_cost = tf.reduce_sum(tf.reshape(control_cost, (batch_size_times_num_nodes, self.n)), axis=1, keepdims=True)
        terminal_cost = tf.reduce_sum(tf.reshape(terminal_cost, (batch_size_times_num_nodes, self.n)), axis=1, keepdims=True)
        
        cost_over_horizon  = state_cost + terminal_cost + control_cost
        
        return cost_over_horizon


    
    ### Construct the consensus matrix W for distributed training
    def consensus_matrix(self, adj_matrix):
        """Compute Metropolis-Hastings weights for a given adjacency matrix.
        """
        G = nx.from_numpy_array(adj_matrix)  # Create graph from adjacency matrix
        N = adj_matrix.shape[0]  # Number of nodes
        c_mtx = np.zeros((N, N))  # Initialize weight matrix
    
        for i in range(N):
            neighbors = list(G.neighbors(i))  # Get neighbors of node i
            d_i = len(neighbors)  # Degree of node i
        
            for j in neighbors:
                d_j = len(list(G.neighbors(j)))  # Degree of node j
                c_mtx[i, j] = 1 / max(d_i, d_j)  # Assign MH weight

            c_mtx[i, i] = 1 - np.sum(c_mtx[i])  # Ensure row sums to 1
    
        return c_mtx   





### Generate a random graph
def _generate_graph(N, k=8, v=1, p=0.8, gtype='gaussian_random_partition', directedG = False, gseed=None):
    if gtype == 'gaussian_random_partition':
        G = nx.gaussian_random_partition_graph(N, s=k, v=v, p_in=p, p_out=p/10, directed = directedG, seed=gseed)
    elif gtype == 'connected_watts_strogatz':
        G = nx.connected_watts_strogatz_graph(N, k, p, tries=1000, seed=gseed)   # Not set directed!!
    elif gtype == 'fast_gnp_random':
        G = nx.generators.random_graphs.fast_gnp_random_graph(N, float(k) / float(N), seed=gseed)   # Not set directed!!
    elif gtype == 'barabasi_albert':
        G = nx.generators.random_graphs.barabasi_albert_graph(N, int(np.round(k * p)), seed=gseed)  # Not set directed!!
    else:
        raise ValueError('Undefined graph type!')

    # Here, we generate the normalized adjacency matrix S
    G = nx.gaussian_random_partition_graph(N, s=k, v=v, p_in=p, p_out=p/10, directed = directedG, seed=gseed)
    S = nx.adjacency_matrix(G, nodelist=list(range(N)), weight=None).toarray()
    norm_S = np.linalg.norm(S, ord=2)
    S = S / norm_S    # Normalize the adjacency matrix by the Frobenius norm of the matrix S
    return G, S

# Function to check the controllability matrix rank
def _is_controllable(A, B):
    full_rank = A.shape[0]
    C = B
    for i in range(1, full_rank):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    return np.linalg.matrix_rank(C) == full_rank


### Generate the dynamic model of the linear networked system (A,B)
def _generate_dynamics(G, S, system_dim, A_norm=0.995, B_norm=1, AB_hops=3):
    """ Generates dynamics matrices (A,B) based on graph G (characterized by S)
    """
    ## TODO:    - Generate sparse A and B based on G
    ##          - make it work for general p and q
    N = S.shape[0]    # Rows of S
    # eig_vecs = np.linalg.eigh(S)[1]   # Eigenvectors of S, [1] is to extract the eigenvalue of S, regardless of the eigenvalues.
    #############################################################################################################################################
    # We may need to change this part to define the system matrix according to S
    A = np.random.randn(system_dim*N, system_dim*N)
    B = np.random.randn(system_dim*N, system_dim*N)
    
    # Here, we construct a network of double integrators
    for i, j in np.ndindex(S.shape):
        if S[i,j]==0 & i!=j:
            # print(f"Element at ({i}, {j}): {S[i, j]}")
            A[i*system_dim:(i+1)*system_dim, j*system_dim:(j+1)*system_dim] = 0
            B[i*system_dim:(i+1)*system_dim, j*system_dim:(j+1)*system_dim] = 0
        # else:
        #     A[i*system_dim:(i+1)*system_dim, j*system_dim:(j+1)*system_dim] = np.array([[0, 1], [0, 0]])
        #     B[i*system_dim:(i+1)*system_dim, j] = np.array([[0], [1]])

    # Normalize A and B
    A = A / np.linalg.norm(A, ord=2) * A_norm
    B = B / np.linalg.norm(B, ord=2) * B_norm

    while not _is_controllable(A, B):
        A = np.random.randn(system_dim*N, system_dim*N)
        B = np.random.randn(system_dim*N, system_dim*N)
        for i, j in np.ndindex(S.shape):
            if S[i,j]==0 & i!=j:
                A[i*system_dim:(i+1)*system_dim, j*system_dim:(j+1)*system_dim] = 0
                B[i*system_dim:(i+1)*system_dim, j*system_dim:(j+1)*system_dim] = 0
            # else:
            #     A[i*system_dim:(i+1)*system_dim, j*system_dim:(j+1)*system_dim] = np.array([[0, 1], [0, 0]])
            #     B[i*system_dim:(i+1)*system_dim, j] = np.array([[0], [1]])
        
        # Normalize A and B
        A = A / np.linalg.norm(A, ord=2) * A_norm
        B = B / np.linalg.norm(B, ord=2) * B_norm
    #############################################################################################################################################
    return A, B


### This function is used to generate the simulation environment
def generate_training_env(N, system_dim, k, v, p, gtype, directedG, gseed, A_norm=0.995, B_norm=1, AB_hops=3):
    G, S = _generate_graph(N, k, v, p, gtype=gtype, directedG=directedG, gseed=gseed)    # Generate a graph G and normalized topology matrix S
    A, B = _generate_dynamics(G, S, system_dim, A_norm, B_norm, AB_hops)   # Generate A and B matrix for the linear networked systems based on the topology S
    Pt = np.eye(system_dim*N)   # Q parameter matrix in the LQR problem
    R = np.eye(system_dim*N)   # R parameter matrix in the LQR problem
    PT = Pt
    
    # A copy of the constructor function (running this function is the same as importing the whole class)
    env = OnlineDistributedEnv(S, A, B, Pt, R, PT)    
    return env, S



































