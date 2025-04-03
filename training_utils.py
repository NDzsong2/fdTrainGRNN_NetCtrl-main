import numpy as np
import scipy.sparse as sp
import networkx as nx
import tensorflow as tf
# from utils import *



def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo().astype(np.float64)
    indices = np.array([coo.row, coo.col]).T
    return tf.SparseTensor(indices, coo.data, coo.shape)



def gcnn_d_training(env, list_adj_matrix, gcnn_model, x0s, timeHorizon, training_weights, activation_func_name):
    # adj = sp.block_diag(list_adj, format='csr')

    # # Normalize the adjacency/Laplacian matrix
    # if shiftop == 'nadj':
    #     # Normalized adjacency
    #     L_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    #     L_norm_tuple = preprocess_adj(adj)
    # else:
    #     # Normalized Laplacian
    #     L_norm = sp.eye(adj.shape[0]) - normalize_adj(adj)
    #     L_norm_tuple = normalize_laplacian(adj)
    batch_size = len(list_adj_matrix)
    if 'gcnn_local' in gcnn_model.__name__:    # Check the existence of this string in this name string. In this condition, we choose the distributed GCNN.
        # gradients = update_gradients_d(training_loss_batch, training_weights)
        with tf.GradientTape() as tape:            
            tape.watch(training_weights)
            ### Step 3.1: get the trajectory of all the nodes during a small time horizon in a distributed manner
            xtraj, utraj = env.get_trajectory(list_adj_matrix, gcnn_model, x0s, timeHorizon, training_weights, activation_func_name)
            
            ### Step 3.2: Give x and u, we compute the cost value for all nodes and mini-batches. 
            # (Here, x involves the all the past time (xt) and the final value (xT), while u only involves the past values (ut).)
            cost_over_horizon = env.gcnn_cost(xtraj, utraj, timeHorizon, batch_size)   
            training_loss_batch = cost_over_horizon

        ### Step 3.3: compute the gradient from the loss and weights
        gradients = tape.gradient(training_loss_batch, training_weights)

        ### Step 3.4: change the shape of the gradient and obtain the gradients that we need
        miniBatch_gradients = []
        miniBatch_count = 0
        for adj_matrix in list_adj_matrix:    # batch_size
            num_nodes = np.shape(adj_matrix)[0]
            miniBatch_grad = []
            for ith_layer_grad in gradients:
                ith_layer_grad_miniBatch = ith_layer_grad[miniBatch_count:miniBatch_count+num_nodes]
                miniBatch_grad.append(ith_layer_grad_miniBatch)    # Append by layers
            miniBatch_gradients.append(miniBatch_grad)     # Append by mini-batches
            miniBatch_count += num_nodes
        gradients = miniBatch_gradients        


    elif 'gcnn_share' in gcnn_model.__name__:
        with tf.GradientTape() as tape:            
            tape.watch(training_weights)
            ### Step 3.1: get the trajectory of all the nodes during a small time horizon in a distributed manner
            xtraj, utraj = env.get_trajectory(list_adj_matrix, gcnn_model, x0s, timeHorizon, training_weights, activation_func_name)
            
            ### Step 3.2: Give x and u, we compute the cost value for all nodes and mini-batches. 
            # (Here, x involves the all the past time (xt) and the final value (xT), while u only involves the past values (ut).)
            cost_over_horizon = env.gcnn_cost(xtraj, utraj, timeHorizon, batch_size)   
            training_loss_batch = cost_over_horizon

        ### Step 3.3: compute the gradient from the loss and weights
        gradients = tape.gradient(training_loss_batch, training_weights)

        ### Step 3.4: change the shape of the gradient and obtain the gradients that we need
        miniBatch_gradients = []
        miniBatch_count = 0
        for adj_matrix in list_adj_matrix:    # batch_size
            num_nodes = np.shape(adj_matrix)[0]
            miniBatch_grad = []
            for ith_layer_grad in gradients:
                ith_layer_grad_miniBatch = ith_layer_grad[miniBatch_count:miniBatch_count+num_nodes]
                miniBatch_grad.append(ith_layer_grad_miniBatch)    # Append by layers
            miniBatch_gradients.append(miniBatch_grad)     # Append by mini-batches
            miniBatch_count += num_nodes
        gradients = miniBatch_gradients        
    elif 'gcnn_c' in gcnn_model.__name__:  # In this condition, we choose the centralized GCNN.
        pass
    return training_loss_batch, gradients



def grnn_d_training(env, list_adj_matrix, gnn_model, x0s, z_tminus1, timeHorizon, noise_std, training_weights, activation_func_name):
    # adj = sp.block_diag(list_adj, format='csr')

    # # Normalize the adjacency/Laplacian matrix
    # if shiftop == 'nadj':
    #     # Normalized adjacency
    #     L_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    #     L_norm_tuple = preprocess_adj(adj)
    # else:
    #     # Normalized Laplacian
    #     L_norm = sp.eye(adj.shape[0]) - normalize_adj(adj)
    #     L_norm_tuple = normalize_laplacian(adj)
    batch_size = len(list_adj_matrix)
    if 'grnn_d' in gnn_model.__name__:    # Check the existence of this string in this name string. In this condition, we choose the distributed GRNN.
        # gradients = update_gradients_d(training_loss_batch, training_weights)
        with tf.GradientTape() as tape:            
            tape.watch(training_weights)
            ### Step 3.1: get the trajectory of all the nodes during a small time horizon in a distributed manner
            xtraj, utraj, ztraj = env.get_trajectory(list_adj_matrix, gnn_model, x0s, z_tminus1, timeHorizon, noise_std, training_weights, activation_func_name)
            
            ### Step 3.2: Give x and u, we compute the cost value for all nodes and mini-batches. 
            # (Here, x involves the all the past time (xt) and the final value (xT), while u only involves the past values (ut).)
            cost_over_horizon = env.gcnn_cost(xtraj, utraj, timeHorizon, batch_size)   
            training_loss_batch = cost_over_horizon

        ### Step 3.3: compute the gradient from the loss and weights
        gradients = tape.gradient(training_loss_batch, training_weights)

        ### Step 3.4: change the shape of the gradient and obtain the gradients that we need
        miniBatch_gradients = []
        miniBatch_count = 0
        for adj_matrix in list_adj_matrix:    # batch_size
            num_nodes = np.shape(adj_matrix)[0]
            miniBatch_grad = []
            for ith_layer_grad in gradients:
                ith_layer_grad_miniBatch = ith_layer_grad[miniBatch_count:miniBatch_count+num_nodes]
                miniBatch_grad.append(ith_layer_grad_miniBatch)    # Append by layers
            miniBatch_gradients.append(miniBatch_grad)     # Append by mini-batches
            miniBatch_count += num_nodes
        gradients = miniBatch_gradients        

    return training_loss_batch, gradients, xtraj, utraj, ztraj



def grnn_d_testing(env, list_adj_matrix_test, gnn_model, x0s_test, z_tminus1_test, timeHorizon, noise_std, testing_weights, activation_func_name):
    # adj = sp.block_diag(list_adj, format='csr')
    # if shiftop == 'nadj':
    #     # Normalized adjacency
    #     L_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    #     L_norm_tuple = preprocess_adj(adj)
    # else:
    #     # Normalized Laplacian
    #     L_norm = sp.eye(adj.shape[0]) - normalize_adj(adj)
    #     L_norm_tuple = normalize_laplacian(adj)
    test_sample_times_num_nodes = x0s_test.shape[0]
    test_sample_size = len(list_adj_matrix_test)
    
    xtraj_test, utraj_test, ztraj_test = env.get_trajectory(list_adj_matrix_test, gnn_model, x0s_test, z_tminus1_test, timeHorizon, noise_std, testing_weights, activation_func_name)
    testing_cost_over_horizon = env.gcnn_cost(xtraj_test, utraj_test, timeHorizon, test_sample_size)   
    testing_loss_batch = testing_cost_over_horizon

    
    return testing_loss_batch, xtraj_test, utraj_test, ztraj_test


    




def get_weights(model, system_dim, input_dim, hidden_dim, output_dim, num_nodes, num_layers=2, w_multiplier=1.0, scale_nbr=2.0):
    weights = []
    if 'gcnn_c' in model.__name__:
    ### Set up the input, hidden and output layers' dimensions
        for ith_layer in range(num_layers):
            if ith_layer == 0:    # The 1st layer: (input_dim, hidden_dim)
                input_dim_layer = input_dim
                output_dim_layer = hidden_dim
            elif ith_layer == num_layers - 1:   # The last layer (Lth layer): (hidden_dim, output_dim) 
                input_dim_layer = hidden_dim
                output_dim_layer = output_dim
            else:     # The 2nd-(L-1)th layer: (hidden_dim, hidden_dim)
                input_dim_layer = hidden_dim
                output_dim_layer = hidden_dim
        
            wi1_true = w_multiplier * np.random.uniform(0, 1, (input_dim_layer, output_dim_layer))  # Layer i, coefficient 1 (\Theta_{1i}^l)
            wi1_gen = tf.Variable(wi1_true)
            weights.append(wi1_gen)
    
        # Setup and stack all the other weights from the ith node and its neighbors (to train the model using the weights with the same size, we simply add all the other nodes's weights.)
            for i in range(num_nodes):     
                wi2_true = w_multiplier * np.random.uniform(0, 1, (input_dim_layer, output_dim_layer)) * scale_nbr  # Layer i, coefficient 2 (\Theta_{2ii}^l)
                wi2_gen = tf.Variable(wi2_true)
                weights.append((wi2_gen))   # .extend: extend a list by appending all the elements from an iterable (like another list, tuple, or string) to the end of the list.
    
    elif 'grnn' in model.__name__:
        # The dimension of \Theta_{1i}\in\mathbb{R}^{p\times p}
        Theta1_row = hidden_dim
        Theta1_col = hidden_dim
        
        # The dimension of \Theta_{2i}\in\mathbb{R}^{n\times p}
        Theta2_row = system_dim
        Theta2_col = hidden_dim

        # The dimension of \Theta_{3i}\in\mathbb{R}^{n\times p}
        Theta3_row = system_dim
        Theta3_col = hidden_dim
        
        # The dimension of \Theta_{4i}\in\mathbb{R}^{p\times m}
        Theta4_row = hidden_dim
        Theta4_col = system_dim

        wi1_true = w_multiplier * np.random.uniform(0, 1, (Theta1_row, Theta1_col))  # Node i, coefficient 1 (\Theta_{1i})
        wi2_true = w_multiplier * np.random.uniform(0, 1, (Theta2_row, Theta2_col))  # Node i, coefficient 2 (\Theta_{2i})
        wi3_true = w_multiplier * np.random.uniform(0, 1, (Theta3_row, Theta3_col))  # Node i, coefficient 3 (\Theta_{3i})
        wi4_true = w_multiplier * np.random.uniform(0, 1, (Theta4_row, Theta4_col))  # Node i, coefficient 4 (\Theta_{4i})
        
        # Set these parameters as weights and collect them in a list here
        wi1_gen = tf.Variable(wi1_true)
        wi2_gen = tf.Variable(wi2_true)
        wi3_gen = tf.Variable(wi3_true)
        wi4_gen = tf.Variable(wi4_true)
        weights.append(wi1_gen)
        weights.append(wi2_gen)
        weights.append(wi3_gen)
        weights.append(wi4_gen)

    return weights







def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
