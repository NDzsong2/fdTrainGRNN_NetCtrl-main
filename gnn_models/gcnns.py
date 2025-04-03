import tensorflow as tf


def gcnn_c(S_in, xt, weights, activation_func_name):
    # Different types of activation function 
    if activation_func_name == 'lrelu':
        act = tf.nn.leaky_relu
    elif activation_func_name == 'sigmoid':
        act = tf.nn.sigmoid
    elif activation_func_name == 'tanh':
        act = tf.nn.tanh
    
    # Set up the shape of the adjacency matrix S
    coords, values, shape = S_in
    coords = tf.cast(coords, tf.int64)    # Cast a tensor to a different data type (dtype).

    # Create and save the adjacency matrix S tensor (indices, non-zero elements and shape)
    # Indices: A 2D tensor containing the indices of the non-zero elements in the sparse tensor. The shape of the indices tensor is (N, rank), where N is the number of non-zero elements and rank is the number of dimensions of the tensor.
    # Values: A 1D tensor containing the non-zero values corresponding to the indices.
    # Shape: A 1D tensor containing the shape of the sparse tensor. This defines the total number of elements in the tensor (including zeros).
    S = tf.sparse.SparseTensor(indices=coords, values=values, dense_shape=shape)
    X_prev = xt   # Input dataset X^{l-1}
    SX_prev = tf.sparse.sparse_dense_matmul(S, X_prev)   # SX^{l-1}
    for i in range(len(weights)//2):    # Number of layers is half of the length of weights, since for each layer, there are two weights.
        w1_l, w2_l = weights[i*2:(i+1)*2]
        #######################################################################################################################
        # The centralized setting
        h_l = X_prev @ w1_l + SX_prev @ w2_l # This is the lth layer's GCNN expression: Xl = sigma_l(X^{l-1}\Theta_0^{l}+SX^{l-1}\Theta_1^{l})          
        #######################################################################################################################
        X_l = act(h_l)
        
        # Set up the inputs for the next layer.
        if i < len(weights)/2-1:   
            X_prev = X_l
            SX_prev = tf.sparse.sparse_dense_matmul(S, X_prev)
    return X_l






# This distributed function is to create the GCNN model for node i only (since the node number has already been involved in the first axis together with the batch_size) 
# In total, we have 100 batches and 100 nodes.
def gcnn_local_d(S_in, xt, weights, num_nodes, activation_func_name):  
    # Different types of activation function 
    if activation_func_name == 'lrelu':
        act = tf.nn.leaky_relu
    elif activation_func_name == 'sigmoid':
        act = tf.nn.sigmoid
    elif activation_func_name == 'tanh':
        act = tf.nn.tanh
    
    # Set up the shape of the adjacency matrix S
    coords, values, shape = S_in
    coords = tf.cast(coords, dtype = tf.int64)

    # Create and save the adjacency matrix S tensor by a tensor (indices, non-zero elements and shape)
    Sij = tf.sparse.SparseTensor(indices=coords, values=values, dense_shape=shape)   # Sparse matrix for stacked Sij's
    Sij = tf.sparse.to_dense(Sij).numpy()
    X_iprev = xt      # Input dataset X^{l-1}. The shape of this tensor is (2000, 2).
    X_jprev = X_iprev   # <2000, 2>
    # SijX_jprev = tf.sparse.sparse_dense_matmul(Sij, X_jprev)    # SX^{l-1} (tf.sparse.sparse_dense_matmul: perform matrix multiplication between a sparse matrix and a dense matrix)
    for ith_layer in range(len(weights)//(num_nodes+1)):  # Number of layers is half of the length of weights, since for each layer, there are two weights.
        w1i_l = weights[ith_layer*(num_nodes+1)]   # \Theta_{1i}^l
        w2ij_l = weights[ith_layer*(num_nodes+1)+1:(ith_layer+1)*(num_nodes+1)]    # \Theta_{2ij}^l     

        #######################################################################################################################
        # The distributed setting, where we compute all batches and all nodes' local GCNNs for all layers.
        term1 = X_iprev[:, tf.newaxis] @ w1i_l
        
        term2 = tf.zeros_like(term1)
        batch_size = X_jprev.shape[0] // num_nodes
        for minibatch in range(batch_size):
            # Extract current batch data
            start_idx = minibatch * num_nodes
            end_idx = (minibatch+1) * num_nodes

            # Extract the current batch of X_jprev and Sij
            X_miniBatch = tf.reshape(X_jprev[start_idx:end_idx], (num_nodes, 1, X_jprev.shape[1]))    # <20, 1, 2>
            S_miniBatch = Sij[start_idx:end_idx, start_idx:end_idx]    # (20, 20)
            
            # Compute neighbor contributions
            for i in range(num_nodes):
                neighbor_contribution = tf.zeros_like(term1[start_idx + i])
                for j in range(num_nodes):
                    if S_miniBatch[i, j] != 0:  # Only compute for actual neighbors
                        w2ij_miniBatch = tf.reshape(w2ij_l[j][i], (X_jprev.shape[1], -1))  # Get appropriate weight <2, 2>
                        neighbor_term = X_miniBatch[j] @ w2ij_miniBatch
                        neighbor_contribution += S_miniBatch[i, j] * neighbor_term
                term2 = tf.tensor_scatter_nd_update(
                    term2,
                    [[start_idx + i]],
                    [neighbor_contribution]
                )
        
        # Compute the hi_l term
        hi_l = term1 + term2
        hi_l = tf.squeeze(hi_l, 1)     # <2000, 2>

        # The output of the lth layer
        Xi_l = act(hi_l)       # <2000, 2>
        
        # Setup for next layer
        if ith_layer < len(weights)/(num_nodes+1)-1:
            X_iprev = Xi_l
            X_jprev = X_iprev

    return Xi_l














def gcnn_share_d(S_in, xt, weights, activation_func_name):
    if activation_func_name == 'lrelu':
        act = tf.nn.leaky_relu
    elif activation_func_name == 'sigmoid':
        act = tf.nn.sigmoid
    elif activation_func_name == 'tanh':
        act = tf.nn.tanh
    
    coords, values, shape = S_in
    coords = tf.cast(coords, tf.int64)
    adj_ts = tf.sparse.SparseTensor(indices=coords, values=values, dense_shape=shape)
    Ti1 = xt
    Ti2 = tf.sparse.sparse_dense_matmul(adj_ts, xt)
    for i in range(len(weights)//2):
        wi1, wi2 = weights[i*2:(i+1)*2]
        h1 = Ti1[:, tf.newaxis] @ wi1 + Ti2[:, tf.newaxis] @ wi2
        h1 = tf.squeeze(h1, 1)
        x1 = act(h1)
        if i < len(weights)/2-1:
            Ti1 = x1
            Ti2 = tf.sparse.sparse_dense_matmul(adj_ts, x1)
    return x1









