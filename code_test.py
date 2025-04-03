import tensorflow as tf

# for ith_layer in range(len(weights)//(num_nodes+1)):  # Number of layers is half of the length of weights, since for each layer, there are two weights.
#     w1i_l = weights[ith_layer*(num_nodes+1)]   # \Theta_{1i}^l
#     w2ij_l = weights[ith_layer*(num_nodes+1)+1:(ith_layer+1)*(num_nodes+1)]    # \Theta_{2ij}^l      ##############We stop at here yesterday!

#     #######################################################################################################################
#     # The distributed setting, where we compute all batches and all nodes' local GCNNs for all layers.
#     term1 = X_iprev[:, tf.newaxis] @ w1i_l
    
#     sum_Sij_times_X_j_times_w2ij_l = []
#     neighbor_contribution = 0
#     batch_size = X_jprev.shape[0] // num_nodes
#     for minibatch in range(batch_size):
#         X_j_miniBatch = tf.reshape(X_jprev[minibatch*num_nodes:(minibatch+1)*num_nodes], ((num_nodes, 1, 2)))
#         Sij_miniBatch = Sij[minibatch*num_nodes:(minibatch+1)*num_nodes, minibatch*num_nodes:(minibatch+1)*num_nodes]
#         for i in range(num_nodes):
#             for j in range(num_nodes):   # Which weights is it in w2ij_l
#                 w2ij_l_i_miniBatch = w2ij_l[j][minibatch*num_nodes:(minibatch+1)*num_nodes]    # The ith weights in w2ij_l and extract out one minibatch
#                 X_j_times_w2ij_l_miniBatch = X_j_miniBatch[j] @ w2ij_l_i_miniBatch[i]
#                 Sij_times_X_j_times_w2ij_l_miniBatch = Sij_miniBatch[i, j] @ X_j_times_w2ij_l_miniBatch
#                 neighbor_contribution += Sij_times_X_j_times_w2ij_l_miniBatch
#             sum_Sij_times_X_j_times_w2ij_l.append(neighbor_contribution)
#             neighbor_contribution = 0
        
#     sum_Sij_times_X_j_times_w2ij_l = tf.convert_to_tensor(sum_Sij_times_X_j_times_w2ij_l, dtype = tf.float64)


#     hi_l = term1 + sum_Sij_times_X_j_times_w2ij_l
#     hi_l = tf.squeeze(hi_l, 1)   #  Removes dimensions of size 1 from a tensor. Now, the shape becomes (10000, 10).
#     #######################################################################################################################
#     Xi_l = act(hi_l)

#     # Set up the inputs for the next layer.
#     if ith_layer < len(weights)/(num_nodes+1)-1:
#         X_iprev = Xi_l
#         X_jprev = X_iprev
#         SijX_jprev = tf.sparse.sparse_dense_matmul(Sij, X_jprev)







# for minibatch in range(batch_size):
#     X_j_miniBatch = tf.reshape(X_jprev[minibatch*num_nodes:(minibatch+1)*num_nodes], ((num_nodes, 1, 2)))
#     Sij_miniBatch = Sij[minibatch*num_nodes:(minibatch+1)*num_nodes, minibatch*num_nodes:(minibatch+1)*num_nodes]
#     for i in range(num_nodes):
#         for j in range(num_nodes):   # Which weights is it in w2ij_l
#             w2ij_l_i_miniBatch = w2ij_l[j][minibatch*num_nodes:(minibatch+1)*num_nodes]    # The ith weights in w2ij_l and extract out one minibatch
#             X_j_times_w2ij_l_miniBatch = X_j_miniBatch[j] @ w2ij_l_i_miniBatch[i]
#             Sij_times_X_j_times_w2ij_l_miniBatch = Sij_miniBatch[i, j] @ X_j_times_w2ij_l_miniBatch
#             neighbor_contribution += Sij_times_X_j_times_w2ij_l_miniBatch
#         sum_Sij_times_X_j_times_w2ij_l.append(neighbor_contribution)
#         neighbor_contribution = 0

# sum_Sij_times_X_j_times_w2ij_l = tf.convert_to_tensor(sum_Sij_times_X_j_times_w2ij_l, dtype = tf.float64)




















# num_nodes = x0s.shape[1]
# list_x0s = [x0 for x0 in x0s]   # transform the np.array into a list
# x0_in = tf.concat(list_x0s, 0)  # Now, we combine the first dimension with the second one, and the initial states x0s are now with the shape (batch_size*num_nodes, n, 1).
#                                         # Here, each elements in list_x0s are stacked one after another along axis 0, i.e., along the batch_size. For example, the first batch_size is to store the nodes from 0 to N-1 and then we stack the second batch_size, which involves the second batch of nodes from 0 to N-1
        
# adj_matrix = sp.block_diag(list_adj_matrix, format='csr')     # Copy the same adjacency matrix S by 100 times, and we will have a 
# adj_matrix_shape = sparse_to_tuple(adj_matrix)

# batch_size_times_num_nodes = x0_in.shape[0]
# # x = tf.zeros(shape = (batch_size_times_num_nodes, 2, self.n, 1))
# # u = tf.zeros(shape = (batch_size_times_num_nodes, 1, self.m, 1))
# xt = x0_in
# xt_in = tf.reshape(xt, (batch_size_times_num_nodes, env.n))

# # x[:,0] = x0_in    # The initial states are filled in the the initial time step, i.e., time t = 0, each is with \mathbb{R}^{n\times 1}
        
# ### Compute for one step and generate the control input and the next states
# ut = gcnn_model(adj_matrix_shape, xt_in, training_weights, num_nodes, activation_func_name)




### Test the gradient computation
# weights = tf.Variable(tf.random.normal((1000,2,2), dtype = tf.float32))

# loss = tf.random.normal((1000,1), dtype=tf.float32)

# with tf.GradientTape() as tape:
#     tape.watch(weights)
#     loss = tf.reduce_mean(loss, axis = -1)

# gradients = tape.gradient(loss, weights)

# gradients






# Create a weight variable
weights = tf.Variable(tf.random.normal((1000, 2, 2)), dtype=tf.float32)

# Input and target
inputs = tf.random.normal((1000, 2, 2))
targets = tf.random.normal((1000, 1))

with tf.GradientTape() as g:
    g.watch(weights)  # Watch weights (not strictly needed if weights are Variables)
    
    # Model computation using weights
    predictions = tf.reduce_mean(tf.matmul(inputs, weights), axis=(1, 2), keepdims=True)
    
    # Compute loss (MSE)
    loss_func = tf.keras.losses.MeanSquaredError()
    loss = tf.reduce_mean(loss_func(targets, predictions))

# Compute gradient
grad = g.gradient(loss, weights)
print(grad)  # Should not be None




with tf.GradientTape() as tape:            
    tape.watch(training_weights)
    ### Step 4: get the trajectory of all the nodes during a small time horizon in a distributed manner
    xtraj, utraj = env.get_trajectory(list_adj_matrix, gcnn_model, x0s, timeHorizon, training_weights, activation_func_name)
    cost_over_horizon = env.gcnn_cost(xtraj, utraj, timeHorizon, batch_size)   # Give x and u, we compute the cost value for all nodes and mini-batches. Here, x involves the all the past time (xt) and the final value (xT), while u only involves the past values (ut)
    training_loss_batch = cost_over_horizon
    gradients = tape.gradient(training_loss_batch, training_weights)

