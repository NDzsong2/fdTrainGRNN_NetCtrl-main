import tensorflow as tf


    

def grnn_d(S_in, t, zt, xt, weights, num_nodes, activation_func_name):
    if activation_func_name == 'lrelu':
        act = tf.nn.leaky_relu
    elif activation_func_name == 'relu':
        act = tf.nn.relu
    elif activation_func_name == 'sigmoid':
        act = tf.nn.sigmoid
    elif activation_func_name == 'tanh':
        act = tf.nn.tanh
    
    coords, values, shape = S_in
    coords = tf.cast(coords, tf.int64)
    adj_ts = tf.sparse.SparseTensor(indices=coords, values=values, dense_shape=shape)
    
    X_in = xt
    Z_tminus1 = zt
    SX_in = tf.sparse.sparse_dense_matmul(adj_ts, X_in)     # 1000*2
    wi1, wi2, wi3, wi4 = weights[0:4]

    ht = Z_tminus1[:, tf.newaxis] @ wi1 + X_in[:, tf.newaxis] @ wi2 + SX_in[:, tf.newaxis] @ wi3  # We need to update the model multiple times?
    ht = tf.squeeze(ht, 1)
    Zt = act(ht)

    # Compute the control inputs
    Ut = Zt[:, tf.newaxis] @ wi4
    Ut = tf.squeeze(Ut, 1)
    
    return Ut, Zt



