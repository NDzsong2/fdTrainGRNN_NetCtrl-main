import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import numpy as np

tf.random.set_seed(42)


def plot_hist(grad, bins=100):  # For debugging
    grad_flat = tf.reshape(grad, [-1])
    plt.hist(grad_flat, bins=bins)


### Here, we manually create the D-SGD optimizer, i.e., decentralize + consensus
class D_SGD:
    def __init__(self, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def type_change(self, c_mtx):
        indices = np.array(np.nonzero(c_mtx)).T
        values = c_mtx[tuple(indices.T)]
        c_shape = c_mtx.shape[0]
        c_mtx_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=(c_shape, c_shape))

        return c_mtx_sparse

    def apply_gradients(self, grads_and_weights, c_mtx):
        """ Assuming that node is the first dim
        """
        c_mtx_sparse = self.type_change(c_mtx)
        weights = []
        for grad, w in grads_and_weights:    # Update the weights with the corresponding gradients
            # grad, w = self.dim_check(grad, w)
            grad_std = tf.math.reduce_std(grad)
            local_sg = (grad+tf.random.normal(grad.shape, mean=0, stddev=0.0*grad_std, dtype=tf.float64))
            
            ### Decentralized update
            w = w - self.alpha*local_sg

            ### Consensus step update
            w_flat = tf.reshape(w, (grad.shape[0], -1))
            w_consensus = tf.sparse.sparse_dense_matmul(c_mtx_sparse, w_flat)
            w = tf.reshape(w_consensus, (10, 2, 2))
        
            ### Now, update the weights
            weights.append(w)

        return weights








