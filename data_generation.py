import numpy as np
import tensorflow as tf
from gnn_models.gcnns import *
import env.training_env as env

# def get_data(train_samples, test_samples, num_nodes, input_dim, feattype='mix'):
#     pass


class data_generator():
    def __init__(self, S, A, B, Q, R, QT):
        self.S, self.A, self.B, self.Q, self.R, self.QT = S, A, B, Q, R, QT
        self.N = self.S.size(0)
        self.n = int(self.A.size(0) / self.S.size(0)) 
        self.m = int(self.B.size(1) / self.S.size(0))
        self.QT = self.P



    

    def get_trajectory(self, model, env, x0s, T):
        # Their architecture takes in an additional time dimension
        batch_size = x0s.size(0)
        N = x0s.size(1)
        x = tf.zeros(shape = (batch_size, T+1, N, 1))
        u = tf.zeros(shape = (batch_size, T, N, 1))
        x[:,0] = x0s
        for t in range(T):
            ut = model.forward(x[:,t].clone().reshape(batch_size, 1, 1, N)).reshape(batch_size, N, 1)  # Call this forward method to specify the input type
            x[:, t+1] = env.step(x[:, t].clone(), ut)
            u[:, t] = ut
        return x, u




