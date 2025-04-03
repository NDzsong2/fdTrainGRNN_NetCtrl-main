import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from gnn_models.gcnns import gcnn_c, gcnn_local_d, gcnn_share_d
from gnn_models.grnns import grnn_d
# import data_generation as dg
# random_x0, get_trajectory
from env.training_env import *
# from env.training_env import generate_graph, generate_dynamics, consensus_matrix
from training_utils import get_weights, grnn_d_training, gcnn_d_training, grnn_d_testing
# minibatch_gnn, minibatch_gnn_test, convert_sparse_matrix_to_sparse_tensor, 
from optimizers import D_SGD


from results_plot import plot_training_results


np.random.seed(42)



def main(args):
    
    ### Set up some basic parameters for data generation, training and model construction
    architecture = 'distributed_grnn'   # Setup the training type: distributed or centralized
    w_multiplier = 1
    activation_func_name = 'tanh'
    batch_size = 100  # Number of mini-batches in one epoch
    opt = 'dsgd' # 'dadam'
    num_layers = 1    
    shiftop = 'nlap'   # Select the type of the graph shift operator S
    gtype = 'gaussian_random_partition'   # Select a graph type
    directedG = True    # Set the graph as a directed graph
    preset = True

    num_nodes = 10    # Number of subsystems
    system_dim = 2    # Each subsystem's dimension
    input_dim = system_dim    # Input layer dimension  
    hidden_dim = system_dim   # Hidden layer dimension  
    output_dim = system_dim   # Output layer dimension  
    noise_std = 0.1
    timeHorizon = 10
    test_samples = 20     # Number of samples generated for testing
    num_topologies = 1   # Total length of the system evolution
    num_controllers = 1
    log_interval = 10
    num_epoch = 200    # Length of epochs
    lr = 0.001   # Learning rate
     
    x0_mean = 2 
    x0_std = 1


    ### Select the device to run the code
    # device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    # print(f"Using device: {device}. ")

    ### Select a model (centralized GCNN/distributed GCNN)
    gnn_model = grnn_d if architecture == 'distributed_grnn' else gcnn_c
    print(f"Performing {architecture} training: ")


    #########################################################################################################
    # This is the main part (custom training loop) for the distributed training of the GCNN model 
    #########################################################################################################
    start_time = time.time()

    ### Create some containers to store our data 
    num_log_points = int(num_epoch / log_interval)
    rel_costs_table = tf.zeros((num_topologies, num_controllers, num_log_points))    # Construct a torch.Tensor (50,5,75), which is a multi-dimensional matrix containing elements of a single data type.
    autonomous_costs = tf.zeros(num_topologies)    # 50
    num_sparse_edges = tf.zeros(num_topologies)    # 50
    num_env_edges = tf.zeros(num_topologies)       # Each element count the number of the edges according to the non-zero elements for the current topology S. Here, it is 50.
                                                        # num_topologies - Number of the topologies. This is the same as the time stamps. here it is 50. 
    
    train_x0s = [] # Write down the train x0s to reuse for training other types of models
    training_loss = np.zeros((num_epoch, num_nodes))
    testing_loss = np.zeros((num_epoch, num_nodes))
    results = pd.DataFrame([], columns=['epoch', 'training_loss', 'testing_loss'])
    # results = pd.DataFrame([], columns=['epoch', 'training_loss'])
    
    # with tf.device(device):
    for ith_topology in range(num_topologies):
        ### Step 1: generate a graph (topology) and construct its normalized adjacency matrix S
        env, S = generate_training_env(num_nodes, system_dim, k=8, v=1, p=0.8, gtype=gtype, directedG=directedG, gseed=3, A_norm=0.995, B_norm=1, AB_hops=3)
        # num_env_edges[ith_topology] = tf.reduce_sum(tf.cast(S != 0, tf.int32)).numpy().item()
        # Generate the consensus matrix W for online training
        c_mtx = env.consensus_matrix(S)
        
        # Set a list of adjacency matrices
        list_adj_matrix_train = [S]*batch_size
        list_adj_matrix_test = [S]*np.min((batch_size, test_samples))

        # Get the weights for a certain node i (\Theta_{1i} and \Theta_{2ij}, j\in\mathcal{I}_{N})
        weights = get_weights(gnn_model, system_dim, input_dim, hidden_dim, output_dim, num_nodes, num_layers=num_layers, w_multiplier=w_multiplier, scale_nbr=2.0)   
        if architecture == 'distributed_grnn':
            weights = [np.repeat(w[tf.newaxis], num_nodes, axis=0) for w in weights]    # This is a local copy of the weights for all nodes. 
                                    # np.repeat: Repeat elements of an array along a specified axis. This copy \Theta_{1i} num_nodes times, and \Theta_{2ij}, j\in\mathcal{I}_{N} num_nodes times.
        weights = [tf.Variable(w, trainable=True) for w in weights]   # Set all the weights as trainable parameters, and the setting the weights as the initial values

        ### Set up the optimizer (D-SGD optimizer)
        optimizer = D_SGD(alpha=lr, beta=0.001)
        # adj_sp = convert_sparse_matrix_to_sparse_tensor(adj)
        
        # Now, start our training process!
        for epoch in range(num_epoch):
            ### Step 2: generate (random) initial state datasets (initial states X^0 = X(t)). Here, we create a batch of states for each individual node, each is with the shape (batch_size, num_nodes, 1, p)
            if epoch == 0:
                initial_x0s = env.random_x0(batch_size, num_nodes, x0_mean, x0_std, seed=None)  
                list_x0s = [x0 for x0 in initial_x0s]   # transform the np.array into a list
                x0s = tf.concat(list_x0s, 0)    # Now, we combine the first dimension with the second one, and the initial states x0s are now with the shape (batch_size*num_nodes, n, 1).
                                                # Here, each elements in list_x0s are stacked one after another along axis 0, i.e., along the batch_size. For example, the first batch_size is to store the nodes from 0 to N-1 and then we stack the second batch_size, which involves the second batch of nodes from 0 to N-1
                z_tminus1 = tf.zeros(shape = (batch_size*num_nodes, env.p), dtype=tf.float64)
                train_x0s.append(x0s)
            else:
                x0s = new_x0
                z_tminus1 = new_z_tminus1
                train_x0s.append(x0s)

            if architecture == 'centralized_gcnn':
                training_weights = weights
            elif architecture == 'distributed_grnn':
                training_weights = [tf.Variable(tf.tile(w_d, [batch_size, 1, 1])) for w_d in weights] # We get 21 weights, each is a tensor with shape <batch_size*num_nodes,2,2>.
                
            
            ### Step 3: compute the training losses and gradients over all weights and nodes
            if 'grnn_d' in gnn_model.__name__:                                        
                training_loss_batch, gradients, xtraj, utraj, ztraj = grnn_d_training(env, list_adj_matrix_train, gnn_model, x0s, z_tminus1, timeHorizon, noise_std, training_weights, activation_func_name)
                new_x0 = xtraj[:, timeHorizon]     # Reset the initial states to the latest xtraj
                new_z_tminus1 = ztraj[:,timeHorizon]   # Reset the Z_tminus1 to the latest ztraj
            elif 'gcnn_c' in gnn_model.__name__:
                training_loss_batch, gradients = gcnn_d_training(env, list_adj_matrix_train, gnn_model, x0s, timeHorizon, training_weights, activation_func_name)
                
            ### Step 4: store the training loss over all nodes and the time horizon
            allNodes_training_loss = tf.reduce_sum(tf.reshape(training_loss_batch, (batch_size, num_nodes)), axis=0, keepdims=True) / batch_size
            training_loss[epoch] = allNodes_training_loss  


            ### Step 5: update the weights online using the computed gradients in Step 3
            if architecture == 'centralized_gcnn':
                pass
            elif architecture == 'distributed_grnn':   # decentralized + consensus
                gradients_batch = list(zip(*gradients))        # transposes the first two dimensions
                gradients_batch = [(tf.reduce_mean(grad, 0)) for grad in gradients_batch]          # sum over minibatches
                weights = optimizer.apply_gradients(zip(gradients_batch, weights), c_mtx)    # Update each weight based on distributed optimization updates (decentralization + consensus)
                

            ### Step 6: test the model we learned by computing its testing loss
            # Step 6.1: set up the weights
            if architecture == 'centralized_gcnn':
                testing_weights = weights
            elif architecture == 'distributed_grnn':
                testing_weights = [tf.Variable(tf.tile(w_d, [np.min((batch_size, test_samples)), 1, 1])) for w_d in weights]
                
            if epoch == 0:
                x0s_test_samples = env.random_x0(np.min((batch_size, test_samples)), num_nodes, x0_mean, x0_std, seed=None)
                list_x0s_test = [x0_test for x0_test in x0s_test_samples]   # transform the np.array into a list
                x0s_test = tf.concat(list_x0s_test, 0)
                z_tminus1_test = tf.zeros(shape = (np.min((batch_size, test_samples))*num_nodes, env.p), dtype=tf.float64)
            else:
                x0s_test = new_x0_test
                z_tminus1_test = new_z_tminus1_test

            # Step 6.2: compute the testing loss over all nodes and the time horizon
            testing_loss_batch, xtraj_test, utraj_test, ztraj_test = grnn_d_testing(env, list_adj_matrix_test, gnn_model, x0s_test, z_tminus1_test, timeHorizon, noise_std, testing_weights, activation_func_name)
            new_x0_test = xtraj_test[:, timeHorizon]     # Reset the initial states to the latest xtraj
            new_z_tminus1_test = ztraj_test[:,timeHorizon]

            # Step 6.3: store the testing loss over all nodes and the time horizon
            allNodes_testing_loss = tf.reduce_sum(tf.reshape(testing_loss_batch, (np.min((batch_size, test_samples)), num_nodes)), axis=0, keepdims=True) / batch_size
            testing_loss[epoch] = allNodes_testing_loss  
                


            ### Step 7: Now, we print/plot and save the obtained results (training and testing losses)
            print(f"Epoch: {epoch}: ", end="")
            print(", ".join([f"Training Loss{i+1}: {loss_tr:.5f}" for i, loss_tr in enumerate(training_loss[epoch])])) 
            print(f"           " + ", ".join([f"Testing Loss{i+1}: {loss_te:.5f}" for i, loss_te in enumerate(testing_loss[epoch])]))
            



            
    #     epoch_res = pd.DataFrame({
    #         'epoch': epoch,
    #         'train_loss': loss_tr[epoch]
    #         # 'test_loss' : loss_te[epoch],
    #     }, index=[epoch])
    #     res = pd.concat([res, epoch_res], ignore_index=True)    # pd.concat(): concatenates two or more DataFrames along a particular axis (default is along the rows, i.e., axis=0).
    #                                                             # ignore_index=True: resets the index after concatenation, meaning that the new DataFrame will have a new integer index starting from 0.
    #     res.to_csv("./output/regression_chebnet_loss_trajectory_{}_{}_{}_{}_wm{}_{}_preset{}_l{}.csv".format(
    #         opt, shiftop, gtype, feattype, w_multiplier, act_str, int(preset), num_layers))
    # print("Training completed! Total runtime {:.3f}".format(time.time() - start_time))










if __name__ == '__main__':
    main(sys.argv[1:])
