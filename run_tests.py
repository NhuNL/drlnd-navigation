import os

#Hyperparameters grid
algos = ['ddqn']
batch_sizes = [32, 64, 128, 256]
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
batch_norms = ['True', 'False']
eps_inits = [1.0, 0.5]
eps_decays = [0.99, 0.995]

for algo in algos:
    for i_batch_size, batch_size in enumerate(batch_sizes):
        for i_learning_rate, learning_rate in enumerate(learning_rates):
            for i_batch_norm, batch_norm in enumerate(batch_norms):
                for i_eps_init, eps_init in enumerate(eps_inits):
                    for i_eps_decay, eps_decay in enumerate(eps_decays):
                        #Grid index for test name interpretability
                        grid_index = str(i_batch_size)+str(i_learning_rate)+str(i_batch_norm)+str(i_eps_init)+str(i_eps_decay)
                        test_name = algo+'-'+grid_index+' '
                        #Optional arguments
                        kwarg_algo = '--algo='+str(algo)+' '
                        kwarg_batch_size = '--batch_size='+str(batch_size)+' '
                        kwarg_lr = '--lr='+str(learning_rate)+' '
                        kwarg_BN = '--BN='+str(batch_norm)+' '
                        kwarg_eps_init = '--eps_init='+str(eps_init)+' '
                        kwarg_eps_decay = '--eps_decay='+str(eps_decay)
                        
                        #concatenate kwargs
                        kwargs = kwarg_algo + kwarg_batch_size + kwarg_lr + kwarg_BN + kwarg_eps_init + kwarg_eps_decay
                        
                        run_line = 'python main.py -t '+ test_name + kwargs
                        
                        os.system(run_line)