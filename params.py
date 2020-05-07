import numpy as np
def Set_params():
    Parameters={}
    #defines the Searchspace hyperopt is looking through
    #hp.uniform('name',a,b) draws a value soemwhere between a and b
    #hp.quniform('name',a,b,c) draws from [a,1+c,a+2c,...,b-c,b] This has to fit, so a+Xc=b must be true
    #hp.loguniform('name',np.log(a),np.log(b)) draws logarithmic between values a and b, but is very heavy 
    #tending to a (since its a logarithmic function) !Setting a as 0 will get you an error!
    Parameters["space"]= [hp.quniform('nLayers', 0, 5,1), #Amount Layers the FCN uses.Number of actual layers is nLayers + 1
                          hp.loguniform('batch_size', np.log(2), np.log(2000 ), #The Size of the "Batches" that are ran through the newtork at the same time
                          hp.loguniform('learning_rate', np.log(0.0001), np.log(1)),#how fast the Network learns aka how fast it accepts new ideas and purposely makes mistakes to possibly make a new discovery
                          hp.quniform('Nodes', 1, 256,8)#Amount of nodes in each Layer
                          ]
    Parameters["Quick"]=1 #Activating(1) or deactivating(0) Quick mode. This will Train etc. only with 2000 Events
                          #recommended for setup and debugging, very fast              
    Parameters["data_location"]="./data"  #Path where all 3 files are saved       
    Parameters["Epochs_FCN"]=10  #Amound of Epochs the FCN Trains for
    Parameters["Iterations"]= 100 #Amount of total iterations the Optimization progress makes, set to 1000 if total number is irrelevant 
