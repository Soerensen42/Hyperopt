import pickle
import sys
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK,trials_from_docs
from FCN import *
import os
import numpy as np

####################Input the parameters#########################
def set_params():
    #defines the Searchspace hyperopt is looking through
    #hp.uniform('name',a,b) draws a value soemwhere between a and b
    #hp.quniform('name',a,b,c) draws from [a,1+c,a+2c,...,b-c,b] This has to fit, so a+Xc=b must be true
    #hp.loguniform('name',np.log(a),np.log(b)) draws logarithmic between values a and b, but is very heavy 
    #tending to a (since its a logarithmic function) !Setting a as 0 will get you an error!
    Parameters["space"]= [hp.quniform('nLayers', 0, 5,1), #Amount Layers the FCN uses.Number of actual layers is nLayers + 1
                          hp.loguniform('batch_size', np.log(2), np.log(2000 )), #The Size of the "Batches" that are ran through the newtork at the same time
                          hp.loguniform('learning_rate', np.log(1), np.log(0.0001)),#how fast the Network learns aka how fast it accepts new ideas and purposely makes mistakes to possibly make a new discovery
                          hp.quniform('Nodes', 1, 256,8)]#Amount of nodes in each Layer
                          
    Parameters["Quick"]=1 #Activating(1) or deactivating(0) Quick mode. This will Train etc. only with 2000 Events
                          #recommended for setup and debugging, very fast              
    Parameters["data_location"]="./data"  #Path where all 3 files are saved       
    Parameters["Epochs_FCN"]=10  #Amound of Epochs the FCN Trains for
    Parameters["Iterations"]= 100 #Amount of total iterations the Optimization progress makes, set to 1000 if total number is irrelevant 

####################Start of the Script##########################

Parameters = {}
set_params()

def objective(args):
    #include Crashes in search history (for things like hardware limitations)
    try:
        res = Iteration(*args)
        return(res) 
    except:
        #forwarding worst possible result to hyperopt when training crashes (since im optimizing for 1-Accuracy its 1)
        return(1)

#loading into the search history if it exists.
trials_one=Trials()
if os.path.exists('./History/Trials.p'):
    trials_one = pickle.load(open("./History/Trials.p", "rb"))

if Parameters["Iterations"] < len(list(trials_one)):
    sys.exit()
#Moving everything to gpu, only needed for this specific network, maybe useful for others
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#print(device) #debug

#Setting The Criterions, specific to this Network
criterion = nn.CrossEntropyLoss()

#running one Iteration of the Network
best = fmin(objective,Parameters["space"],trials=trials_one,algo = tpe.suggest,max_evals=len(list(trials_one))+1) 

#Calling the Trials object in case there are new entries (there should be)
if os.path.exists('./History/Trials.p'):
    trials_old = pickle.load(open("./History/Trials.p", "rb"))
else:
    #if no Trials object exists: Create a new one
    trials_old=Trials()
       
#Adding the new search to the Trials object and saving it in the shared Trials file
trials_new=trials_from_docs(list(trials_old)+[list(trials_one)[len(list(trials_one))-1]])   
pickle.dump(trials_new, open("./History/Trials.p", "wb"))
print("progress:",len(list(Trials)),"/",Parameters["Iterations"]) #show progress

#Saving the best params for in optimization use and result viewing
with open('./Results.txt', 'w') as f:
    json.dump(best, f)
