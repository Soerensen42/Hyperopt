import pickle
import sys
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK,trials_from_docs
from FCN import * #implements the FCN model change, when swapping networks
import os
import numpy as np
import json
import time

####################Input the parameters#########################
Parameters={}
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
Parameters["WorstCase"]=1 #return the worst possible outcome for your network
####################Start of the Script##########################
Accuracy=0
def objective(args):
    #include Crashes in search history (for things like hardware limitations)
    try:
        Accuracy = Iteration(*args,Parameters)
        return(Accuracy) 
    except:
        #forwarding worst possible result to hyperopt when training crashes (since im optimizing for 1-Accuracy its 1)
        Accuracy=Parameters["WorstCase"]
        return(Parameters["WorstCase"])

#loading into the search history if it exists.
trials_one=Trials()
if os.path.exists('./History/Trials.p'):
    trials_one = pickle.load(open("./History/Trials.p", "rb"))

#running one Iteration of the Network
best = fmin(objective,Parameters["space"],trials=trials_one,algo = tpe.suggest,max_evals=len(list(trials_one))+1) 
best["Accuracy"]=1-Accuracy
#Calling the Trials object in case there are new entries (there should be)
if os.path.exists('./History/Trials.p'):
    trials_old = pickle.load(open("./History/Trials.p", "rb"))
else:
    #if no Trials object exists: Create a new one
    trials_old=Trials()
       
#Adding the new search to the Trials object and saving it in the shared Trials file
trials_new=trials_from_docs(list(trials_old)+[list(trials_one)[len(list(trials_one))-1]])   
pickle.dump(trials_new, open("./History/Trials.p", "wb"))
print("progress:{}/100".format(len(list(trials_new)))) #show progress

best_old={"Accuracy":0}
if os.path.exists('./Results.txt'):
    with open('./Results.txt') as f:
        best_old = json.load(f)
#Saving the best params for in optimization use and result viewing
save=0
if best_old["Accuracy"]<best["Accuracy"]:
    while saved==0:
        try:
            with open('./Results.txt', 'w') as f:
                json.dump(best, f)
            saved=1    
        except:
            time.sleep(30)
