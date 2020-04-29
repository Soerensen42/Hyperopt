import pickle
import sys
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from FCN.py import *




def objective(args):
    #handling the arguments
    nLayers, batch_size, learning_rate, l1, l2, l3, l4, l5 = args
    
    #make sure parameters are the type they should if needed
    nLayers=int(nLayers)
    batch_size=int(2**batch_size)
    l1=int(l1)
    l2=int(l2)
    l3=int(l3)
    l4=int(l4)
    l5=int(l5)
    
    #Include crashing networks for hyperopt to learn
    #replace Network with your network function 
    try:
        res = Iteration(nLayers, batch_size, learning_rate, l1,l2,l3,l4,l5)
        return(res) 
    except:
        return(0.5)

#loading into the search history, if it exists, change path to match Trials location
trials_one=Trials()
if os.path.exists('/path/Trials.p'):
    trials_one = pickle.load(open("path/Trials.p", "rb"))


#Searchspace, for more options look at the hyperopt documentation     
space = [hp.quniform('nLayers', 1, 5,1),
            hp.quniform('batch_size', 1, 11,1),hp.loguniform('learning_rate', -9.21034, 0),
            hp.quniform('l1', 2, 256,1),hp.quniform('l2', 2, 256,1),
            hp.quniform('l3', 2, 256,1),hp.quniform('l4', 2, 256,1),hp.quniform('l5', 2, 256,1)]

#setting gpu device, only needed for this specific network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#print(device) #debug

criterion = nn.CrossEntropyLoss()

#running one Iteration, before consulting the updated Trials object
best = fmin(objective,space,trials=trials_one,algo = tpe.suggest,max_evals=len(list(trials_one))+1) 

if os.path.exists('/path/Trials.p'):
    trials_old = pickle.load(open("/path/Trials.p", "rb"))
else:
    trials_old=Trials()
       
#Updating the trials object
trials_new=trials_from_docs(list(trials_old)+[list(trials_one)[len(list(trials_one))-1]])   
pickle.dump(trials_new, open("/path/Trials.p", "wb"))
