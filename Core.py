import pickle
import sys
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from FCN.py import *

def objective(args):
    #include Crashes in search history (for things like hardware limitations)
    try:
        res = Iteration(*args)
        return(res) 
    except:
        #forwarding worst possible result to hyperopt when iteration crashes
        return(0.5)

#loading into the search history if it exists. change path to match Trials location
trials_one=Trials()
if os.path.exists('./History/Trials.p'):
    trials_one = pickle.load(open("./History/Trials.p", "rb"))


#Searchspace, for more options look at the hyperopt documentation    
##########outsorce to params file###########
space = [hp.quniform('nLayers', 1, 5,1), #actual number of layers is nLayers + 1
            hp.quniform('batch_size', 1, 11,1),hp.loguniform('learning_rate', -9.21034, 0),
            hp.quniform('l1', 1, 256,8),hp.quniform('l2',  1, 256,8),
            hp.quniform('l3',  1, 256,8),hp.quniform('l4', 1, 256,8),hp.quniform('l5',  1, 256,8)]

#Moving everything to gpu, only needed for this specific network, maybe useful for others
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#print(device) #debug

#Setting The Criterions, specific to this Network
criterion = nn.CrossEntropyLoss()

#running one Iteration of the Network
best = fmin(objective,space,trials=trials_one,algo = tpe.suggest,max_evals=len(list(trials_one))+1) 

#Calling the Trials object in case there are new entries (there should be)
if os.path.exists('/path/Trials.p'):
    trials_old = pickle.load(open("./History/Trials.p", "rb"))
else:
    #if no Trials object exists: Create a new one
    trials_old=Trials()
       
#Adding the new search to the Trials object and  saving it in the shared Trials file
trials_new=trials_from_docs(list(trials_old)+[list(trials_one)[len(list(trials_one))-1]])   
pickle.dump(trials_new, open("./History/Trials.p", "wb"))
