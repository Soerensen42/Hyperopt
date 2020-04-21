import pickle
import sys
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def objective(args):
    #handling the arguments
    arg1, arg2 = args
    
    #make sure parameters are the type they should if needed
    arg1=int(arg1)
    
    #Include crashing networks for hyperopt to learn
    #replace Network with your network function 
    try:
        res = Network(arg1,arg2)
        return(res) 
    except:
        return(1)

#loading into the search history, if it exists, change path to match Trials location
trials_one=Trials()
if os.path.exists('/path/Trials.p'):
    trials_one = pickle.load(open("path/Trials.p", "rb"))


#Searchspace, for more options look at the hyperopt documentation     
space = [hp.quniform('arg1', 1,10,1), hp.loguniform('arg2', -9.21034, -1.60943791243)] 

#running one Iteration, before consulting the updated Trials object
best = fmin(objective,space,trials=trials_one,algo = tpe.suggest,max_evals=len(list(trials_one))+1)

if os.path.exists('/path/Trials.p'):
    trials_old = pickle.load(open("/path/Trials.p", "rb"))
else:
    trials_old=Trials()
       
#Updating the trials object
trials_new=trials_from_docs(list(trials_old)+[list(trials_one)[len(list(trials_one))-1]])   
pickle.dump(trials_new, open("/path/Trials.p", "wb"))