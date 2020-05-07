import pickle
import sys
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from FCN.py import *

#getting parameters
from params import *
Parameters={}
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

if Parameters[Iterations] < len(list(trials_one)):
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
print("progress:",len(list(Trials)),"/",Parameters["Iteration"]) #show progress

#Saving the best params for in optimization use and result viewing
with open('./Results.txt', 'w') as f:
    json.dump(best, f)
