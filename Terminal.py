#import
import time
import pickle
import os
import subprocess
from hyperopt import Trials
import numpy as np
import matplotlib.pyplot as plt

print("start")
filler_2 =0
#params

#start off, where you left
if os.path.exists('/path/Trials.p'):
    trials_one = pickle.load(open("/path/Trials.p", "rb"))
    hp_opt = len(list(trials_one))  
else:
    trials_one = Trials()
    pickle.dump(trials_one, open("/path/Trials.p", "wb"))  
    
filler = 0

#continiously check what jobs are running
while len(list(trials_one)) <100: #suspec to change to ~100 again
    
    #check if job is running, to create multiple runners just copy and paste the check, change USERNAME and NUMBER
    if len(subprocess.check_output("squeue -u USERNAME --name=HP-Runner_Number", shell=True))== 85:
        os.system('sbatch HP_Optimizer_NUMBER.sh')
    
    #get the trials object to track progress, try to avoid checking while saving errors
    try:
        trials_one = pickle.load(open("/path/Trials.p", "rb"))
    except:
        time.sleep(30)
        trials_one = pickle.load(open("/path/Trials.p", "rb")) #evade double open error       
    time.sleep(30)    

print("waiting for everything to finish")
while filler_2 == 0: #copy and paste the if check (if ... and ... and ...) for as many cores as you're running
    if len(subprocess.check_output("squeue -u USERNAME --name=HP-Runner_NUMBER", shell=True))== 85 :
        filler_2=1
    else:
        time.sleep(30)
print("done")