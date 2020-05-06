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
    #else: create  new Trials object/location
    trials_one = Trials()
    pickle.dump(trials_one, open("/path/Trials.p", "wb"))  

#continiously check what jobs are running
while len(list(trials_one)) <100: #Amount of searches
    
    #check if job is running
    #Set the number of simultanious runners
    if len(subprocess.check_output("squeue -u USERNAME --name=HP-Runner_Number", shell=True))== 85: #empty list is 85 characters long
        #The sbatch just calls a Core.py
        os.system('sbatch HP_Optimizer_NUMBER.sh')
    
    #get the trials object to track progress, try to avoid checking while saving errors
    try:
        trials_one = pickle.load(open("/path/Trials.p", "rb"))
    except:
        time.sleep(30)    
    time.sleep(30)                                                            

print("waiting for everything to finish")
while filler_2 == 0: #copy and paste the if check (if ... and ... and ...) for as many cores as you're running
    if len(subprocess.check_output("squeue -u USERNAME --name=HP-Runner_NUMBER", shell=True))== 85 :
        filler_2=1
    else:
        time.sleep(30)
print("done")
