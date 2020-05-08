# Hyperopt
Using hyperopt to independently train neural networks

The Example Data set i used and some explaination about it can be found here:
https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6
https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit

The Process uses the Slurm job submitting system, for other solutions look into the Terminal section


# Setup:
The Project contains a Core Runner and Terminal.sh. Theres also an optional small analyser code (wip). The Optimizer implements a fully connected network, so its usable out of the box. The Terminal File is Maxwell specific, if youre not using Slurm or Maxwell this will need some tinkering.

First Setup:
Download all the files and set up a "History" folder (it should be empty). Change the Data path on top of the Core.py file. It is currently set to look into a folder named "data" with the 3 Data sets. Setup the .batch file to handle the Jobs. It is possible to set The script to send an email when finished. This will only send one. The used environment can be found in FCN.yml. And installes with
'''
conda env create -f Hyperopt-env.yml
'''


.sh File:
You do have to change the .sh file manually:
-FCN into whatever you named ur environment (line 23)
-USER and NETWORK to ur username and folder with the network (line 26)
-The amount of searches if needed (line 12)

When all is done start with '''sbatch Terminal.sh'''

Changing the Network:
-Modify the Network Training to be callable as a function returning the value that should be optimised for. (remember optimization always tries to minimize so for accuracy would be something like return(1-accuracy))
-Remove both Code lines Specific to the example in Core.py.
-Make sure the network selected hyperparamters are the right type, most fitting would be at the start of the function.

Reset:
To Start another search either delete the Trials.p and Results.txt files or rename them. If The Terminal crashes somehow You need to adapt the maximum amounts of jobs the Terminal issues now. (already finished jobs is gained by looking at the Trials object)
'''
print(len(list(Trials)))
'''

Results:
Currently best Paraeters set can be found in the Results.txt file. There is also the possibility to play with the Trials.p for more analyzation options. There is an example in the Analyzer.py file (wip)

# Core

The Core is called for each Iteration which then does one training of the network. 
While Searching, the runner can and should be called multiple times and ran simultaniously.

# Paramters

The Parameters can be set on top of the Core.py file. Of special interest should be the searchspace definition:
While it is possible to set a large amount of search parameters keep in mind, that having, for example 5 different variables that all can be one of 256 values might take a few more runs to optimize correctly. That being said the optimizer should recognize less important Values and not focus on them too much.

Setting the search space is possible with:

hp.uniform('name',a,b) draws a value soemwhere between a and b
hp.quniform('name',a,b,c) draws from [a,1+c,a+2c,...,b-c,b] This has to fit, so a+Xc=b must be true or the function crashes
hp.loguniform('name',np.log(a),np.log(b)) draws logarithmic between values a and b, but is very heavy on a

These are just the ones i found most helpful, alternatives can be found at:
https://github.com/hyperopt/hyperopt/wiki/FMin

There is also a Quick scan option for setup and debugging. This uses an extremely small data sample for the FCN.

Since the .sh script and the Core.py script cant communicate, there is the option to set a maximum amount of searches in the params file. When the number is reached, every Core.py script will isntantly terminate after starting. If there is no reason to stick exactly to the amount of searches set in the .sh file, just set it to a high number (ex. 1000).

# Fully Connected Network (FCN)

This is the default Network. It is recommended To setup the Optimizer with the FCN once to see if everything works and get a feeling for the progress. The FCN consists of a few Layers each consisiting of Nodes. Each node is connected to every other node of the previous and following layer. 

# Screen
!Not neccessary if you use the provided Slurm solution!
The recommended setup for the Terminal is calling the Terminal.py file from within a Screen. To do this type "screen -S NAME" into your console and replace NAME. start the Terminal with "python Terminal.py". Maybe you have to navigate to the corresponding folder first. Now the Screen can be deatached with ctrl+a and pressing d (ctrl+a and k will terminate it instead). Returning to the screen will be possible with "screen -r". This does have the advantage of running even while after the command window is closed (it has to be detached first though. 

# Terminal:

Old solution for The Core Handling, this might be helpful if Slurm is not used for handling job submissions. The length of the Trials object can be used to keep track of how far the search is. 

# Networkrecommendations for faster search

-early stopping
-scanning over a reduced training sample
