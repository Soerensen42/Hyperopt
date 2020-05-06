# Hyperopt
Using hyperopt to independently train neural networks

The Example Data set i used and some explaination about it can be found here:
https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6
https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit

The Process uses the Slurm job submitting system, for other solutions look into the Terminal section


# Setup:
The Project contains a Core Runner and Terminal. Also included is a .sbatch file to start the Runner. Theres also an optional small analyser code (wip). The Optimizer implements a Fully connected network, so its usable out of the box.

First Setup:
Download all the files and set up a "History" folder. Change the Data path in the Params file. It is currently set to look into a folder named "data" with the 3 Data sets. Setup the .batch file to handle the Jobs. I do not reccoemnd setting the job to send an email when finished since that would lead to a bit more than 100 emails.

Changing the Network:
-Modify the Network Training to be callable as a function returning the value that should be optimised for. (remember optimization always tries to minimize so for auc score would be something like return(1-accuracy))
-Remove both Code lines Specific to the example.
-Make sure the network selected hyperparamters are the right type, most fitting would be at the start of the function.

# Core

The Core is called for each Iteration which then does one training of the network. 
While Searching, the runner can and should be called multiple times and ran simultaniously.

#Paramters

The Parameters file is used to set all the Parameters in one place. Of special interest should be the searchspace definition:
While it is possible to set a large amount of search parameters keep in mind, that having, for example 5 different variables that all can be one of 256 values might take a few more runs to optimize correctly. 

That being said the optimizer should recognize less important Values and not focus on them too much.

Setting the search space is possible with:

hp.uniform('name',a,b) draws a value soemwhere between a and b
hp.quniform('name',a,b,c) draws from [a,1+c,a+2c,...,b-c,b] This has to fit, so a+Xc=b must be true or the function crashes
hp.loguniform('name',np.log(a),np.log(b)) draws logarithmic between values a and b, but is very heavy on a

These are just the ones i found most helpful, alternatives can be found at:
https://github.com/hyperopt/hyperopt/wiki/FMin

There is also a Quick scan option for setup and debugging. This uses an extremely small data sample for the FCN.

# Fully Connected Network (FCN)

This is the default Network. It is recommended To setup the Optimizer with the FCN once to see if everything works and get a feeling for the progress. The FCN consists of a few Layers each consisiting of Nodes. Each node is connected to every other node of the previous and following layer. 

# Terminal:

Old solution for The Core Handling, this might be helpful if Slurm is not used for handling job submissions. The length of the Trials object can be used to keep track of how far the search is. 

# Networkrecommendations for faster search

-early stopping
-scanning over a reduced training sample
