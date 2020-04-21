# Hyperopt
Using hyperopt to independently train neural networks


# Setup:
The Project contains a Core Runner and Terminal. Also included is a .sbatch file to start the Runner. Theres also an optional small analyser code

For the Setup you need to Set a Place for the Trials object (technically the searchhistory) in the Terminal and Core Runner.
The network needs o be callable as a function and returning the value, the network should optimize for(Hyperopt always 
tries to minimize this value). 

There might be the need to recheck the number in the subprocess check

# Terminal:

The Terminal will coordinate all of the Runners. It is recmmended to run it, where its not canceled, although 
there are checkpoints implemented if it does. It is not very taxing computationwise. There is a loop sending jobs, i chose 
10 simultanious runners. But this can be easily varied. Just be sure to give every job a unique name, so the loop can detect 
when its finished.

Recommended scan number is 100, can of course be modified as needed.

# Core

The Core contains the hyperparameter searchspace and is called for each Iteration. Each Core then makes one Iteration of the network. 
The path of the Trials object has to be inserted at the start and the end of the File. While Searching, the runner can be called multiple times and ran simultaniously, as long as each iteration has a new name.

The objective function is needed for optimizing more than one parameter. It may also be necessary to confirm that certain Variables are the type they should be to avoid unnecessary crashes (especially integers).

# Recommendations for faster search

-early stopping or
-scanning over a reduced training sample
