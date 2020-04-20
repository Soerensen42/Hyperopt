# Hyperopt
Using hyperopt to independently train neural networks


# Setup:
The Project contains a Core Runner, Terminal and dump file. Also included is a .sbatch file to start the Runner. Theres also an optional small analyser code

### For the Setup you need to Set a Place for the Trials object (technically the searchhistory) in the Terminal and Core Runner.

# Terminal:

The Terminal will coordinate all of the Runners. It is recmmended to run it, where its not canceled, although 
there are checkpoints implemented if it does. It is not very taxing computationwise. There is a loop sending jobs, i chose 
10 simultanious runners. But this can be easily varied. Just be sure to give every job a unique name, so the loop can detect 
when its finished.

# Core

The Core contains the hyperparameter searchspace and is called for each Iteration. Each Core then makes one Iteration of the network. 
The path of the Trials object has to be inserted at the start and the end of the File.
