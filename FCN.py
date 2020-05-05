import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pandas

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def Load():
  store_train = pandas.HDFStore('path/train.h5')
  df_train = store_train.select("table", stop = 2000)  
  store_train.close()
  df_train_labels = df_train["is_signal_new"].to_numpy()

  store_val = pandas.HDFStore('path/val.h5')
  df_val = store_val.select("table", stop = 2000)
  store_val.close()
  df_val_labels = df_val["is_signal_new"].to_numpy()

  store_test = pandas.HDFStore('path/test.h5')
  df_test = store_test.select("table", stop = 2000)   
  store_test.close()

  df_signal = df_train[df_train["is_signal_new"]==1]
  df_background = df_train[df_train["is_signal_new"]==0]

  df_signal_sum = df_signal.sum(axis = 0, skipna = True)/len(df_signal.index)
  df_background_sum = df_background.sum(axis = 0, skipna = True)/len(df_background.index)

  cols = [c.format(i) for i in range(20) for c in ["E_{0}",  "PX_{0}",  "PY_{0}",  "PZ_{0}"]]

  vectors_val = df_val[cols].to_numpy()
  vectors_val_labels = df_val["is_signal_new"].to_numpy()

  vectors_train = df_train[cols].to_numpy()
  vectors_train_labels = df_train["is_signal_new"].to_numpy()

  print('loading done')

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
                
        self.fc1 = nn.Linear(80,l1)
        self.fc2 = nn.Linear(l1,l2)        
        self.fc3 = nn.Linear(l2,l3)           
        self.fc4 = nn.Linear(k3,k4)         
        self.fc5 = nn.Linear(k4,k5)
        self.fc6 = nn.Linear(k5,2)
        
        
    def forward(self, x): 
        #Running through all used Layers and fitting them in with a relu function
        for k in range(1,5):
            if r{}.format(k) == 1:
                x = F.relu(self.fc{}.format(k)(x))
        #old version of this
        '''        
        if r1 == 1:
            x = F.relu(self.fc1(x))
        if r2 == 1:    
            x = F.relu(self.fc2(x))
        if r3 == 1:
            x = F.relu(self.fc3(x))
        if r4 == 1:    
            x = F.relu(self.fc4(x))
        if r5 == 1:   
            x = F.relu(self.fc5(x))
        '''
        #Using a softmax on the last layer to fit it between 0...1
        x = F.softmax(self.fc6(x))
        
        return x

def Iteration(nLayers, batch_size, learning_rate, l1,l2,l3,l4,l5):
    #setting Variables
    n_epochs = 10    
    test_accs = []
    #Confirm variables are they right type 
    nLayers=int(nLayers)
    batch_size=int(batch_size)
    l1=int(l1)
    l2=int(l2)
    l3=int(l3)
    l4=int(l4)
    l5=int(l5)
    r1=r2=r3=r4=r5=1
    
    #only using as many layers as nLayers suggests and setting l5 to connect to the last used layer 
    for i in range (1,4):
        if nLayers == {}.format(i):
          for a in range (5-i,5):
              r{}.format(i)=0
          l5 = l{}.format(i)    
          
    #old version of this      
    '''
    if nLayers == 1:
        r2=r3=r4=r5=0
        l5=l1
    if nLayers == 2:
        r3=r4=r5=0
        l5=l2
    if nLayers == 3:
        r4=r5=0
        l5=l3
    if nLayers == 4:
        r5=0 
        l5=l4 
    '''    
    
    #Handling the model  over to the GPU
    model = FCN().to(device)
    
    #Setting up x and y values
    x_train = vectors_train
    y_train = vectors_train_labels
    x_val = vectors_val    
    y_val = vectors_val_labels
    
    #defining what optimizer to use (standart is Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_examples = x_train.shape[0]
    n_batches = int(train_examples/batch_size)

    for ep in range(n_epochs):

        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)
        
        x_train = x_train[idx]
        y_train = y_train[idx]
        
        for i in range(n_batches):
        
            optimizer.zero_grad()
            
            i_start = int(i*batch_size)
            i_stop  = int((i+1)*batch_size)
           
            x = torch.tensor(x_train[i_start:i_stop],dtype=torch.float).to(device)
            y = torch.tensor(y_train[i_start:i_stop],dtype=torch.long).to(device)

            net_out = model(x)
        
            loss = criterion(net_out,y)
  
            loss.backward()
        
            optimizer.step()
        
        
    x_test = df_test[cols].to_numpy()
    y_test = df_test["is_signal_new"].to_numpy()
    
    y_pred_test_model = model(torch.tensor(x_test,dtype=torch.float).to(device)).detach().cpu()
    y_pred_test = nn.Softmax(dim=1)(y_pred_test_model).numpy()
    test_acc = sum(y_test == np.argmax(y_pred_test,axis=1)) / y_test.shape[0]
    test_accs.append(test_acc)
    
    return(1-max(test_accs)) 
