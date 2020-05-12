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

def Load(Parameters):
  #loading in the 3 needed data sets
  store_train= pandas.HDFStore('{}/train.h5'.format(Parameters["data_location"]))
  #if quick is set to 1: only read first 2000 entries (great for debugging/testing)
  if Parameters["Quick"]==1: 
      df_train = store_train.select("table", stop = 2000)  
  else:
      df_train = store_train.select("table")
  #closing the file    
  store_train.close()
  #saving labels for accuracy calculation
  df_train_labels = df_train["is_signal_new"].to_numpy()
  
  store_val = pandas.HDFStore('{}/val.h5'.format(Parameters["data_location"]))
  if Parameters["Quick"]==1:
      df_val = store_val.select("table", stop = 2000)
  else:
      df_val = store_val.select("table")
  store_val.close()
  df_val_labels = df_val["is_signal_new"].to_numpy()

  store_test = pandas.HDFStore('{}/test.h5'.format(Parameters["data_location"]))
  if Parameters["Quick"]==1:
      df_test = store_test.select("table", stop = 2000) 
  else:
      df_test = store_test.select("table")
  store_test.close()
  
  #seperating signal and background
  df_signal = df_train[df_train["is_signal_new"]==1]
  df_background = df_train[df_train["is_signal_new"]==0]
  
  #determining amound of signal and background
  df_signal_sum = df_signal.sum(axis = 0, skipna = True)/len(df_signal.index)
  df_background_sum = df_background.sum(axis = 0, skipna = True)/len(df_background.index)

  #only using the first 20 values since everything past that would be too little to influence the network 
  # => removing unneccessary calculation of empty data
  cols = [c.format(i) for i in range(20) for c in ["E_{0}",  "PX_{0}",  "PY_{0}",  "PZ_{0}"]]

  vectors_val = df_val[cols].to_numpy()
  vectors_val_labels = df_val["is_signal_new"].to_numpy()

  vectors_train = df_train[cols].to_numpy()
  vectors_train_labels = df_train["is_signal_new"].to_numpy()

  print('loading done')
  return(vectors_val,vectors_val_labels,vectors_train,vectors_train_labels,df_test[cols].to_numpy(),df_test["is_signal_new"].to_numpy())

class FCN(nn.Module):
    def __init__(self,Nodes,nLayers):
        super(FCN, self).__init__()
        
        #setting the FCN Layers, 
        self.fcstart = nn.Linear(80,Nodes)
        self.fc1 = nn.Linear(Nodes,Nodes)
        self.fc2 = nn.Linear(Nodes,Nodes)
        self.fc3 = nn.Linear(Nodes,Nodes)
        self.fc4 = nn.Linear(Nodes,Nodes)
        self.fc5 = nn.Linear(Nodes,Nodes)
        self.fcend = nn.Linear(Nodes,2)
        self.nLayers = nLayers
        
        
    def forward(self, x): 
      	# Activating the First Layer
        x = F.relu(self.fcstart(x))
        # Activating the amount of Layers the variable nLayers dictates with a relu function
        if self.nLayers > 1:
            x = F.relu(self.fc1(x))
        if self.nLayers > 2:
            x = F.relu(self.fc2(x))
        if self.nLayers > 3:
            x = F.relu(self.fc3(x))
        if self.nLayers > 4:   
            x = F.relu(self.fc4(x))
        if self.nLayers > 5:   
            x = F.relu(self.fc5(x))        
        #Using a softmax on the last layer to fit result between 0 and 1
        x = F.softmax(self.fcend(x))
        return x

def Iteration(nLayers, batch_size, learning_rate,Nodes,Parameters):
    #Moving everything to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    #print(device) #debug
    #load all the stuff
    vectors_val,vectors_val_labels,vectors_train,vectors_train_labels,x_test,y_test=Load(Parameters)
    #setting Variables
    n_epochs = Parameters["Epochs_FCN"]   
    #Confirm variables are they right type 
    nLayers=int(nLayers)
    batch_size=int(batch_size)
    Nodes=int(Nodes) 
    
    #Handing the model  over to the GPU
    model = FCN(Nodes,nLayers).to(device)
    
    #Setting up x and y values
    x_train = vectors_train
    y_train = vectors_train_labels
    x_val = vectors_val    
    y_val = vectors_val_labels
    
    #defining what optimizer to use (standard is Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_examples = x_train.shape[0]
    
    #grouping the data in bunches to not overload the GPU 
    n_batches = int(train_examples/batch_size)

    for ep in range(n_epochs):
        
        #start of one epoch
        idx = np.arange(x_train.shape[0])
        #randomizing the array so the Network doesnt just count the numbers
        np.random.shuffle(idx)
        
        x_train = x_train[idx]
        y_train = y_train[idx]
        
        for i in range(n_batches):
        
            optimizer.zero_grad()
            
            #setting the batch size range
            i_start = int(i*batch_size)
            i_stop  = int((i+1)*batch_size)
            
            #setting up the datasets
            x = torch.tensor(x_train[i_start:i_stop],dtype=torch.float).to(device)
            y = torch.tensor(y_train[i_start:i_stop],dtype=torch.long).to(device)
            
            #running everything through the model
            net_out = model(x)
            
            #calcualting the loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(net_out,y)
            
            #creating new model (learning)
            loss.backward()
            optimizer.step()
    
    #comparing the guessed values to the real ones
    y_pred_test_model = model(torch.tensor(x_test,dtype=torch.float).to(device)).detach().cpu()
    y_pred_test = nn.Softmax(dim=1)(y_pred_test_model).numpy()
    test_acc = sum(y_test == np.argmax(y_pred_test,axis=1)) / y_test.shape[0]
    
    
    #returning as 1-accuracy as hyperopt tries to minimize
    return(1-test_acc)
