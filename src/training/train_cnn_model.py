import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
from torchsummary import summary
import joblib
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import  precision_score, recall_score, classification_report,confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os
from pathlib import Path
import sys
from tqdm import tqdm

# important parameters from configuration file
from src.config import ROOT_DIR
from src.config import datasets_dir
from src.config import results_dir
from src.config import models_dir

from src.config import datasets
from src.config import feature_col
from src.config import ml_models_cnn
from src.config import imbalance_learn_methods_cnn
from src.config import kernel_method_svm
from src.config import random_state

ml_model = ml_models_cnn[0]
imbalance_learn_method = imbalance_learn_methods_cnn[0]

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, random_state, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle, random_state=random_state)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = random_state
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model_ = nn.Sequential(
                nn.Conv1d(1, 10, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=10,stride=2),
                nn.Flatten(),
                nn.Linear(320, 100),
                nn.ReLU(),
                nn.Dropout(0.2),  # Dropout with 20% probability
                nn.Linear(100, 2),
        )

    def forward(self, x):
        return self.model_(x)


def train_cnn_model():
    """
    Train svm models for ds1, ds2 and ds3, 
    store performane metrcis,
    save trained models for future deployment
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Using device: {device}')
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    lr=0.001
    num_epochs = 100
    batch_size = 100

    # Load data
    pbar = tqdm(datasets)
    for dataset in pbar: 
    #for dataset in tqdm(datasets, desc="Training"):
        if dataset not in ['ds1','ds2','ds3']:
            continue
        pbar.set_description(f"CNN model training for {dataset}")
        data_dir = os.path.join(datasets_dir,dataset)
        #print(data_dir)
        #print(dataset)
        df_train = pd.read_hdf(f"{data_dir}/df_train.h5")
        df_test = pd.read_hdf(f"{data_dir}/df_test.h5")
    
        # convert time-series data into tensors
        # (training data)
        dim1 = len(df_train.index)
        dim2 = len(df_train['Data'][df_train.index[0]])
        X_train = np.empty((dim1,dim2))
        for i,index in enumerate(df_train.index):
            seq = df_train['Data'][index]
            X_train[i,:] = seq
        y_train = np.array(df_train["labels"])
        y_train = y_train.astype(np.float)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
    
        # (test data)
        dim1 = len(df_test.index)
        dim2 = len(df_test['Data'][df_test.index[0]])
        X_test = np.empty((dim1,dim2))
        for i,index in enumerate(df_test.index):
            seq = df_test['Data'][index]
            X_test[i,:] = seq
        y_test = np.array(df_test["labels"])
        y_test = y_test.astype(np.float)
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
    
        # Data normalizatiom
        #---------------------------------------------------------------
        mean_train = torch.mean(X_train)
        std_train = torch.std(X_train)
        #print(f'mean_train: {mean_train}, std_train: {std_train}')
        X_train_normalized  = (X_train-mean_train)/std_train
        X_test_normalized  = (X_test-mean_train)/std_train
    
        #Reshape tensor dimensions for 1D CNN
        X_train_normalized = X_train_normalized.unsqueeze(1)
        X_test_normalized = X_test_normalized.unsqueeze(1)
        #print(X_train_normalized.shape, X_test_normalized.shape)
    
        # Create dataloader
        train_dataset = TensorDataset(X_train_normalized, y_train)
        test_dataset = TensorDataset(X_test_normalized, y_test)
    
        train_loader = DataLoader(train_dataset,batch_sampler=StratifiedBatchSampler(y_train, batch_size=batch_size,random_state=random_state))
        test_loader = DataLoader(test_dataset,batch_sampler=StratifiedBatchSampler(y_test, batch_size=batch_size,random_state=random_state))
    
        # compute class weights: 
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float)
    
        # Normalize class weights
        normalized_class_weights = class_weights / class_weights.sum()
        #print(normalized_class_weights)
    
        model = CNN()
        model.to(device)
    
        #Define loss function and optimize
        criterion = nn.CrossEntropyLoss(weight= normalized_class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr) 
    
        # exponential LR scheduler
        #scheduler = optim.lr_scheduler.ExponentialLR(
        #optimizer,
        #gamma=0.95
    #)
      
        # model training
        for epoch in range(1,num_epochs+1):
         model.train()
         epoch_loss = 0.0
        
         for inputs, labels in train_loader:
             optimizer.zero_grad()
    
             inputs, labels = inputs.to(device), labels.to(device) 
             # Forward pass
             outputs = model(inputs)   # compute binary outputs
             batch_loss = criterion(outputs, labels) #compute loss
             
             # compute accuracy for current batch
             _, predicted = torch.max(outputs.data, 1)          
                
             # Backward pass and optimization
             batch_loss.backward()
             optimizer.step()
            
             epoch_loss += batch_loss.item()
         
         #scheduler.step()
         epoch_loss_avg_per_batch = epoch_loss / (len(train_loader))
         #print(f"Epoch {epoch}/{num_epochs}, epoch loss for training data (average per batch): {epoch_loss_avg_per_batch:.4f}")
    
    
        # Evaluate performance (training set)
        model = model.to('cpu')
        model.eval()
        with torch.no_grad():
            inputs = X_train_normalized
            labels = y_train   
            # Forward pass
            outputs = model(inputs) # raw logits
            probs = F.softmax(outputs, dim=1)   # probabilities for each class
            _, y_train_pred = torch.max(outputs.data, 1)
    
        train_precision = precision_score(y_train, y_train_pred, average=None)
        train_recall = recall_score(y_train, y_train_pred, average=None)
        train_f1 = f1_score(y_train, y_train_pred, average=None)
        # save performance metrics
        df_perfromance_metrics = pd.DataFrame(columns = ['precision_class0', 'recall_class0', 'f1_score_class0',
                                                        'precision_class1', 'recall_class1', 'f1_score_class1'],
                                                         index= [1])
        
        df_perfromance_metrics['precision_class0'] = train_precision[0]
        df_perfromance_metrics['recall_class0'] = train_recall[0]
        df_perfromance_metrics['f1_score_class0'] = train_f1[0]
        df_perfromance_metrics['precision_class1'] = train_precision[1]
        df_perfromance_metrics['recall_class1'] = train_recall[1]
        df_perfromance_metrics['f1_score_class1'] = train_f1[1]

        dir_ = results_dir / "training" / "performance_metrics"
        dir_.mkdir(parents=True, exist_ok=True)
        fname = f"{dir_}/{ml_model}_{imbalance_learn_method}_{dataset}.h5"
        df_perfromance_metrics.to_hdf(fname, key="df",mode='w')
    
        # Evaluate performance (test set)
        model = model.to('cpu')
        model.eval()
        with torch.no_grad():
            inputs = X_test_normalized
            labels = y_test   
            # Forward pass
            outputs = model(inputs) # raw logits
            probs = F.softmax(outputs, dim=1)   # probabilities for each class
            _, y_test_pred = torch.max(outputs.data, 1)
    
        test_precision = precision_score(y_test, y_test_pred, average=None)
        test_recall = recall_score(y_test, y_test_pred, average=None)
        test_f1 = f1_score(y_test, y_test_pred, average=None)
        # save performance metrics
        df_perfromance_metrics = pd.DataFrame(columns = ['precision_class0', 'recall_class0', 'f1_score_class0',
                                                        'precision_class1', 'recall_class1', 'f1_score_class1'],
                                                         index= [1])
        
        df_perfromance_metrics['precision_class0'] = test_precision[0]
        df_perfromance_metrics['recall_class0'] = test_recall[0]
        df_perfromance_metrics['f1_score_class0'] = test_f1[0]
        df_perfromance_metrics['precision_class1'] = test_precision[1]
        df_perfromance_metrics['recall_class1'] = test_recall[1]
        df_perfromance_metrics['f1_score_class1'] = test_f1[1]

        dir_ = results_dir / "test" / "performance_metrics"
        dir_.mkdir(parents=True, exist_ok=True)
        fname = f"{dir_}/{ml_model}_{imbalance_learn_method}_{dataset}.h5"
        df_perfromance_metrics.to_hdf(fname, key="df",mode='w')
    
        torch.cuda.empty_cache()
        del model

        if dataset == 'ds1':
            # combine all data and train model that can be used for future deployment
            df = pd.concat([df_train,df_test])
            # convert time-series data into tensors
            # (training data)
            dim1 = len(df.index)
            dim2 = len(df['Data'][df.index[0]])
            X = np.empty((dim1,dim2))
            for i,index in enumerate(df.index):
                seq = df['Data'][index]
                X[i,:] = seq
            y = np.array(df["labels"])
            y = y.astype(np.float)
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).long()
        
            # Data normalizatiom
            #---------------------------------------------------------------
            mean = torch.mean(X)
            std = torch.std(X)
            #print(f'mean: {mean}, std: {std}')
            X_normalized  = (X-mean)/std
        
            #Reshape tensor dimensions for 1D CNN
            X_normalized = X_normalized.unsqueeze(1)
            #print(X_normalized.shape)
        
            # Create dataloader
            train_dataset = TensorDataset(X_normalized, y)
            train_loader = DataLoader(train_dataset,batch_sampler=StratifiedBatchSampler(y, batch_size=batch_size,random_state=random_state))
        
            # compute class weights: 
            classes = np.unique(y)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y.numpy())
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        
            # Normalize class weights
            normalized_class_weights = class_weights / class_weights.sum()
            #print(normalized_class_weights)
        
            model = CNN()
            model.to(device)
        
            #Define loss function and optimize
            criterion = nn.CrossEntropyLoss(weight= normalized_class_weights)
            optimizer = optim.Adam(model.parameters(), lr=lr) 
        
            # exponential LR scheduler
            #scheduler = optim.lr_scheduler.ExponentialLR(
            #optimizer,
            #gamma=0.95
        #)
          
            # model training
            for epoch in range(1,num_epochs+1):
                 model.train()
                 epoch_loss = 0.0
                
                 for inputs, labels in train_loader:
                     optimizer.zero_grad()
            
                     inputs, labels = inputs.to(device), labels.to(device) 
                     # Forward pass
                     outputs = model(inputs)   # compute binary outputs
                     batch_loss = criterion(outputs, labels) #compute loss
                     
                     # compute accuracy for current batch
                     _, predicted = torch.max(outputs.data, 1)          
                        
                     # Backward pass and optimization
                     batch_loss.backward()
                     optimizer.step()
                    
                     epoch_loss += batch_loss.item()
                 
                 #scheduler.step()
                 epoch_loss_avg_per_batch = epoch_loss / (len(train_loader))
                 #print(f"Epoch {epoch}/{num_epochs}, epoch loss for training data (average per batch): {epoch_loss_avg_per_batch:.4f}")
        
            #save model parameters
            mean.to('cpu')
            std.to('cpu')
            model.eval()
            model.to('cpu')
             
            current_model_dir = f"{models_dir}/{ml_model}_{imbalance_learn_method}/"
            os.makedirs(current_model_dir, exist_ok =True)
             
            #save normalization parameters
            scaler_name =  f"{current_model_dir}scaler.pkl"
            joblib.dump((mean, std), scaler_name)
            ml_model_name = f"{current_model_dir}ml_model.pt"
            model_scripted = torch.jit.script(model)
            model_scripted.save(ml_model_name)
                
            torch.cuda.empty_cache()
            del model

                    






