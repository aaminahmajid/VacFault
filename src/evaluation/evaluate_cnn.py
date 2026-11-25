import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import  precision_score, recall_score, classification_report,confusion_matrix, balanced_accuracy_score
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import os

# important parameters from configuration file
from src.config import ROOT_DIR
from src.config import datasets_dir
from src.config import results_dir
from src.config import models_dir
from src.config import datasets
from src.config import feature_col
from src.config import ml_models_cnn
from src.config import imbalance_learn_methods_cnn

ml_models = ml_models_cnn
imbalance_learn_methods = imbalance_learn_methods_cnn

def evaluate_cnn():
    """load pre trained cnn model and evaluate its perfomance for deployment data"""

    # Load deployment data subset
    dataset = 'deployment_data'
    data_dir = os.path.join(datasets_dir,dataset)
    #print(data_dir)
    df = pd.read_hdf(f"{data_dir}/df_deployment.h5")
    
    
    for ml_model in ml_models:
        for imbalance_learn_method in imbalance_learn_methods:
            current_model_dir = f"{models_dir}/{ml_model}_{imbalance_learn_method}/"
    
            # Load scaler parameters
            scaler_name = f"{current_model_dir}scaler.pkl"
            mean, std = joblib.load(scaler_name) 
            # Load ML model parameters
            model_name = f"{current_model_dir}ml_model.pt"
            classifier_model = torch.jit.load(model_name) 
    
            # data conversion
            dim1 = len(df.index)
            dim2 = len(df['Data'][df.index[0]])
            X = np.empty((dim1,dim2))
            for i_,index_ in enumerate(df.index):
                seq = df['Data'][index_]
                X[i_,:] = seq
            
            X = torch.from_numpy(X).float()
            # Data normalization
            X_normalized = (X-mean)/std
    
            # Evaluate performance
            classifier_model.eval()
            outputs = classifier_model(X_normalized.unsqueeze(1))       
            _, y_pred = torch.max(outputs.data, 1)    
            pred_prob = F.softmax(outputs, dim=1)
    
            y_pred = y_pred.numpy()
            y = df['labels']
            y = y.astype(int)
    
            # cimpute performance metrics based on predictions
            precision = precision_score(y, y_pred, average=None)
            recall = recall_score(y, y_pred, average=None)
            f1 = f1_score(y, y_pred, average=None)
            # save performance metrics
            df_perfromance_metrics = pd.DataFrame(columns = ['precision_class0', 'recall_class0', 'f1_score_class0',
                                                            'precision_class1', 'recall_class1', 'f1_score_class1'],
                                                             index= [1])
            
            df_perfromance_metrics['precision_class0'] = precision[0]
            df_perfromance_metrics['recall_class0'] = recall[0]
            df_perfromance_metrics['f1_score_class0'] = f1[0]
            df_perfromance_metrics['precision_class1'] = precision[1]
            df_perfromance_metrics['recall_class1'] = recall[1]
            df_perfromance_metrics['f1_score_class1'] = f1[1]

            dir_ = results_dir / "deployment" / "performance_metrics"
            dir_.mkdir(parents=True, exist_ok=True)
            fname = f"{dir_}/{ml_model}_{imbalance_learn_method}_{dataset}.h5"
            df_perfromance_metrics.to_hdf(fname, key="df",mode='w')
            



    