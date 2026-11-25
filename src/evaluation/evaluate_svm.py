import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import  precision_score, recall_score, classification_report,confusion_matrix, balanced_accuracy_score
import joblib
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
from src.config import ml_models_svm
from src.config import imbalance_learn_methods_svm

ml_models = ml_models_svm
imbalance_learn_methods = imbalance_learn_methods_svm



def evaluate_svm(save_results=True):
    """load pre trained svm model and evaluate its perfomance for deployment data"""
    # Load deployment data subset

    dataset = 'deployment_data'
    data_dir = os.path.join(datasets_dir,dataset)
    #print(data_dir)
    df = pd.read_hdf(f"{data_dir}/df_deployment.h5")   
    
    for ml_model in ml_models:
        for imbalance_learn_method in imbalance_learn_methods:
            current_model_dir = f"{models_dir}/{ml_model}_{imbalance_learn_method}/"
            print(ml_model,imbalance_learn_method)
            # Load pipeline parameters
            pipeline_name = f"{current_model_dir}pipeline.pkl"
            pipeline = joblib.load(pipeline_name)
    
            # Separate features vs.labels
            X = df[feature_col]
            y = df['labels']
            y = y.astype(int)
    
            # Evaluate performance
            y_pred = pipeline.predict(X)
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

            if save_results==True:
                dir_ = results_dir / "deployment" / "performance_metrics"
                dir_.mkdir(parents=True, exist_ok=True)
                fname = f"{dir_}/{ml_model}_{imbalance_learn_method}_{dataset}.h5"
                df_perfromance_metrics.to_hdf(fname, key="df",mode='w')
            

    