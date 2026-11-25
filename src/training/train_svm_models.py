import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold,GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import  precision_score, recall_score, classification_report,confusion_matrix, balanced_accuracy_score
from sklearn.svm import SVC
import joblib
import os
from pathlib import Path
import sys
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from tqdm import tqdm


# important parameters from configuration file
from src.config import ROOT_DIR
from src.config import datasets_dir
from src.config import results_dir
from src.config import models_dir

from src.config import datasets
from src.config import feature_col
from src.config import ml_models_svm
from src.config import imbalance_learn_methods_svm
from src.config import kernel_method_svm
from src.config import random_state

ml_models = ml_models_svm
imbalance_learn_methods = imbalance_learn_methods_svm

def train_svm_models():
    """
    Train svm models for ds1, ds2 and ds3, 
    store performane metrcis
    save trained models for future deployment
    """
    # Load data
    for dataset in datasets:
        if dataset not in ['ds1','ds2','ds3']:
            continue
        data_dir = os.path.join(datasets_dir,dataset)
        #print(data_dir)
        df_train = pd.read_hdf(f"{data_dir}/df_train.h5")
        df_test = pd.read_hdf(f"{data_dir}/df_test.h5")
        
        X_train = df_train[feature_col]
        y_train = df_train['labels']
        X_test = df_test[feature_col]
        y_test = df_test['labels']
        
        for ml_model in tqdm(ml_models, desc=f"SVM models training for {dataset}"):
            for imbalance_learn_method in imbalance_learn_methods:
                kernel = kernel_method_svm[ml_model]
                #print(dataset,ml_model, kernel, imbalance_learn_method)
                # fit the model on training data
                svm = SVC(class_weight='balanced', probability=True,gamma='auto',kernel=kernel,random_state=random_state)
                if imbalance_learn_method == 'cost_sensitive': 
                    pipeline = Pipeline([
                                        ('scaler', StandardScaler()),
                                        ('svm', svm),
                                        ])   
                    pipeline.fit(X_train, y_train)
                elif imbalance_learn_method == 'smote_plus_cost_sensitive':
                    pipeline = Pipeline([
                                        ('scaler', StandardScaler()),
                                        ('smote', SMOTE(sampling_strategy=0.5,random_state=42)),
                                        ('svm', svm),
                                        ])                              
                    pipeline.fit(X_train, y_train)
    
                # Evaluate performance (training set)
                y_train_pred = pipeline.predict(X_train)
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
                y_test_pred = pipeline.predict(X_test)
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
    
    
                # save the scaler and models for deployment    
                if dataset == 'ds1':
                    # combine all data and train model that can be used for future deployment
                    df = pd.concat([df_train,df_test])
                    # Separate features vs.labels
                    X = df[feature_col]
                    y = df['labels']
    
                    svm = SVC(class_weight='balanced', probability=True,kernel=kernel,random_state=random_state)
                    if imbalance_learn_method == 'cost_sensitive': 
                        pipeline = Pipeline([
                                        ('scaler', StandardScaler()),
                                        ('svm', svm),
                                        ])   
                        pipeline.fit(X, y)
                    elif imbalance_learn_method == 'smote_plus_cost_sensitive':
                        pipeline = Pipeline([
                                        ('scaler', StandardScaler()),
                                        ('smote', SMOTE(sampling_strategy=0.5,random_state=42)),
                                        ('svm', svm),
                                        ])                              
                        pipeline.fit(X, y)
    
                    # saving
                    current_model_dir = f"{models_dir}/{ml_model}_{imbalance_learn_method}/"
                    os.makedirs(current_model_dir, exist_ok =True)
                    pipeline_name = f"{current_model_dir}pipeline.pkl"
                    joblib.dump(pipeline, pipeline_name)

                


    
    return


