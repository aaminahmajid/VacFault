import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

# important parameters from configuration file
from src.config import ROOT_DIR
from src.config import results_dir
from src.config import reports_dir
from src.config import datasets
from src.config import performance_metrics
from src.config import ml_models
from src.config import imbalance_learn_methods



tables_dir = reports_dir / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)

def export_results_to_csv(save=True, display_=False):
    
    # create datafram structure to save results
    # row multiindex
    rows = pd.MultiIndex.from_product(
        [datasets, ml_models],
        names=["Dataset", "ML model"]
    )
    
    # column multiindex
    cols = pd.MultiIndex.from_product(
        [imbalance_learn_methods, performance_metrics]
    )
    
    table = pd.DataFrame(index=rows, columns=cols)
    
    for dataset in datasets:
        if dataset in ['ds1','ds2','ds3']:
            performance_metrics_dir = f"{results_dir}/test/performance_metrics/"
        if dataset in ['deployment_data']:
            performance_metrics_dir = f"{results_dir}/deployment/performance_metrics/"
    
        # fetch results for each dataset
        df_results_dict = dict()
        for file_name in os.listdir(performance_metrics_dir):
            #print(file)
            if dataset in file_name:
                #print(file_name)
                df_results_dict[file_name] = pd.read_hdf(f"{performance_metrics_dir}{file_name}")
        # For given dataset, plot all performance metrics
        for metric in performance_metrics:  
            #print(f"{dataset}:")
            for ml_model in ml_models:
                for imbalance_method in imbalance_learn_methods:
                    for fname in df_results_dict.keys():
                        if f"{ml_model}_{imbalance_method}_{dataset}" in fname:
                            #print(fname)
                            df_performance_metrics = df_results_dict[fname]
            
                            # save data
                            data = df_performance_metrics[metric]
                            table.loc[(dataset, ml_model), (imbalance_method, metric)] = data.values[0]
    
    
    # save table containing results
    if save==True:
        table = table.round(2).astype(str)
        table.to_csv(f"{tables_dir}/results_table.csv",  float_format="%.2f")
    if display_ == True:
        pd.options.display.float_format = '{:.2f}'.format
        display(table)
    