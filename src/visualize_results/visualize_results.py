import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#from matplotlib.ticker import MultipleLocator
#from matplotlib.lines import Line2D
#from matplotlib.patches import Patch
from pathlib import Path
import sys
from matplotlib.patches import Patch


# important parameters from configuration file
from src.config import ROOT_DIR
from src.config import results_dir
from src.config import reports_dir
from src.config import datasets
from src.config import performance_metrics


figures_dir = reports_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

def visualize_results(save=True,display_=False):

    models_labels = {'svm_linear': "SVM (linear)", 'svm_rbf': "SVM (RBF)", 
               'svm_ploy': "SVM (polynomial)", 'cnn': "CNN"}
    imbalance_learn_labels = {'cost_sensitive' : 'Cost-sensitive learning',
                                 'smote_plus_cost_sensitive':'SMOTE + Cost-sensitive learning',
                         }

    # x-poistion of bar plot in figure
    model_position = {'svm_linear': 0, 'svm_rbf': 1, 
                   'svm_ploy': 2, 'cnn': 3}  

    colors = {'cost_sensitive' : 'tab:blue',  'smote_plus_cost_sensitive':'tab:orange',  }
    hatch_styles = {'cost_sensitive' : None, 'smote_plus_cost_sensitive':'///',  }
    markers =  {'cost_sensitive' : 'o', 'smote_plus_cost_sensitive':'d',  }
    bar_width = 0.3
    ylabels = {'precision_class1': 'Precision (class: Fault)', 
               'recall_class1': 'Recall (class: Fault)',
                'f1_score_class1': 'F1-score (class: Fault)'}


    # figure parameters
    fontsizeLabel=26
    fontsizeTicks=26
    fontsizeLegend=26
    fig_width =  20#15
    fig_height = 7
    """visualize results for ds1,d2,ds3 (test sets) and deployment data """
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
            ylabel = ylabels[metric]
        
            fig, ax = plt.subplots(figsize=(fig_width,fig_height))
            #print(f"{dataset},{metric}")
            for model in models_labels.keys():
                for imbalance_method in imbalance_learn_labels.keys():
                    for fname in df_results_dict.keys():
                        if f"{model}_{imbalance_method}_{dataset}" in fname:
                            #print(fname)
                            df_performance_metrics = df_results_dict[fname]
            
                            # make plot
                            data = df_performance_metrics[metric]
                            
                            if imbalance_method == 'smote_plus_cost_sensitive':
                                position = model_position[model] +  bar_width/2
                            if imbalance_method == 'cost_sensitive':
                                position = model_position[model] -  bar_width/2
                            
                            bars = ax.bar(position, data, bar_width, 
                                    color = colors[imbalance_method],
                                    hatch = hatch_styles[imbalance_method],
                                    alpha = 1
                                   )
                            for bar in bars:
                                bar_height = bar.get_height()
                                ax.annotate(f'{bar_height:.2f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, bar_height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        fontsize = fontsizeLabel,
                                        ha='center', va='bottom')
                                            
                           
            # Legend (proxy)
            legend_elements = list()
            for imbalance_method in imbalance_learn_labels.keys():
                legend_elements.append(Patch(facecolor=colors[imbalance_method],
                                            label = imbalance_learn_labels[imbalance_method],
                                            hatch = hatch_styles[imbalance_method]
                                            )
                                     )
                
            
            ax.legend(handles=legend_elements,fontsize = fontsizeLegend,
                     loc="lower center",
                     bbox_to_anchor=(0.5, 1.005),
                     ncol=2,
                     frameon = False) #loc='upper right')
    
            
            
            
            ax.set_ylim(0,1.1)
            ax.set_ylabel(ylabel, fontsize = fontsizeLabel)
            ax.set_xlabel('ML Model',  fontsize = fontsizeLabel)
            ax.set_xticks(list(model_position.values()), fontsize = fontsizeTicks)
            ax.set_xticklabels(list(models_labels.values()), fontsize = fontsizeTicks)
            ax.tick_params(axis='y', labelsize=fontsizeTicks,pad=20)
            
            
            plt.tight_layout()
            if save == True:
                figname = f"{figures_dir}/{dataset}_{metric}.pdf"
                plt.savefig(figname, bbox_inches='tight'  ,pad_inches=0.2) 
            if display_==True:
                print(f"{dataset}_{metric}:")
                plt.show()
        
