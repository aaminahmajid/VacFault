from pathlib import Path

# Base project directory
ROOT_DIR = Path(__file__).resolve().parents[1]

datasets_dir = ROOT_DIR / "data"
results_dir = ROOT_DIR / "results"
models_dir = ROOT_DIR / "models"
reports_dir = ROOT_DIR / "reports"



# Other parameters
random_state = 42
datasets =  ['ds1','ds2','ds3','deployment_data']
feature_col=['std', 'skewness', 'kurtosis', 'hyperskewness'] 
ml_models_svm = ['svm_linear','svm_rbf','svm_poly']
kernel_method_svm = {
    'svm_linear': 'linear',
    'svm_rbf': 'rbf',
    'svm_poly': 'poly'    
                    }
imbalance_learn_methods_svm = ['cost_sensitive', 'smote_plus_cost_sensitive']
ml_models_cnn = ['cnn']
imbalance_learn_methods_cnn = ['cost_sensitive']
# combine all ml_models infor 
ml_models = list(set(ml_models_svm) | set(ml_models_cnn))
imbalance_learn_methods= list(set(imbalance_learn_methods_svm) | set(imbalance_learn_methods_cnn))
performance_metrics = ['precision_class1','recall_class1','f1_score_class1']