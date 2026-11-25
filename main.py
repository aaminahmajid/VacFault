from src.training.train_svm_models import train_svm_models
from src.training.train_cnn_model import train_cnn_model
from src.evaluation.evaluate_svm import evaluate_svm
from src.evaluation.evaluate_cnn import evaluate_cnn
from src.visualize_results.visualize_results import visualize_results
from src.visualize_results.export_results_to_csv import export_results_to_csv

from src.training.train_optimize_svm_models import train_optimize_svm_models

def main():
    # Train and evaluate svm models for ds1,ds2 and ds3, save trained models for deployment 
    
    train_svm_models()

    # Train and evaluate cnn model for ds1,ds2 and ds3, save trained model for deployment 
    train_cnn_model()


    # Evaluation
    #------------
    print("Evaluating performance")
    
    #evaluate svm models for deployment data
    evaluate_svm()
    
    #evaluate cnn model for deployment data 
    evaluate_cnn()
    

    # Visualize results
    #-------------------------
    print("Generating performance comparison plots")
    visualize_results()

    #Export results to table
    print("Exporting results to csv file")
    export_results_to_csv()

    print("done")



if __name__=="__main__":
    main()