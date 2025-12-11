# Genetic Risk Calculator: A Genomic Data-Based Diagnostic Assistant for Leukemia

A Streamlit-based web application that serves as an AI-powered diagnostic assistant for leukemia. The system integrates a base classification model with a fine-tuned large language model (LLM) to assist analysis. Complex prompt engineering is applied to guide the AI outputs.

## PROJECT OVERVIEW
This project uses an XGBoost model to perform binary classification of gene expression data from ALL (Acute Lymphoblastic Leukemia) and AML (Acute Myeloid Leukemia) samples.

https://github.com/user-attachments/assets/72c62552-483f-4bd8-9263-c8a502b65b1c

## KEY COMPONENTS

* **Root folder:** example input/output files, trained model and predictions
* **modeling:** data preprocessing, feature engineering, model training and evaluation in a Jupyter Notebook
* **app.py:** standalone Streamlit script for interactive web interface

## DIRECTORY STRUCTURE

```text
|-- app/                                      # code for web application
|   |-- api_key.env                           # environment variables (ChatGPT API keys)
|   |-- app.py                                # Streamlit web app main script
|   |-- example_input.csv                     # sample input file
|   |-- example1.csv                          # sample input variation 1
|   |-- example2.csv                          # sample input variation 2
|   └-- gene_means.csv                        # per-gene mean expression values computed during training
|   └-- model.pkl                             # XGBoost model
|   └-- probs.npz                             # predicted probabilities for the train and test set 
|   └-- requirement.txt                       # required packages for web application
|
|-- modeling/                                 # training workflow and results
|   |-- actual.csv                            # true labels for the independent test set
|   |-- data_set_ALL_AML_train.csv            # training dataset
|   |-- data_set_ALL_AML_independent.csv      # independent test dataset
|   |-- genedata.ipynb                        # Jupyter Notebook: preprocessing, training, evaluation
|   |-- model.pkl                             # model saved within the Notebook (can be merged with root)
|   |-- xgboost_confusion_matrix.png          # confusion matrix visualization
|   └-- xgboost_feature_importance.png        # feature importance plot
