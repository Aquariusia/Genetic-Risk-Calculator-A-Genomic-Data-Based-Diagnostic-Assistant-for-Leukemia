# PROJECT NAME: Genetic-Risk-Calculator-A-Genomic-Data-Based-Diagnostic-Assistant-for-Leukemia
A Streamlit-based web application that serves as an AI-powered diagnostic assistant for leukemia. The system integrates a base classification model with a fine-tunes large language model (LLM) to assist analysis. Complex prompt engineering is applied to guide the AI outputs.

PROJECT OVERVIEW
This project uses an XGBoost model to perform binary classification of gene expression data from ALL (Acute Lymphoblastic Leukemia) and AML (Acute Myeloid Leukemia) samples.


https://github.com/user-attachments/assets/72c62552-483f-4bd8-9263-c8a502b65b1c




KEY COMPONENTS

* Root folder: example input/output files, trained model and predictions
* modeling: data preprocessing, feature engineering, model training and evaluation in a Jupyter Notebook
* app.py : standalone Streamlit script for interactive web interface

DIRECTORY STRUCTURE

|-- app                                       code for web application

|   |-- api\_key.env                              environment variables (ChatGPT API keys)

|   |-- app.py                                    Streamlit web app main script

|   |-- example\_input.csv                        sample input file

|   |-- example1.csv                              sample input variation 1

|   |-- example2.csv                              sample input variation 2

|   └-- gene\_means.csv                           per-gene mean expression values computed during training

|   └-- model.pkl                                 XGBoost model

|   └-- probs.npz                                 predicted probabilities for the train and test set 

|   └-- requirement.txt                           required packages for web application

|

|-- modeling/                                  training workflow and results

|   |-- actual.csv                            true labels for the independent test set

|   |-- data\_set\_ALL\_AML\_train.csv            training dataset

|   |-- data\_set\_ALL\_AML\_independent.csv      independent test dataset

|   |-- genedata.ipynb                          Jupyter Notebook: preprocessing, training, evaluation

|   |-- model.pkl                               model saved within the Notebook (can be merged with root)

|   |-- xgboost\_confusion\_matrix.png            confusion matrix visualization

|   └-- xgboost\_feature\_importance.png          feature importance plot

USAGE NOTES

* Keep model.pkl synchronized with your latest training results.
* Exclude large data and model files from version control by adding them to .gitignore.
* To update the model, retrain in modeling/ and copy the new model.pkl to the root directory.

LICENSE
This material, developed for the University of Sydney and Accenture co-delivered course, remains the intellectual property of the author. It is provided for educational and internal use only. Redistribution, modification, or commercial exploitation without prior written permission from the author is prohibited.
