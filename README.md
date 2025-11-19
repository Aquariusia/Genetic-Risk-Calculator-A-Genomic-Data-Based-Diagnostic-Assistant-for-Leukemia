# PROJECT NAME: Genetic-Risk-Calculator-A-Genomic-Data-Based-Diagnostic-Assistant-for-Leukemia
A Streamlit-based web application that serves as an AI-powered diagnostic assistant for leukemia. The system integrates a base classification model with a fine-tunes large language model (LLM) to assist analysis. Complex prompt engineering is applied to guide the AI outputs.

PROJECT OVERVIEW
This project uses an XGBoost model to perform binary classification of gene expression data from ALL (Acute Lymphoblastic Leukemia) and AML (Acute Myeloid Leukemia) samples.

KEY COMPONENTS

* Root folder: example input/output files, trained model and predictions
* modeling: data preprocessing, feature engineering, model training and evaluation in a Jupyter Notebook
* app.py : standalone Streamlit script for interactive web interface

DIRECTORY STRUCTURE

\|-- app/                                       code for web application
\|   |-- api\_key.env                              environment variables (ChatGPT API keys)
\|   |-- app.py                                    Streamlit web app main script
\|   |-- app (for streamlit version <= 1.2.0).py   Streamlit web app main script
\|   |-- example\_input.csv                        sample input file
\|   |-- example1.csv                              sample input variation 1
\|   |-- example2.csv                              sample input variation 2
\|   └-- gene\_means.csv                           per-gene mean expression values computed during training
\|   └-- model.pkl                                 XGBoost model
\|   └-- probs.npz                                 predicted probabilities for the train and test set 
\|   └-- requirement.txt                           required packages for web application
|
\|-- modeling/                                  training workflow and results
\|   |-- actual.csv                            true labels for the independent test set
\|   |-- data\_set\_ALL\_AML\_train.csv            training dataset
\|   |-- data\_set\_ALL\_AML\_independent.csv      independent test dataset
\|   |-- genedata.ipynb                          Jupyter Notebook: preprocessing, training, evaluation
\|   |-- model.pkl                               model saved within the Notebook (can be merged with root)
\|   |-- xgboost\_confusion\_matrix.png            confusion matrix visualization
\|   └-- xgboost\_feature\_importance.png          feature importance plot

ENVIRONMENT SETUP

Option1: Python Virtual Environment Setup

macOS / Linux:
   cd /path/to/your/project/app
   python3 -m venv aml_venv
   source aml_venv/bin/activate
   pip install --upgrade pip
   pip install -r requirement.txt
   If you encounter an error related to libomp: brew install libomp
windows:
   cd C:\path\to\your\project\app
   python -m venv aml_venv
   aml_venv\Scripts\activate
   python.exe -m pip install --upgrade pip
   pip install -r requirement.txt

   

Option2: Conda Environment
1. Create and activate a Conda environment:
   conda create -n aml\_pred python=3.8 -y
   conda activate aml\_pred

2. Install dependencies:
   pip install -r requirements.txt
   pip install "streamlit>=1.2.0"

3. Configure environment variables:
   Create a file named api\_key.env in the project root (already included in folder):
   API\_KEY=your\_api\_key\_here



MODEL TRAINING AND EVALUATION (modeling/)

1. Open modeling/genedata.ipynb and run all cells:

   * Load training and test data (data\_set\_ALL\_AML\_train.csv, data\_set\_ALL\_AML\_independent.csv)
   * Handle missing values, normalization or scaling
   * Compute gene means (gene\_means.xlsx)
   * Train XGBoost model
   * Generate performance plots (xgboost\_confusion\_matrix.png, xgboost\_feature\_importance.png)
   * Export model to model.pkl and test probabilities to probs.npz

2. (Optional) Use actual.csv to reproduce predictions and calculate metrics.

WEB APPLICATION (Streamlit >= 1.2.0)

Windows Command Prompt:

1. Open Command Prompt and navigate to the project root:
   cd C:\your\path\to\project\app
if you use Python Virtual Environment: aml_venv\Scripts\activate
2. Run Streamlit:
   streamlit run app.py

macOS Terminal:

1. Open Terminal and navigate to the project root:
   cd /your/path/to/project/app
if you use Python Virtual Environment: source aml_venv/bin/activate
2. Run Streamlit:
   streamlit run app.py
3. In both cases, open the displayed URL (e.g. [http://localhost:8501](http://localhost:8501)) in your browser. Upload an Excel file (same format as the examples) and view the predicted labels and probabilities.

If your interface is in dark mode, please switch to light mode in Settings for the best visual experience.

USAGE NOTES

* Keep model.pkl synchronized with your latest training results.
* Exclude large data and model files from version control by adding them to .gitignore.
* To update the model, retrain in modeling/ and copy the new model.pkl to the root directory.

LICENSE
This material, developed for the University of Sydney and Accenture co-delivered course, remains the intellectual property of the author. It is provided for educational and internal use only. Redistribution, modification, or commercial exploitation without prior written permission from the author is prohibited.
