# BIS687-project
This repository is for data science capstone project - **Sepsis and ICU Futures: Comprehensive Predictions of ICU Outcomes Through Structured and Narrative Data**

Author: Yining Chen, Yutong Ge, Ivan Wu

Adivisor: Michael Kane

Overview of Sepsis Outcome Prediction Project

Project Aims: This research project focuses on improving the prediction of critical ICU outcomes such as mortality and readmission rates for patients with sepsis using the MIMIC-III database. Our goal is to integrate both structured and unstructured data to uncover insights that can enhance patient management strategies and influence clinical practices.

Innovation: We address gaps in existing ICU outcome prediction methods by leveraging a Fusion-based LSTM model, which combines structured clinical data (e.g., vital signs, lab results) with unstructured data (e.g., clinical notes). This approach aims to provide a richer understanding of sepsis progression, which is critical for accurate and timely predictions.

Specific Aims:

Mortality Prediction: Develop models to predict ICU mortality by integrating structured and unstructured data using advanced machine learning techniques such as Fusion-CNN and Fusion-LSTM.

Readmission Prediction: Extend methodologies to predict hospital readmissions post-sepsis, focusing on key risk factors and long-term health indicators.

Research Strategy: Utilizing the MIMIC-III database, we employ various identification criteria to ensure accurate classification of sepsis patients. Our comprehensive approach includes preprocessing data for normalization and employing deep learning architectures to analyze both the temporal dynamics of clinical metrics and the depth of narrative information contained in clinical notes.

Instructions for Running the Project

Preprocessing

Extract Features: Run get_features.py to gather necessary clinical data from the database.

Initial Preprocessing: Execute preprocess.py to standardize and normalize data, preparing it for further analysis.

Text Data Processing: Use doc2vec.py to transform clinical notes into vectorized formats, enabling their integration into predictive models.

(The preprocessing can be skipped if using uploaded data files, pleaase chnage the direactory in data/preprocessed/files/splits.json to your local directory)

Model Training and Evaluation
Baseline Models: Start with baseline.py to establish baseline predictions using simple machine learning models.

Deep Learning Models: Proceed with main.py to apply advanced Fusion-CNN and Fusion-LSTM models to both mortality and readmission prediction tasks.