# Incidence of delirium in different clinical settings

Delirium is an acute state of brain failure characterized by unexpected disorientation, a fluctuating course, inattention, and frequently an abnormal degree of awareness. Although delirium is a common illness, it is often overlooked in hospitals. Due to the diverse and fluctuating nature of delirium, as well as its overlap with other psychiatric diseases, it is hard for healthcare professionals to identify delirium patients. This study aims to allow prompt delirium prevention and intervention in hospitalized patients by developing an EHR-based machine learning model capable of accurately predicting incident delirium. In this study, traditional machine learning, deep learning, and transfer learning models will be used to determine the best machine learning model for predicting the risk score of delirium using our datasets. We then propose the best model for predicting the delirium risk score based on our evaluation of the models' performance. Prior studies on delirium used private datasets restricted to certain hospitals or departments. To achieve the goal of developing a model for predicting delirium risk score, we derive two distinct datasets for ICU delirium and delirium in hospitalized patients in order to determine the prevalence of delirium in each clinical context independently and to identify the clinical features that may serve as delirium indicators in each clinical context. In order to avoid the progression of delirium, a proposed set of preventative interventions will be automatically included in the treatment plan of hospitalized patients based on their delirium risk score. This order set would be included in the model's output depending on clinical team feedback.

## ICU Delirium MIMIC-III Derived Dataset
The ICU Delirium MIMIC-III Dataset Notebook includes a comprehensive derivation process for a novel dataset aimed at investigating the incidence of delirium in intensive care units (ICUs).

The following 10 tables are used to extract demographic and clinical information regarding the patients. All tables are provided as CSV files.
1. PATIENTS
2. ADMISSIONS
3. ICUSTAYS
4. CHARTEVENTS
5. D ITEMS
6. OUTPUTEVENTS
7. INPUTEVENTS CV
8. INPUTEVENTS MV
9. DIAGNOSIS ICD
10. D ICD DIAGNOSES

To replicate this dataset, repeat the steps outlined in the aforementioned notebook.
To extract more clinical features, we recommend following the text similarity technique outlined in the manuscript in order to add additional features and reduce the percentage of missing values within the features.
To generate a dataset for a different disease or disorder, replace the delirium ICD-9 codes in the notebook with the corresponding ICD-9 codes.
