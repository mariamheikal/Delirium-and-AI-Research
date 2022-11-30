# Machine Learning Approaches
These notebooks entail the machine learning experiments conducted in an effort to develop a machine learning model capable of predicting an incidence risk score for delirium in ICUs and hospital rooms.

## Traditional Machine Learning Models
### CatBoost Classifier
We train CatBoost classifier on unprocessed data to evaluate its performance on raw data along with training it on data undergoing, categorical variables label encoding and numerical variables normalizing, preporcessing steps. We use CatBoost’s python package which is available for installation through pip.

### Extreme Gradient Boosting Classifier (XGB)
XGBoost does not support non-numerical input variables. Accordingly, the model would be trained on data with ordinal encoding and leave-one-out encoding (LOOE) applied to process categorical variables. In this study, we use the XGBoost’s python library

## Deep Learning Approaches 
These notebooks include experiments conducted using two deep learning specialized architectures, NODE and TransTab, for tabular data. 
### Neural Oblivious Decision Ensembles (NODE)
In this study, we use NODE through the [PyTorch Tabular](https://github.com/manujosephv/pytorch_tabular/tree/main/pytorch_tabular/models/node) wrapper library a which offers varous deep learning models for tabular data. The library offers a learning rate finder module (LRFinder), which can automatically derive the best initial learning rate for the model based on the training data, which we employ to obtain the best learning rate for our model. We search for the best batch size to use with our data through training and evaluating the model using the following batch sizes: 8, 16, 32, and 64. We train the model using Google Colab’s GPU.


### Transferable Tabular Transformers (TransTab)
We use [TransTab’s](https://github.com/RyanWangZf/transtab) python package for our experiments. TransTab includes preprocessing of tabular data implemented within its python package, which is applied on tabular data when reading the dataset through their package’s function. We modify the reading function to allow us to use the dataset without applying the preprocessing implemented within their package, so that we can test the performance of the model on data with and without preprocessing applied. TransTab requires identifying the types of every feature in the dataset as categorical, numerical, or binary features as the model uses this information while tokenizing and embedding its inputs.  <br />

In our experiments, we use TransTab for both supervised learning and transfer learning. We examine if training TransTab on the ICU delirium MIMIC-III derived dataset and then fine-tuning it on the non-ICU delirium AUBMC extracted dataset would improve TransTab’s performance relative to using supervised learning alone. We train the model using Google Colab’s GPU.  <br />

A random search is used to identify the suitable hyperparameters to use for TransTab. The following hyperparameters search space is used: <br />
• Learning Rate: 1e-4, 5e-5, 2e-5  <br />
• Weight Decay: 0, 1e-4  <br />
• Batch Size: 32, 64, 128 <br />
• Patience: 10, 30  <br />
• Number of epochs: 100  <br />
