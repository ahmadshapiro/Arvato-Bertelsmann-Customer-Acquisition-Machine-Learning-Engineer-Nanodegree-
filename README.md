# Machine Learning Engineer Nanodegree
## Capstone Project
### Arvato: Customer Segmentation and Prediction


### Table of Contents

1. [Installation](#installation)
2. [Introduction](#introduction)
3. [Project Motivation](#motivation)
4. [Files](#files)



## Installation <a name="installation"></a>
Besides the libraries included in the Anaconda distribution for Python 3.7 the following libraries have been included in this project:
* `LGBM` -  Gradient Boosting framework based on Microsoft's LGBM tree methods , library valiable to download 
```
pip install lightlgm 
```
* `XGBoost` - optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
```
pip install xgboost 
```
* `CatBoost` - CatBoost is an algorithm for gradient boosting on decision trees. It is developed by Yandex researchers and engineers, and is used for search, recommendation systems, personal assistant, self-driving cars, weather prediction and many other tasks at Yandex and in other companies, including CERN, Cloudflare, Careem taxi.
```
pip install catboost 
```

* `skopt` - simple but computationally expensive library for optimizing hard and complex functions. 
```
pip install skopt 
```



## Introduction <a name="introduction"></a>
This project was made available for Udacity by Arvato.
The goal is to find if there are particular patterns in individuals based on collected data that makes them more likely to be responsive to a mail-order campaign by Arvato for the sale of organic products.

The project is divided in 2 sections:
1. Data Cleaning Notebook :- Assesing and cleaning the datasets  and imputing the missing values using multiple approaches. 
2. Machine Learning Notebook :- Which is divided into two sections

  2.1. Unsupervised Learnign :- Finiding the principle components among the features of the data , and the clustering (Segmenting them) to identify the clusters  of population that are more likely to be customers of the company. 
  2.2 Supervised Learning (Classifcation) :- Using the training set and the previous analysis to build model that predict if the a given person will positivly repsonse or not to a mail marketing campaing of the company. 


## Project Motivation <a name="motivation"></a>
This project provided by Arvato Financial Solutions was one of the available capstone projects. I chose this project mainly for several:

* It's an end to end data science project , starting from data cleaning to clustering (unsupervised learning) to building a supervised learning model (classification). 
* Since it's a kaggle open competition , i have the chance to test my efforts against other's , also it's still open so there's no access for any given solution posted by other. 


## Files <a name="files"></a>
Provided by Arvato(NOT AVAILABLE ONLY THROUGH KAGGLE OR UDACITY):

There are 4 datasets to be explored in this project:
❏Udacity_AZDIAS_052018.csv: Demographics data for the general population of
Germany; 891,211 persons (rows) x 366 features (columns)

❏ Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a
mail-order company; 191,652 persons (rows) x 369 features (columns).

❏ Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who
were targets of a marketing campaign; 42,982 persons (rows) x 367 (columns).

❏ Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were
targets of a marketing campaign; 42,833 persons (rows) x 366 (columns).

And 2 metadata files associated with these datasets:

❏ DIAS Information Levels — Attributes 2017.xlsx​ : list of attributes and
descriptions, organized by informational category

❏ DIAS Attributes — Values 2017.xlsx​ : a detailed mapping of data values for each
feature in alphabetical order.


Avaialble Files :

•tools.py : includes all the helper functions to perform data preprocessing and running the prediction models

•Data Cleaninng.ipynb: Data Assesing and cleaning notebook 

•Machine Learning.ipynb: Clustering and classification notebook.


