# Meta Learning Framework

A framework to apply machine learning on how to combine models (learn to ensemble). It implements a machine learning classifier at the **instance level**, choosing a subset of models to be employed, instead of using a simple bagging method (called *naive ensemble* in this work).

## Table of contents

* [Introduction](#Introduction)
* [Dependencies](#Dependencies)
* [Installation](#Installation)
* [Examples](#Examples)
  * [Classification - Binary](#Classification---Binary-Mode)
  * [Classification - Score](#Classification---Score-Mode)
  * [Regression](#Regression-Score-mode-only)
  * [Forecasting](#Forecasting)
* [Performance](#Performance)

## Introduction

It contains three main classes: [MetaLearningModel](https://github.com/CaioUeno/meta-learning-framework/blob/master/meta_learning_framework/meta_learning_model.py), [BaseModel](https://github.com/CaioUeno/meta-learning-framework/blob/master/meta_learning_framework/base_model.py) and [MetaClassifier](https://github.com/CaioUeno/meta-learning-framework/blob/master/meta_learning_framework/meta_classifier.py). BaseModel and MetaClassifier are abstract classes to be used as *parents*. They contain attributes and methods that a base model (**classifier or regressor**) and the meta classifier **must** have.

**MetaLearningModel** is the class that does all the work. The fit method will use a cross-validation (or a simple train_test_split validation, depends on the cv param) to create a training set to the meta classifier. In this process, it train every base model and predict a batch of instances, comparing the output with the true values.

Depending on the task and mode (**combiner** and **error_measure** fuction as well), it will select the best base model(s) for each instances in the batch. It can be a multi-class or even a multi-label task. The meta model training set will be composed of **instances** (as they are in the original problem) and **targets** (arrays of zeros and ones), indicating which base model(s) were selected or not for each instance (they have number_of_base_models length).

In the prediction step, first the **MetaLearningModel** will predict the instance choosing which base models are going to be used. Then, **only those selected bse models** are going to predict the given instance. Finally, their outputs is combined using the **combiner** fuction.

You can see more about this meta learning approach on this paper: <https://link.springer.com/chapter/10.1007/978-3-030-61380-8_29>.

## Dependencies

* numpy
* sklearn
* pandas
* tqdm
* [sktime](https://github.com/alan-turing-institute/sktime/tree/master/sktime) (optional - test code)
* [dtw-python](https://github.com/DynamicTimeWarping/dtw-python) (optional - test code)
* [tensorflow](https://github.com/tensorflow/tensorflow) (optional - test code)

## Installation

Simply run:

```bash
pip install meta_learning_framework
```

You can also clone the repository and run:

```bash
cd meta-learning-framework/
pip install .
```

## Examples

This section presents how to execute some test codes for you to better understand how this framework is supposed to work.

### Classification - Binary Mode

This example uses [sktime](https://github.com/alan-turing-institute/sktime/tree/master/sktime) framework for time series classification. Binary mode indicates that when creating the meta classifier training set, base models that correctly predict instances' class will be selected (soft selection). Notice that it can imply a multi-label classification task.

Run the following commands:

```bash
cd tests/
python3 tsc_classification_example.py sktime_dataset_name binary
```

### Classification - Score Mode

This example uses [sktime](https://github.com/alan-turing-institute/sktime/tree/master/sktime) framework for time series classification as well. The difference to the previous one is the mode. Now, the base model that outputs the best score distribution will be choosen. It implies only a multi-class classification task.

Run the following commands:

```bash
cd tests/
python3 tsc_classification_example.py sktime_dataset_name score
```

### Regression (Score mode only)

This example uses sklearn's regression [datasets](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets). There is a Random Forest classifier which learns to choose the base regressor that will output a prediction with the smallest error possible between all regressors (**multi-class task**).

Run the following commands:

```bash
cd tests/
python3 sklearn_regression_example.py
```

### Forecasting

This example is the most complex. It is a regression task, but since is a forecasting, **you can not use cross-validation directly**. So it uses a generator of train/test split indexes, regarding instances order in time. Also, it is a multi output regression task, since it tries to predict two future values.

Run the following commands:

```bash
cd tests/
python3 forecasting_example.py
```

## Performance

### Classification - Both binary and score

Running the script [sktime_classification_time.py](https://github.com/CaioUeno/meta-learning-framework/blob/master/tests/sktime_classification_example.py) these are the accuracy for a couple of datasets:

|    Sktime Dataset   | Naive Ensemble - Accuracy | Binary - Accuracy | Score - Accuracy  | Binary - models used |
| :-----------------: | :-----------------------: | :---------------: | :---------------: | :------------------: |
| Ham                 | **.63**                   | .62               | .56               |  0.66                |
| PowerCons           | .95                       | **.99**           | .81               |  2.00                |
| MoteStrain          | .94                       | .94               | .81               |  0.00                |
| MedicalImages       | .74                       | **.75**           | .57               |  0.00                |
| DistalPhalanxTW     | **.67**                   | .65               | .58               |  0.32                |
| InsectWingbeatSound | .61                       | .61               |                   |  0.00                |

### Simple regression task

Running the script [sklearn_regression_example.py](https://github.com/CaioUeno/meta-learning-framework/blob/master/tests/sklearn_regression_example.py) using **fetch_california_housing** dataset the following results were achieved:

| Model                 | MAE    | R <sup>2</sup>  |
| :-------------------: | :----: | :-------------: |
| Meta Model            | 0.4678 | 0.6927          |
| Naive Ensemble        | 0.6681 | 0.4622          |
| Individual - Linear   | 0.5414 | 0.6047          |
| Individual - SVR      | 0.8743 | -0.0284         |
| Individual - 3NN      | 0.8384 | 0.1015          |
| Individual - AdaBoost | 0.8083 | 0.3669          |

### Forecasting task

Running the script [forecasting_example.py](https://github.com/CaioUeno/meta-learning-framework/blob/master/tests/forecasting_example.py) using a white noise time series:

| Model                      | Mean Absolute Sum Error    | Mean Absolute Weighted Error |
| :------------------------: | :------------------------: | :--------------------------: |
| Meta Model                 | 0.2554                     | 0.1527                       |
| Naive Ensemble             | 0.3023                     | 0.1720                       |
| Individual - Linear        | 0.2516                     | 0.1510                       |
| Individual - 3NN           | 0.4176                     | 0.2228                       |
| Individual - Random Forest | 0.2698                     | 0.1589                       |
