# Meta Learning Framework

A framework to apply machine learning on how to combine models (learn to ensemble). It implements a machine learning classifier at the **instance level**, choosing a subset of models to be employed, instead of using a simple bagging method (called *naive ensemble* in this work).

# Table of contents

* [Introduction](#Introduction)
* [Dependencies](#Dependencies)
* [Setup](#Setup)
* [Examples](#Examples)
* [Performance](#Performance)

# Introduction

It contains three main classes: [MetaLearningModel], [BaseModel] and [MetaClassifier]. BaseModel and MetaClassifier are abstract classes to be used as *parents*. They contain attributes and methods that a base model (**classifier or regressor**) and the meta classifier **must** have.

**MetaLearningModel** is the class that does all the work. The fit method will use a cross-validation (or a simple train_test_split validation, depends on the cv param) to create a training set to the meta classifier. In this process, it train every base model and predict a batch of instances, comparing the output with the true values. 

Depending on the task and mode (combiner and error_measure fuction as well), it will select the best base models for each instances in the batch. It can be a multi-class or even a multi-label task. The meta model training set will be composed of instances (as they are in the original problem) and arrays of zeros and ones (targets), indicating which base model were selected or not for each instance (they have number_of_base_models length).

You can see more about this meta learning approach on this paper: https://link.springer.com/chapter/10.1007/978-3-030-61380-8_29.

# Dependencies

* numpy
* sklearn
* pandas
* tqdm
* [sktime](https://github.com/alan-turing-institute/sktime/tree/master/sktime) (Optional - test code)
* [tensorflow](https://github.com/tensorflow/tensorflow) (Optional - test code)

# Setup

Installation:

```
$ pip install meta_learning_framework
```

# Examples

This section presents how to execute some test codes for you to better understand how this framework is supposed to work.

## Classification - Binary Mode

This example uses [sktime](https://github.com/alan-turing-institute/sktime/tree/master/sktime) framework for time series classification.

Run the following commands:

```

$ cd tests/
$ python3 tsc_classification_example.py "dataset's name" binary

```

## Classification - Score Mode

This example uses [sktime](https://github.com/alan-turing-institute/sktime/tree/master/sktime) framework for time series classification as well. The difference to the previous one is the mode. Now, the ref.

Run the following commands:

```

$ cd tests/
$ python3 tsc_classification_example.py "dataset's name" score

```

## Regression - Sklearn regression datasets

This example uses sklearn's regression [datasets](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets). There is a Random Forest classifier which learns to choose the base regressor that will output a prediction with the smallest error possible between all regressors (**only multi-class task**).

Run the following commands:

```

$ cd tests/

```
