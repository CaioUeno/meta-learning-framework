# Meta Learning Framework

A framework to apply machine learning on how to combine models (learn to ensemble).

# Table of contents

* [Introduction](#Introduction)
* [Dependencies](#Dependencies)
* [Setup](#Setup)
* [Examples](#Examples)
* [Performance](#Performance)

# Introduction

# Dependencies

# Setup

Installation:

```
$ pip3 install meta_learning_framework
```

# Examples

This section presents how to execute some test codes for you to better understand how this framework is supposed to work.

## Classification - Binary Mode

This example uses sktime framework for time series classification. 

Run the following commands:

```

$ cd tests/
$ python3 tsc_classification_example.py "dataset's name" binary

```

## Classification - Score Mode

This example uses sktime framework for time series classification as well. The diffesrence to the previous one is the mode. Now, the ref.

Run the following commands:

```

$ cd tests/
$ python3 tsc_classification_example.py "dataset's name" score

```

## Regression - Sklearn regression datasets

This example uses sklearn's regression datasets.

Run the following commands:

```

$ cd tests/

```
