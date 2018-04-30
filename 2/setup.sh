#!/bin/bash

echo Setting up Environment!
python3.5 -m venv /scratch/cs434spring2018
source /scratch/cs434spring2018/env_3.5/bin/activate

echo Install needed packages
pip install --upgrade pandas
pip install --upgrade numpy

echo Running KNN
python knn.py

echo Running decision stump
python3.5 DecisionTree.py 1

echo Running decision at with d6
python3.5 DecisionTree.py 6


