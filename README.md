# CA4022 - Spark

## Project intro
We are developing a model to predict flight delays using data from 2019.

## Status
In progress.\
Daniel - Data analysis\
George - Model development

## Current Blockers
No blockers

## Technology used
- Python
- Apache Spark
- Hive
- Neo4j

The following algorithms are tested, using pyspark's machine learning packages, to see which best fit the data:
- Random Forest
- Decisoin Tree
- Multilayer Perceptron
- Naïve Bayes

## Execution of Code
To execute the ML code, download spark and from your spark home execute the following:
```
bin/spark-submit --master local[4] ~path_to_file/ca4022-spark/ML\ models/Multilayer_Perceptron_Classifier.py
```
## Project Contributers
Daniel Sammon, 18364071\
George Dockrell, 18745115

## Results
 

|Model                 | Area Under ROC Curve| Area under PR curve |
| -------------------- | ------------------- | ------------------- |
|Random Forest         |       0.57          |       0.56          |
|Decision Tree         |       0.54          |       0.53          |
|Multilayer Perceptron |       0.52          |       0.52          |
|Naïve Bayes           |       0.49          |       0.49          |
