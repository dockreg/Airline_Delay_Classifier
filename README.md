# CA4022 - Spark

## Project intro
We are developing a model to predict US flight delays using data from 2019 found on kaggle [here](https://www.kaggle.com/sherrytp/airline-delay-analysis). We classify delays as any flight that leaves >15 mins after its departure time. The columns we selected/created as features are:

- Day
- Month
- Hour
- Carrier code
- Flight number
- Origin 
- Destination
- Distance


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
- Decision Tree
- Multilayer Perceptron
- Naïve Bayes

We also tested the use of undersampling as the dataset was extremely imbalanced.

## Execution of Code
To execute the ML code, download spark and from your spark home execute the following:
```
bin/spark-submit --master local[4] ~path_to_file/ca4022-spark/ML\ models/Multilayer_Perceptron_Classifier.py
```
## Project Contributers
Daniel Sammon, 18364071\
George Dockrell, 18745115

## Results
 

| Model                    |    Accuracy   | Area Under ROC Curve| Area under PR curve |
| -----------------------  | ------------- | ------------------- | ------------------- |
|Random Forest             |     0.82      |      **0.67**       |     **0.32**        |
|Decision Tree             |   **0.83**    |        0.60         |       0.22          |
|Multilayer Perceptron     |     0.82      |        0.52         |       0.18          |
|Naïve Bayes               |     0.53      |        0.48         |       0.17          |

| Model with undersampling |    Accuracy   | Area Under ROC Curve| Area under PR curve |
| -----------------------  | ------------- | ------------------- | ------------------- |
|Random Forest             |   **0.63**    |      **0.68**       |     **0.65**        |
|Decision Tree             |   **0.63**    |        0.62         |       0.61          |
|Multilayer Perceptron     |     0.51      |        0.51         |       0.51          |
|Naïve Bayes               |     0.52      |        0.48         |       0.48          |