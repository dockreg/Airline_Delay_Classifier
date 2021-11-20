from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("RandomForest") \
        .getOrCreate()

    # provide path to local file
    path = 'file:///Users/dockreg/ca4022-spark/Data/simple_ml_analysis.csv'

    #read in file to dataframe using the first row as headers
    df = spark.read.csv(path, header='true')

    # convert all columns to integer as all are string type
    df = df.withColumn('Day', df['Day'].cast(IntegerType()))
    df = df.withColumn('Carrier_code', df['Carrier_code'].cast(IntegerType()))
    df = df.withColumn('Flight_num', df['Flight_num'].cast(IntegerType()))
    df = df.withColumn('Origin', df['Origin'].cast(IntegerType()))
    df = df.withColumn('Dest', df['Dest'].cast(IntegerType()))
    df = df.withColumn('Distance', df['Distance'].cast(IntegerType()))
    df = df.withColumn('Is_Delay', df['Is_Delay'].cast(IntegerType()))
    df = df.withColumn('Month', df['Month'].cast(IntegerType()))
    df = df.withColumn('Hour', df['Hour'].cast(IntegerType()))

    #create vector with all features together to be used in pipeline 
    assemblerInputs = ['Day','Month', 'Hour', 'Carrier_code', 'Flight_num', 'Origin', 'Dest', 'Distance']
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol='features')

    # Split the data into training and test sets. 70/30 split
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # Train a Random Forest model using Is_Delay as label.
    rf = RandomForestClassifier(labelCol='Is_Delay', featuresCol='features', seed=42)

    # Chain indexers and model in a Pipeline
    pipeline = Pipeline(stages=[assembler, rf])

    # Train decision tree model
    model = pipeline.fit(trainingData)

    # Make predictions on witheld test data
    predictions = model.transform(testData)

    # Select example rows to display with prediction, raw prediction and features.
    l=['Day','Month','Hour','Carrier_code','Flight_num','Origin','Dest','Distance','prediction','Is_Delay','probability','features','rawPrediction']
    predictions.select(l).show(5)
    
    treeModel = model.stages[1]
    # Explain decision tree and parameters
    print(treeModel)
    
    # evaluations
    print(predictions.groupBy('Is_Delay').count().sort('count').show())
    print(predictions.groupBy('prediction').count().sort('count').show())
    
    # test model accuracy
    evaluator = BinaryClassificationEvaluator(labelCol="Is_Delay")
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    print("Test Area Under PR: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})))
    
    evaluator = MulticlassClassificationEvaluator(labelCol="Is_Delay")
    print("Accuracy: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
    
    spark.stop()
