import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from utils import acoss_make_avg_delta_dette_par_effectif


def get_dataset(VARIABLES):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.options(inferSchema='True', header="True").csv("/projets/TSF/sources/export_clean.csv")
    df = df.select(*VARIABLES)
    return df

def preprocess(df, TO_SCALE):
    df = acoss_make_avg_delta_dette_par_effectif(df)

    assembler = VectorAssembler(inputCols=TO_SCALE, outputCol="to_scale_features")
    output = assembler.transform(df)
    scaler = StandardScaler(inputCol="to_scale_features", outputCol="scaledFeatures")
    scaledData = scaler.fit(output).transform(output)

    scaledData = scaledData.select([f.col('scaledFeatures').alias('features'),
                                    f.col('outcome').cast('integer').alias('label')])
    return scaledData

def get_model_and_grid(conf):
    if conf.model == 'lr':
        # Logistic regression
        model = LogisticRegression(maxIter=10)
        paramGrid = ParamGridBuilder().addGrid(model.regParam, [0.1, 0.05, 0.01]).build()
    elif conf.model == 'gbt':
        # GBT
        model = GBTClassifier(maxIter=10)
        paramGrid = ParamGridBuilder().addGrid(model.maxDepth, [2, 5])\
            .addGrid(model.maxIter, [10, 100])\
            .build()
    else:
        raise NotImplementedError
    # Pipeline
    pipeline = Pipeline(stages=[model])
    return pipeline, paramGrid

def run():
    from conf_small import VARIABLES, TO_SCALE
    conf ={'model' : 'lr'}
  
    # Get dataset 
    df = get_dataset(VARIABLES)
    # Preprocess (add feature and scale) 
    df = preprocess(df, TO_SCALE)
    # Remove all rows with null values
    df = df.na.drop()

    # Split data into training (80%) and test (20%)
    training, test = df.randomSplit([0.8, 0.2], seed=11)

    pipeline, paramGrid = get_model_and_grid(conf)

    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=5)

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(training)
