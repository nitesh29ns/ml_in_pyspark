import os

os.environ['SPARK_HOME'] = "D:\ml\pyspark\env\Lib\site-packages\pyspark" 

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder






spark = SparkSession.builder.appName("logisticregression").getOrCreate()

df = spark.read.csv("adult.csv", inferSchema=True, header=True)

cols = df.columns

cols.remove("fnlwgt")
cols.remove("education-num")
cols.remove("capital-gain")
cols.remove("capital-loss")

#print(cols)

new_df = df.select(*cols)

encoded_columns = [i+"_vector" for i in cols]
stringIndexer = StringIndexer(inputCols=cols, outputCols=encoded_columns)

data = stringIndexer.fit(new_df).transform(new_df)

#data.show(3)

req_data = data.select(*encoded_columns)

#req_data.show(1)

encoded_columns.remove("salary_vector")

assembler = VectorAssembler(inputCols=encoded_columns, outputCol="features")

data = assembler.transform(req_data)

#data.show(1)

data = data.select("features","salary_vector")

(train,test) = data.randomSplit([0.8,0.2])


lr = LogisticRegression(labelCol="salary_vector", featuresCol="features",maxIter=10)

train_model = lr.fit(train)

pred = train_model.transform(test)

eval = MulticlassClassificationEvaluator(labelCol="salary_vector", predictionCol="prediction",metricName="accuracy")
accuracy = eval.evaluate(pred)

print(accuracy)

train_model.save("adult_census_model")