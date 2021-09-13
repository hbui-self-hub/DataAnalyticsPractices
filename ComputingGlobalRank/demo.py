from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, functions as sf, Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# kudos to Mr.Sushil Kumar for "Computing global rank of a row"
# https://medium.com/swlh/computing-global-rank-of-a-row-in-a-dataframe-with-spark-sql-34f6cc650ae5

data = [
    ("James","Smith","2","36636","M",3000),
    ("Michael","Kors","12","40288","M",4000),
    ("Robert","Williams","13","42114","M",4000),
    ("Maria","Jones","15","39192","F",4000),
    ("Smith","Brown","29","5961","F",-1),
    ("Rose","Armi","30","31392","F",-1),
    ("Williams","Sharter","46","1203","F",-2000),
    ("Jen","Loper","71","85012","F",12000),
    ("Anne","Cross","73","592","F",3500),
    ("Mary","Poppins","75","6815","F",2200),
    ("Emma","Watson","84","98213","F",1950),
    ("York","Billy","100","3804","F",8460),
]

schema = StructType([
    StructField("firstname", StringType(),True),
    StructField("lastname", StringType(),True),
    StructField("id", StringType(), True),
    StructField("point", StringType(),True),
    StructField("gender", StringType(), True),
    StructField("salary", IntegerType(), True),
])

sort_column = "salary"

conf = SparkConf().setAppName('Global Ranking')
context = SparkContext(conf=conf).getOrCreate()
sqlContext = SQLContext(context)

df = sqlContext.createDataFrame(data=data,schema=schema)

# Using orderBy operation to sort value
# Note: orderBy operation not only each partition is sorted from within but also among each other, all_values(partition-0) < all_values(partition-1) < all_values(partition-2).
partDf = df.orderBy(sort_column).withColumn("partitionId", sf.spark_partition_id())

# rank row on each partition (rank local)
window = Window.partitionBy("partitionId").orderBy(sort_column)
rankDf = partDf.withColumn("local_rank", sf.rank().over(window))

# calculate sum_factor 
tempDf = rankDf.groupBy("partitionId").agg(sf.max("local_rank").alias("max_rank"))

w = Window.orderBy("partitionId").rowsBetween(Window.unboundedPreceding, Window.currentRow)
statsDf = tempDf.withColumn("cum_rank", sf.sum("max_rank").over(w))

sumFactorDf = statsDf.withColumn('sum_factor',sf.col('cum_rank')-sf.col('max_rank'))

# calculate global rank = local_rank + sum_factor by joining on partitionId
finalDf = rankDf.join(sf.broadcast(sumFactorDf), rankDf.partitionId == sumFactorDf.partitionId, "inner").withColumn("rank", sf.col("local_rank") + sf.col("sum_factor"))
finalDf.sort(sf.asc("rank")).show()
