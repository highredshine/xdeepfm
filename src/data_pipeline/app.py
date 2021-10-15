from pyspark.sql import SparkSession
from pyspark.sql.types import *
from utils import tools
from data_pipeline.spark import eda, fill


def get_spark(appName, connector = ""):
    if connector:
        return SparkSession.builder.appName(appName).config('spark.jars', connector).getOrCreate()
    else:
        return SparkSession.builder.appName(appName).getOrCreate()

def setup(config):
    data_dir = config["root_dir"] + config["data_dir"] 
    schema = tools.read_json(data_dir + config["schema_file"])
    metadata = tools.read_json(data_dir + config["metadata_file"])
    part_dirs = tools.get_subpaths(data_dir + config["target_dir"] + "part*/")
    return schema, metadata, part_dirs

def load(ss, path, schema):
    return ss.read.option("header", "false").option("delimiter", "\t").schema(StructType.fromJson(schema)).csv(path)

def select(df, columns):
    return df.select(*columns)
    
def write(dir, df, k):
    weights = [1 / k] * k
    for i, fold in enumerate(df.randomSplit(weights)): 
        path = dir + "fold{}.tfrecord".format(i + 1)
        fold.write.format("tfrecords").option("recordType", "Example").save(path)

def run_eda(df, dir, metadata): 
    print("count")
    eda.count(dir + "count/", df)
    print("stats")
    eda.stats(dir + "stats/", df, metadata["num_features"])
    print("vocab")
    eda.vocabs(dir + "vocabs/", df, metadata["cat_features"])

def run_to_tfrecord(df, dir, metadata):
    # TODO: criteo is split to one file but should accomodate many scenarios
    print("numeric")
    stats = tools.read_json(dir + "stats/stats.json")
    total = tools.read_json(dir + "count/count.json")["total"]
    df = fill.numeric(df, total, stats, metadata["num_features"])
    print("categorical")
    df = fill.categorical(df, dir, metadata["cat_features"])
    tools.write_tfrecord(dir + "data/", df)