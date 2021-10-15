from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from utils import tools


def numeric(df, total, stats, num_features, normalizer = "minmax"):
    def bring_stat(stats, feature):
        keys = ["min", "max", "avg", "median", "mode", "nulls", "stddev_samp"]
        stat = {}
        for key in keys:
            stat[key]= stats["{}({})".format(key, feature)]
        return stat 

    for feature in num_features:
        stat = bring_stat(stats, feature)
        # if mode is not 0, safe to assume missing = 0. (assuming non-negative data)
        if stat["mode"] != 0:
            df = df.fillna(value = 0.0, subset = [feature])
        else: # else if mode is 0, then:
            null_ratio = stat["nulls"] / total
            if null_ratio < 0.1: # if null ratio is low, safe to assume missing = 0
                df = df.fillna(value = 0.0, subset = [feature])
            elif null_ratio > 0.6: # if null percentage is high: use mean
                df = df.fillna(value = stat["avg"], subset = [feature])
            else:  # if null percentage is mid: use median
                df = df.fillna(value = stat["median"], subset = [feature])
        df = df.withColumn(feature, df[feature].cast(FloatType()))
        # # normalizer
        # if normalizer == "minmax": 
        #     df = df.withColumn(feature, (col(feature) - stat['min']) / (stat["max"] - stat["min"]))
        # elif normalizer == "standard":
        #     df = df.withColumn(feature, (col(feature) - stat['avg']) / stat['stddev'])
    return df


def categorical(df, dir, cat_features):
    vocab_dir = dir + "vocabs/"
    for feature in cat_features:
        json_path = vocab_dir + "{}/count.json".format(feature)
        null_feature = tools.read_json(json_path)["null_feature"]
        # go through the column and mark null as the new feature.
        df = df.fillna(value = null_feature, subset = [feature])
    return df