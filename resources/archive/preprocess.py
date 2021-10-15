from pyspark.sql.functions import array, col, concat, size, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.functions import vector_to_array


def find_nulls(df):
    nulls = {}
    for i in df.columns[1:]:
        nulls[i] = int(df.filter(col(i).isNull()).count() / df.count() * 10000) / 100
    to_drop, to_transform, to_fill = [], [], []
    for feature in nulls:
        if nulls[feature] > 70:
            to_drop.append(feature)
        elif nulls[feature] > 40:
            to_transform.append(feature)
        else:
            to_fill.append(feature)
    return to_drop, to_transform, to_fill


def balance(df):
    # balance
    label_count = df.groupBy("label").count().withColumn("ratio", \
        (col("count") / df.count())).orderBy('label')
    ratio = label_count.collect()[1]['ratio'] / label_count.collect()[0]['ratio']
    clicked = df.filter(col("label") == 1)
    unclicked = df.filter(col("label") == 0).sample(False, ratio)
    df = clicked.union(unclicked)
    return df


def cleanup(df, to_drop, to_transform):
    # drop
    df = df.drop(*to_drop) # * : for each
    # transform to boolean
    for feature in to_transform:
        df = df.withColumn(feature, when(df[feature].isNull(), 0).otherwise(1).cast('string'))
    return df


def fill_num(df, to_fill_num):
    stats = {}
    described = df.select(to_fill_num).describe().collect()
    for feature in to_fill_num:
        stats[feature] = {}
        stats[feature]["count"] = int(described[0][feature])
        stats[feature]["mean"] = round(float(described[1][feature]), 2)
        stats[feature]["stddev"] = round(float(described[2][feature]), 2)
        stats[feature]["q1"], stats[feature]["q2"], stats[feature]["q3"] \
            = df.approxQuantile(feature, [0.25, 0.50, 0.75], 0)
        iqr = stats[feature]['q3'] - stats[feature]['q1']
        stats[feature]['lower'] = stats[feature]['q1'] - (iqr * 1.5)
        stats[feature]['upper'] = stats[feature]['q3'] + (iqr * 1.5)

    for feature in to_fill_num:
        st = stats[feature]
        # fill nulls with median and ceil negative values
        df = df.withColumn(feature, when(col(feature).isNull(), st['q2']).\
            when(col(feature) < 0, 0).otherwise(col(feature)))
        # replace outliers with median (again, mean is skewed)
        df = df.withColumn(feature, when((col(feature) > st['upper']) | \
            (col(feature) < st['lower']), st['q2']).otherwise(col(feature)))
        # standardize (z-scoring)
        df = df.withColumn(feature, (col(feature) - st['mean']) / st['stddev'])
    return df


def fill_cat(df, to_fill_cat):
    # fill nulls with mode for categorical features
    label = df["label"]
    clicked = df.filter(label == 1)
    unclicked = df.filter(label == 0)
    for feature in to_fill_cat:
        clicked_mode = clicked.filter(col(feature).isNotNull()).groupBy(feature).count().\
            orderBy("count", ascending=False).first()[0]
        unclicked_mode = unclicked.filter(col(feature).isNotNull()).groupBy(feature).count().\
            orderBy("count", ascending=False).first()[0]
        df = df.withColumn(feature, when(col(feature).isNull() & (label == 1), clicked_mode).\
            when(col(feature).isNull() & (label == 0), unclicked_mode).otherwise(col(feature)))
    return df


def encode(df, numeric_columns, categorical_columns):
    # pipeline for transforming categorical features
    # string indexer
    original_cols = categorical_columns
    indexed_cols = [s + "_indexed" for s in categorical_columns]
    stringIndexer = StringIndexer(inputCols = original_cols, outputCols = indexed_cols)
    df = stringIndexer.fit(df).transform(df)
    # one-hot encoder
    encoded_cols = [s + "_vec" for s in categorical_columns]
    ohe = OneHotEncoder(inputCols = indexed_cols, outputCols = encoded_cols, dropLast = False)
    df = ohe.fit(df).transform(df)
    df = df.drop(*original_cols).drop(*indexed_cols)

    # clean up columns and obtain metadata for each feature
    cardinalities = dict(zip(numeric_columns, [1] * len(numeric_columns)))
    for feature, encoded in zip(original_cols, encoded_cols):
        df = df.withColumnRenamed(encoded, feature).withColumn(feature, vector_to_array(feature))
        cardinalities[feature] = df.select(size(feature).alias("n")).collect()[0]['n']
    # concatenating cateogircal columns and creating dense numeric column
    df = df.select(['label'] + 
        [array(numeric_columns).alias('numeric'), concat(*categorical_columns).alias('categorical')])
    df = df.select(['label'] + 
        [concat(*['numeric', 'categorical']).alias("features")])
    return df, cardinalities
