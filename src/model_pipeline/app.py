from utils import tools
from .models import xdeepfm
import tensorflow as tf

def get_model(config, num_fields, cat_fields, vocab_sizes):
    tf.random.set_seed(config["seed"])
    if config["model"] == "xdeepfm":
        return xdeepfm.Model(config, num_fields, cat_fields, vocab_sizes)

def dummy_feature(field):
    return  tf.io.FixedLenFeature(
        shape = field["len"], 
        dtype = field["type"],
        default_value = [field["default_value"]] * field["len"]
    )

def define_feature_columns(part_dir, fields, num_fields, metadata):
    stats = tools.load_part_stats(part_dir, metadata["num_features"])
    using_features = []
    dummy_features = {}
    for field in fields:
        name = field["name"]
        if name in metadata["num_features"]:
            if name in num_fields: # normalize numeric fields (used in the model)
                mean_key = "avg({})".format(name)
                mean = stats[mean_key] if mean_key in stats else 0
                stddev_key = "stddev_samp({})".format(name)
                stddev = stats[stddev_key] if stddev_key in stats else 1
                def normalize(x):
                    return (x - mean) / stddev
                using_features.append(tf.feature_column.numeric_column(
                    key = name, normalizer_fn = normalize, default_value = 0.0
                ))
            else: # add dummy Feature object just for the use of parsing
                dummy_features[field["name"]] = dummy_feature(field)
        else: # add dummy Feature object just for the use of parsing
            dummy_features[field["name"]] = dummy_feature(field)
    features = {**tf.feature_column.make_parse_example_spec(using_features), **dummy_features}
    return features

def get_dataset(config, part_dir, fields, num_fields, metadata):
    # get tfrecord paths for a partition
    filenames = tools.get_subpaths(part_dir + config["record_dir"] + "part*")
    # build feature columns (stats from partition, vocabs from full)
    features = define_feature_columns(part_dir, fields, num_fields, metadata)
    def parse(serialized):
        return tf.io.parse_example(serialized, features)
    return tf.data.TFRecordDataset(filenames = filenames).map(parse)

def train(config, train_datasets, num_fields, cat_fields, vocab_sizes, fold_dir):
    model = get_model(config, num_fields, cat_fields, vocab_sizes)
    train_data = tools.concat_datasets(train_datasets).\
        repeat(model.epochs).\
        batch(model.batch_size, drop_remainder=True)
    batch_num = 0
    history = {"loss" : [], "auc" : []}
    for batch in train_data:
        x = tools.get_x(batch, num_fields + cat_fields)
        y = batch[config["label"]["name"]]
        hist = model.fit(x, y, 
            batch_size = model.batch_size, 
            verbose = 2, 
            class_weight = config["class_weight"],
        )
        result = hist.history
        if tf.math.is_nan(tf.constant(result["loss"])):
            break
        history["loss"].append(result["loss"])
        history["auc"].append(result["auc"])
        batch_num += 1
        if batch_num % config["steps"] == 0:
            tools.write_pd_csv(
                fold_dir + "history/history{}.csv".format(batch_num % config["steps"]),
                history, 
                dict = True
            )
            print(history)
        print("parsing data...")
    # model.save_weights(fold_dir + "checkpoints/checkpoint{}".format(batch_num), save_format='tf')
    return model, batch_num

def test(model, test_datasets, config, num_fields, cat_fields, fold_dir):
    test_data = tools.concat_datasets(test_datasets).\
        batch(model.batch_size * config["steps"], drop_remainder=True)
    batch_num = 0
    for batch in test_data:
        x = tools.get_x(batch, num_fields + cat_fields)
        y = batch[config["label"]["name"]]
        log_loss, auc = model.evaluate(x, y,
            batch_size = model.batch_size, 
        )
        result = {"loss" : log_loss, "auc" : auc}
        if tf.math.is_nan(tf.constant(result["loss"])):
            break
        tools.write_pd_csv(fold_dir + "evals/validation{}.csv".format(batch_num), result, dict = True)
        batch_num += 1
        print("parsing data...")

def run_kfold_validation(config, part_dirs, fields, metadata):
    k = config["k"]
    bins = [[] for _ in range(k)]
    # for each partition
    num_fields = tools.num_fields(config, metadata) # select numeric fields
    cat_fields = tools.cat_fields(config, metadata) # select categorical fields
    vocabs = tools.load_full_vocabs(config, cat_fields)
    for i, part_dir in enumerate(part_dirs): 
        # parse a tf.dataset out of this partition
        dataset = get_dataset(config, part_dir, fields, num_fields, metadata)
        # use modulo to bin each partition data to k-folds.
        bins[i % k].append(dataset)
    # running k-folds
    history_dir = config["root_dir"] + config["model_dir"] + config["model"] + "/" + config["data"] + "/" 
    tools.makedir(history_dir)
    for i in range(k):
        indices = list(range(k))
        indices.remove(i)
        fold_dir = history_dir + "fold{}/".format(i)
        tools.makedir(fold_dir)
        tools.makedir(fold_dir + "history/")
        tools.makedir(fold_dir + "checkpoints/")
        tools.makedir(fold_dir + "evals")
        # run train
        model = None
        train_datasets = tools.flatten_datasets(bins, indices)
        if len(train_datasets) != 0:
            print("training for fold {}...".format(i))
            model, batch_num = train(config, train_datasets, num_fields, cat_fields, vocabs, fold_dir)
        # testing the i-th fold
        if model != None:
            print("testing for fold {}...".format(i))
            model.load_weights(fold_dir + "checkpoints/checkpoint{}".format(batch_num))
            test_datasets = tools.flatten_datasets(bins, [i])
            test(model, test_datasets, config, num_fields, cat_fields)

def run(config):
    data_dir = config["root_dir"] + config["data_dir"]
    fields = tools.read_json(data_dir + "features.json")["fields"]
    metadata = tools.read_json(data_dir + "metadata.json")
    part_dirs = tools.get_subpaths(data_dir + config["target_dir"] + "part*/")
    run_kfold_validation(config, part_dirs, fields, metadata)

def save_vocabs(config):
    data_dir = config["root_dir"] + config["data_dir"]
    metadata = tools.read_json(data_dir + "metadata.json")
    part_dirs = tools.get_subpaths(data_dir + config["target_dir"] + "part*/")
    print(part_dirs)
    vocabs = tools.concat_vocabs(part_dirs, metadata["cat_features"])
    vocab_dir = data_dir + config["full_dir"] + "vocabs/"
    tools.makedir(vocab_dir)
    for field in vocabs:
        field_dir = vocab_dir + "{}/".format(field)
        tools.makedir(field_dir)
        tools.write_pd_csv(field_dir + "vocabs.csv", vocabs[field])