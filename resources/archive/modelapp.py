
def load_tfrecords(part_dirs, config):
    filenames = []
    for part_dir in part_dirs:
        # load tfrecord partitions
        filenames.append(tools.get_subpaths(part_dir + config["record_dir"] + "part*"))
    k = config["k"] # number of bins to bucketize datasets
    if k == 1:
        return [path for f in filenames for path in f]
    tfrecords = [[] for _ in range(k)]
    for i, f in enumerate(filenames):
        tfrecords[i % k] += f
    return tfrecords


# def save_stats(config):
#     data_dir = config["root_dir"] + config["data_dir"]
#     metadata = tools.read_json(data_dir + "metadata.json")
#     num_fields = tools.num_fields(config, metadata)
#     part_dirs = tools.get_subpaths(data_dir + config["target_dir"] + "part*/")
#     print(part_dirs)
#     stats_dir = data_dir + config["full_dir"] + "stats/"
#     tools.makedir(stats_dir)
#     stats = {}
#     for part_dir in part_dirs:
#         cnt = tools.read_json(part_dir + "count/*.json", glob = True)["total"]
#         part_stats = tools.read_json(part_dir + "stats/*.json", glob = True)
#         for key in part_stats:
#             if key not in stats:
#                 stats[key] = 0
#             else:
#                 if "stddev"
#     tools.write_json(stats_dir + "stats.csv", stats)
    


    filenames = [] 
    for j in indices: # concatenate train tfrecords
        filenames += tfrecords[j]
    print("train")
    train_data = get_dataset(filenames, features, num_fields, cat_fields).shuffle(
        config["model_params"]["batch_size"], 
        reshuffle_each_iteration = True
    )
        # feature_description[feature["name"]] = tf.io.FixedLenFeature(
        #     shape = feature["len"], 
        #     dtype = feature["type"],
        #     default_value = [feature["default_value"]] * feature["len"]
        # )



            # (
                # CategoryEncoding( # sparse one-hot encoding for linear input
                #     max_tokens = vocab_sizes[field] + 1,
                #     output_mode = "binary",
                #     sparse = True
                # ),