import json, re, csv, glob, math, random, string, os

def get_subpaths(path):
    return glob.glob(path)

def read_csv(path):
    csvfile = open(path, newline='')
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    return list(reader)

def read_vocab_csv(path, glob = False):
    if glob:
        reader = read_csv(get_subpaths(path)[0])
    else:
        reader = read_csv(path)
    vocabs = [row[0] for row in reader]
    return vocabs
    
def write_csv(path, data):
    f = open(path, 'w+')
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()

def write_pd_csv(path, data, dict = False):
    import pandas as pd
    if dict:
        df = pd.DataFrame(data)
        df.to_csv(path, header = False, index = False)
    else:
        df = pd.DataFrame({"" : data})
        df.to_csv(path, header = False, index = False)
    
def read_json(path, glob = False):
    if glob:
        return json.load(open(get_subpaths(path)[0]))
    else:
        return json.load(open(path))

def write_json(path, data):
    file = open(path, "w+")
    file.write(json.dumps(data)) 
    file.close()


def flatten_datasets(bins, indices):
    datasets  = []
    for j in indices:
        for dataset in bins[j]:
            datasets.append(dataset)
    return datasets

def concat_datasets(datasets):
    data = datasets[0]
    for dataset in datasets[1:]:
        data = data.concatenate(dataset)
    return data

def get_fields(config, metadata):
    fields = config["features"]
    if len(fields) == 0:
        fields = metadata["num_features"] + metadata["cat_features"]
    return fields

def get_x(batch, fields):
    return {field : batch[field] for field in fields}

def load_full_vocabs(config, cat_fields):
    if len(cat_fields) == 0:
        return {}
    vocab_dir = config["root_dir"] + config["data_dir"] + config["full_dir"] + "vocabs/"
    vocabs = {}
    for field in cat_fields:
        field_dir = vocab_dir + "{}/".format(field)
        vocabs[field] = read_vocab_csv(field_dir + "vocabs.csv")
    return vocabs

def load_part_stats(part_dir, num_fields):
    if len(num_fields) == 0:
        return {}
    stats = read_json(part_dir + "stats/*.json", glob = True)
    return stats

def num_fields(config, metadata):
    fields = []
    if len(config["features"]) == 0:
        fields = metadata["num_features"]
    else:
        for field in config["features"]: 
            if field in metadata["num_features"]:
                fields.append(field)
    return fields

def cat_fields(config, metadata):
    fields = []
    if len(config["features"]) == 0:
        fields = metadata["cat_features"]
    else:
        for field in config["features"]: 
            if field in metadata["cat_features"]:
                fields.append(field)
    return fields

def makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def generate_cat(size):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(size))

def sqrt(x):
    return math.sqrt(x)

def write_tfrecord(dir, df):
    df.write.format("tfrecords").option("recordType", "Example").save(dir)

def filter_columns(df):
    numeric_columns, categorical_columns = [], []
    for field in df.schema.fields:
        if field.name == 'label':
            continue
        if str(field.dataType) in ('IntegerType', 'DoubleType'):
            numeric_columns.append(field.name)
        elif str(field.dataType) in ('StringType'):
            categorical_columns.append(field.name)
    return numeric_columns, categorical_columns

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def sort_columns(df):
    numeric_columns, categorical_columns = filter_columns(df)
    return natural_sort(numeric_columns), natural_sort(categorical_columns)

def get_metadata(numeric_columns, categorical_columns, cardinalities):
    metadata = {}
    metadata["numeric_fields"] = numeric_columns
    metadata["categorical_fields"] = categorical_columns
    metadata["num_fields"] = len(categorical_columns) + len(numeric_columns)
    metadata["num_features"] = sum(list(cardinalities.values()))
    metadata["cardinalities"] = cardinalities
    return metadata


def concat_vocabs(part_dirs, fields):
    vocabs = {}
    if len(fields) == 0:
        return vocabs
    for field in fields:
        vocabs[field] = []
    for part_dir in part_dirs:
        vocab_dir = part_dir + "vocabs/"
        # unionize vocab list for each partition, for each feature
        for field in fields:
            field_dir = vocab_dir + field + "/"
            print(field_dir)
            vocab_file = get_subpaths(field_dir + "*.csv")[0]
            vocab_list = read_vocab_csv(vocab_file)
            vocabs[field] = list(set(vocabs[field]) | set(vocab_list))
    return vocabs

