import yaml, sys, os
from src import app
from src.data_pipeline import tools, preprocess

def run(ss, config):
    data_dir = config["hdfs_dir"] + config["data_dir"] 
    data_path = data_dir + config["data_file"]
    schema = tools.read_json(data_dir + config["schema_file"])
    df = app.load(ss, data_path, schema)
    to_drop, to_transform, to_fill = preprocess.find_nulls(df)
    df, to_fill_num, to_fill_cat = preprocess.clean(df)
    df = preprocess.fill_num(df, to_fill_num)
    df = preprocess.fill_cat(df, to_fill_cat)
    numeric_columns, categorical_columns = tools.sort_columns(df)
    df, cardinalities = preprocess.encode(df, numeric_columns, categorical_columns)
    metadata = tools.get_metadata(numeric_columns, categorical_columns, cardinalities)
    save_dir = data_dir + config["model_name"] + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    tools.write_json(save_dir + config["metadata_path"], metadata)
    app.write(save_dir, df, config["k"])

if __name__ == "__main__":
    config_path = sys.argv[1]
    config_file = open(config_path)
    config = yaml.load(config_file, Loader = yaml.Loader)
    config_file.close()

    ss = app.get_spark(config["appName"])
    run(ss, config)
