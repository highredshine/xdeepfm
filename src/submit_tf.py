import os, sys, yaml
from data_pipeline import app
from utils import tools

def run(ss, config):
    schema, metadata, part_dirs = app.setup(config)
    for part_dir in part_dirs:
        print(part_dir)
        data_path = tools.get_subpaths(part_dir + "raw/part*.txt")[0]
        df = app.load(ss, data_path, schema)
        app.run_to_tfrecord(df, part_dir, metadata)


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_file = open(config_path)
    config = yaml.load(config_file, Loader = yaml.Loader)
    config_file.close()

    if config["local"]:
        os.environ['SPARK_HOME'] = config["root_dir"] + config["spark_home_dir"]
    ss = app.get_spark(config["appName"], connector = config["root_dir"] + config["connector"])
    run(ss, config)
