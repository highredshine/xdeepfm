import argparse, yaml
from model_pipeline import app

if __name__ == "__main__":
    # set model module and dataset
    parser = argparse.ArgumentParser(description='pCTR')
    parser.add_argument('task', help='task: train / infer')
    parser.add_argument('config', help='yaml file path')
    args = parser.parse_args()
    config_file = open(args.config)
    config = yaml.load(config_file, Loader = yaml.Loader)
    config_file.close()

    if args.task == "train":
        app.run(config)
    elif args.task == "vocab":
        app.save_vocabs(config)
    print("running complete.")
