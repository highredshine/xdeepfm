import csv
import glob
import logging
import os.path

import numpy as np
import tensorflow as tf

from utils import multiprocess
from features import generator
from apps.wrapper import AppWrapper
from parsers.configs.data import DataConfig


ROW_PER_FILE = 200000

logger = logging.getLogger("util.csv_to_tfrecord")


def to_tfrecord(src_path, n_skip_line, names, dtypes, dst_dir, delimiter=","):
    """convert csv to tfrecord without additional processing.
    - set feature type based on `dtypes`. see `features.generator.create_feature`
    
    Arguments:
        src_path {[type]} -- [description]
        n_skip_line {[type]} -- [description]
        names {[type]} -- [description]
        dtypes {[type]} -- [description]
        dst_dir {[type]} -- [description]
    
    Keyword Arguments:
        delimiter {str} -- [description] (default: {","})
    """
    if delimiter == "\\t":
        delimiter = "\t"

    file_dix = 0

    with open(src_path, 'r') as src_f:
        reader = csv.reader(src_f, delimiter=delimiter)
        for _ in range(n_skip_line):
            next(reader)
        for idx, row in enumerate(reader):
            if idx % ROW_PER_FILE == 0:
                file_idx = int(idx / ROW_PER_FILE)
                dst_path = dst_filepath_generator(src_path, file_idx, dst_dir)

                logger.info("convert csv to tfrecord: %s -> %s (from %d line)" % \
                            (src_path, dst_path, idx))

                if idx != 0:
                    writer.close()
                writer = tf.python_io.TFRecordWriter(dst_path)

            try:
                if len(row) == 0: continue
                example = generator.create_example(row, names, dtypes)
                writer.write(example.SerializeToString())
            except Exception as e:
                logger.error("[err1] %s" % str(e))
                logger.error("[err2] %s" % str(row))
                logger.error("[err3] %s" % str(dst_path))
    writer.close()


def to_tfrecords(src_paths, *args):
    for src_path in src_paths:
        to_tfrecord(src_path, *args)


def dst_filepath_generator(src_filepath, idx, dst_dir):
    filename = os.path.basename(src_filepath)
    fileroot, fileext = os.path.splitext(filename)

    return os.path.join(dst_dir, "%s.%d.tfrecord" % (fileroot, idx))


class App(AppWrapper):
    def run(self, params):
        self._params = params

        filepath_chunk, data_config = self.prepare()
        
        n_procs = min(len(filepath_chunk), params.n_procs)

        self.convert(filepath_chunk, data_config, params.delimiter, params.n_skip_line,
                     params.dst_dir, n_procs)

    def prepare(self):
        params = self._params

        # read data config
        logger.info("Read data config: %s" % params.data_config)
        data_config = DataConfig(params.data_config)

        # make file_chunk for each process
        file_paths = glob.glob(params.csv_file)
        logger.info("Find %d csv file(s)"  % len(file_paths))

        if len(file_paths) <= params.n_procs:
            filepath_chunk = [file_paths]
        else:
            filepath_chunk = np.array_split(file_paths, params.n_procs)

        if not os.path.exists(params.dst_dir):
            os.makedirs(params.dst_dir)

        return filepath_chunk, data_config

    def convert(self, filepath_chunk, data_config, delimiter, n_skip_line, 
                dst_dir, n_procs):
        fn_args = [
            (filepath_chunk[i], n_skip_line, data_config.names, data_config.dtypes,
             dst_dir, delimiter)
            for i in range(n_procs)
        ]

        logging.info("Convert csv file(s) with %d procsess" % (n_procs))
        multiprocess.fn(to_tfrecords, fn_args, n_procs)