#!/bin/bash

PYTHON_ZIP="hdfs://matching/user/irteam/juhochoi/envs/spark_submit_env.zip#pythonlib"
PYSPARK_PYTHON="./pythonlib/bin/python3"
PROJECT_EGG=./dist/spark_submit-0.0.1-py3.8.egg
REQUIREMENTS=./requirements.zip
SPARK_CONFIG=./spark_submit_config.yml
CONNECTOR=../ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector_2.12-1.11.0.jar

SPARK_CMD="spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYSPARK_PYTHON} \
--archives ${PYTHON_ZIP} \
--py-files ${PROJECT_EGG},${REQUIREMENTS} \
--master yarn \
--deploy-mode cluster \
--jars ${CONNECTOR} \
--files ${SPARK_CONFIG}#config.yml \
spark_submit_program.py config.yml"

eval ${SPARK_CMD}