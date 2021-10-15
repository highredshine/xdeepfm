#!/bin/bash

python -m venv minimal_env
cd minimal_env
zip -r ../spark_submit_env.zip .
cd ..
hdfs dfs -copyFromLocal spark_submit_env.zip juhochoi/envs/

source minimal_env/bin/activate
pip install -t ./requirements -r requirements.txt
cd requirements
zip -r ../requirements.zip .
cd ..