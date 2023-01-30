#!/bin/sh
preprocessing_data="./data/preprocessing_Chiyoda_20230130.csv"
result_fpath="./result/result.csv"
python3 ./script/MachineLearning.py $preprocessing_data $result_fpath