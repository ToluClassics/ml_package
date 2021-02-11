export TRAINING_DATA=input/train_folds.csv
export FOLD=0
export MODEL=$1
export TEST_DATA=input/test_cat.csv

python src/train.py

