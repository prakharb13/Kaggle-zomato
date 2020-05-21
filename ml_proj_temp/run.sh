export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv
export MODEL=$1

python src/cross_folds.py

FOLD=0 python src/train.py
FOLD=1 python src/train.py
FOLD=2 python src/train.py
FOLD=3 python src/train.py
FOLD=4 python src/train.py

python src/predict.py