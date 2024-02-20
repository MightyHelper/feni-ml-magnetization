#!/usr/bin/env bash
DATASET=./datasets/06_full_rerun2.csv
N_FOLDS=10
Rscript all_models.r glmnet $N_FOLDS $DATASET
Rscript all_models.r svm $N_FOLDS $DATASET
Rscript all_models.r ranger $N_FOLDS $DATASET
Rscript all_models.r catboost $N_FOLDS $DATASET
