#!/bin/bash
NFOLD=10
DATASET=$1
CLIP=TRUE
SPLIT=TRUE
cp $DATASET ./dataset.csv
for model in glmnet svm ranger catboost; do
    /usr/bin/time -o $model.time bash -c "Rscript all_models.r $model $NFOLD dataset.csv $CLIP $SPLIT 2>&1 | tee $model.out"
done
