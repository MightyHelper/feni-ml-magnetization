NFOLD=10
DATASET=$1
CLIP=TRUE
SPLIT=FALSE
cp $DATASET ./dataset.csv
Rscript all_models.r glmnet $NFOLD dataset.csv $CLIP $SPLIT
Rscript all_models.r svm $NFOLD dataset.csv $CLIP $SPLIT
Rscript all_models.r ranger $NFOLD dataset.csv $CLIP $SPLIT
Rscript all_models.r catboost $NFOLD dataset.csv $CLIP $SPLIT
