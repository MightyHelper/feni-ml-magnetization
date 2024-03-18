NFOLD=10
DATASET=$1
cp $DATASET ./dataset.csv
Rscript all_models.r glmnet $NFOLD dataset.csv
Rscript all_models.r svm $NFOLD dataset.csv
Rscript all_models.r ranger $NFOLD dataset.csv
Rscript all_models.r catboost $NFOLD dataset.csv
