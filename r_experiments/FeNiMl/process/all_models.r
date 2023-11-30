if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  caret,
  dplyr,
  ggplot2,
  glmnet,
  kernlab,
  ranger,
  readr,
  tidyr,
  doMC
)
if (!require("catboost")) {
  print("Installing catboost...")
  install.packages("devtools")
  pacman::p_load(
    pkgdown,
    roxygen2,
    rversions,
    urlchecker,
    devtools
  )
  devtools::install_url('https://github.com/catboost/catboost/releases/download/v1.2.2/catboost-R-Linux-1.2.2.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
}
#
# library(ranger)   # Random Forest
# library(glmnet)   # elastic net
# library(kernlab)  # SVM
library(catboost) # CatBoost
#
# library(dplyr)
# library(ggplot2)
# library(readr)
# library(caret)


args <- commandArgs(trailingOnly = TRUE)
# Default command line: R -f catboost.r --args ../new_dataset.csv TRUE catboost TRUE 10
dataset_path <- if (length(args) > 0) args[1] else "../new_dataset.csv"
clip_dataset <- if (length(args) > 1) as.logical(args[2]) else TRUE
model_name <- if (length(args) > 2) args[3] else "catboost"
split_data <- if (length(args) > 3) as.logical(args[4]) else TRUE
n_folds <- if (length(args) > 4) as.numeric(args[5]) else 2

arguments <- data.frame(
  dataset_path = dataset_path,
  clip_dataset = clip_dataset,
  model_name = model_name,
  split_data = split_data,
  n_folds = n_folds
)

print("Arguments:")
print(arguments)

dataset <- read_csv(dataset_path, show_col_types = FALSE)
if (clip_dataset) {
  nzv <- nearZeroVar(dataset, saveMetrics = FALSE)
  dataset <- dataset[, -nzv]
}
dataset <- dataset %>% select(-name) # TODO: remove this


if (split_data) {
  set.seed(123) # for reproducibility
  splitIndex <- createDataPartition(dataset$tmg, p = 0.7, list = FALSE)
  train_data <- dataset[splitIndex,]
  test_data <- dataset[-splitIndex,]
  train_data <- train_data %>% as.data.frame()
} else {
  train_data <- dataset
  test_data <- dataset
}

ctrl <- trainControl(
  method = "cv",
  number = n_folds,
  returnResamp = 'final',
  savePredictions = 'final',
  classProbs = TRUE
  # verboseIter = F,
  # allowParallel = F
)

train_catboost <- function() {
  catboostgrid <- expand.grid(
    depth = c(6, 8, 10), # Maximum depth of trees. Deeper trees can model more complex relationships, but risk overfitting and require more data and time to train.
    learning_rate = c(0.01, 0.1), # Learning rate, or shrinkage factor. This parameter scales the contribution of each tree. Lower values can achieve better performance but require more trees.
    iterations = c(100, 200), # Maximum number of trees to be built, or the number of boosting steps. More iterations lead to a more complex model, but also increase the risk of overfitting and the time to train the model.
    l2_leaf_reg = c(1, 3), # L2 regularization term for the cost function. This parameter applies a penalty for complexity in the structure of the individual trees. Higher values make the model more conservative.
    rsm = c(0.8, 1), # Fraction of features to be used for each tree, a technique to reduce overfitting and speed up training.
    border_count = c(32, 64) # Number of splits considered for each feature. Higher values can lead to finer splits, but are more computationally expensive.
  )
  model <- train(
    x = train_data %>% select(-tmg),
    y = train_data$tmg,
    method = catboost.caret,
    trControl = ctrl,
    tuneGrid = catboostgrid, # for catboost
    #tunelength = 2,
    logging_level = "Silent",
    preProcess = c("center", "scale"), # Standardization
  )
  return(model)
}

train_svm <- function() {
  svm_grid <- expand.grid(
    C = 10^seq(0, 4, length = 20),
    sigma = 10^seq(-7, -1, length = 20)
  )
  model <- train(
    # x = train_data %>% select(-tmg),
    # y = train_data$tmg,
    tmg ~ .,
    data = train_data,
    method = "svmRadial",
    trControl = ctrl,
    tuneGrid = svm_grid,
    # preProcess = c("center", "scale"), # Standardization
    verbose = 100
  )
  return(model)
}

train_glment <- function() {
  glmnet_grid <- expand.grid(
    alpha = 10^seq(-2, -1, length = 50),
    lambda = 10^seq(-6, -3.5, length = 50)
  )
  model <- train(
    x = train_data %>% select(-tmg),
    y = train_data$tmg,
    method = "glmnet",
    trControl = ctrl,
    tuneGrid = glmnet_grid,
    preProcess = c("center", "scale"), # Standardization
    verbose = 100
  )
  return(model)
}

train_ranger <- function() {
  ranger_grid <- expand.grid(
    mtry = c(2, 3, 4, 5, 6, 7, 8, 9, 10),
    min.node.size = c(1, 3, 5, 7, 9, 11, 13, 15),
    splitrule = c("variance", "extratrees")
  )
  model <- train(
    x = train_data %>% select(-tmg),
    y = train_data$tmg,
    method = "ranger",
    trControl = ctrl,
    tuneGrid = ranger_grid,
    preProcess = c("center", "scale"), # Standardization
    verbose = 100,
    num.thread = 16
  )
  return(model)
}


hyperparameters_glmnet <- c("alpha", "lambda")
hyperparameters_ranger <- c("mtry", "min.node.size", "splitrule")
hyperparameters_svm <- c("C", "sigma")
hyperparameters_catboost <- c("depth", "learning_rate", "iterations", "l2_leaf_reg", "rsm", "border_count")

print("Training model...")

if (model_name == "catboost") {
  model <- train_catboost()
  hyper <- hyperparameters_catboost
}
if (model_name == "svm") {
  model <- train_svm()
  hyper <- hyperparameters_svm
}
if (model_name == "glmnet") {
  model <- train_glment()
  hyper <- hyperparameters_glmnet
}
if (model_name == "ranger") {
  model <- train_ranger()
  hyper <- hyperparameters_ranger
}

print("Model:")

print(model)


num_rows <- model$results %>%
  tidyr::unite(col = a_l, hyper, sep = "_") %>% # unite the hyperparameters
  nrow()
print(paste0("Saving results... (", num_rows, " rows)"))

plot <- model$results %>%
  tidyr::unite(col = a_l, hyper, sep = "_") %>%
  ggplot(aes(x = a_l, y = RMSE)) +
  geom_point(color = 'red') +
  geom_errorbar(
    aes(ymin = RMSE - RMSESD, ymax = RMSE + RMSESD),
    width = .02,
    color = 'orange'
  ) +
  theme_classic() +
  labs(title = "Model: Mean and Standard deviation after hyper-parameter tuning") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
# png(filename = paste0(model_name, "_hyperparameters.png"), width = 20 * num_rows, height = 1000)
# dev.off()
ggsave(paste0(model_name, "_hyperparameters.png"), plot, width = 20 * num_rows, height = 2000, units = "px", limitsize = FALSE)

print("Considering heatmap... ")
# If only 2 hyperparameters, plot the results in a grid, as a heatmap
if (length(hyper) == 2) {
  print(paste0("Plotting hyperparameters grid for ", model_name, " because it has ", length(hyper), " hyperparameters", "(", hyper[[1]], ", ", hyper[[2]], ")"))
  print(model$results[[hyper[[1]]]])
  print(model$results[[hyper[[2]]]])
  print(model$results$RMSE)
  # Heatmap of RMSE by hyperparameters
  # custom_color_scale <- scales::gradient_n_pal(colours=c("blue", "green", "yellow", "red"))

  plot <- ggplot(model$results, aes(x = as.factor(model$results[[hyper[[1]]]]), y = as.factor(model$results[[hyper[[2]]]]), fill = RMSE)) +
    geom_tile() +
    scale_fill_gradientn(colours = c("blue", "green", "yellow", "red")) +
    labs(title = paste0(model_name, ": RMSE by hyperparameters"), x = hyper[[1]], y = hyper[[2]]) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  # png(filename = paste0(model_name, "_hyperparameters_grid.png"))
  unique_count_hyper1 <- length(unique(model$results[[hyper[[1]]]]))
  unique_count_hyper2 <- length(unique(model$results[[hyper[[2]]]]))
  ggsave(paste0(model_name, "_hyperparameters_grid.png"), plot, width = 1000 + 30 * unique_count_hyper1, height = 1000 + 30 * unique_count_hyper2, units = "px", limitsize = FALSE)
} else {
  print(paste0("Not plotting hyperparameters grid for ", model_name, " because it has ", length(hyper), " hyperparameters"))
}

png(filename = paste0(model_name, "_importance.png"))
varImp(model)
importance_results <- as.data.frame(varImp(model, scale = FALSE)$importance)
plot(varImp(model), top = 20)
dev.off()
# Export importance to csv
write.csv(importance_results, paste0(model_name, "_importance.csv"))

predictions <- predict(model, test_data)
# # Compute the RMSE (Root Mean Square Error)
RMSE <- sqrt(mean((predictions - test_data$tmg)^2))

print("Hyperparameters:")
hyperparameters_rmse <- data.frame(
  bestTune = model$bestTune,
  RMSE = RMSE
)
print(hyperparameters_rmse)
write.csv(hyperparameters_rmse, paste0(model_name, "_hyperparameters_rmse.csv"), row.names = FALSE)

#
png(filename = paste0(model_name, "_results.png"))
results <- data.frame(Reference = test_data$tmg, Predicted = as.vector(predictions))
ggplot(results, aes(x = Reference, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(intercept = 0, slope = 1, color = 'red') +
  ggtitle(paste0("Model: ", model_name, " - RMSE: ", RMSE)) +
  xlab("Reference Values") +
  ylab("Predicted Values") +
  theme_bw()

dev.off()
