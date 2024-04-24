if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  dplyr,
  ggplot2,
  glmnet,
  kernlab,
  ranger,
  readr,
  tidyr,
  doMC,
  caret
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
library(catboost) # CatBoost
options(error = traceback, ragg.max_dim = c(500000, 500000))


args <- commandArgs(trailingOnly = TRUE)
# Default command line: Rscript all_movels.r catboost 10 ../new_dataset.csv TRUE FALSE
model_name <- if (length(args) > 0) args[1] else "catboost" # Allowed values: catboost, svm, glmnet, ranger
n_folds <- if (length(args) > 1) as.numeric(args[2]) else 2 # Number of folds for cross-validation
dataset_path <- if (length(args) > 2) args[3] else "../new_dataset.csv" # Path to dataset
clip_dataset <- if (length(args) > 3) as.logical(args[4]) else TRUE # Remove near-zero variance features
split_data <- if (length(args) > 4) as.logical(args[5]) else FALSE # Split data into train and test (Or use all data for training)

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
dataset <- dataset %>% select(-name, -tmg_std) # TODO: remove this
set.seed(123123) # for reproducibility
if (split_data) {
  splitIndex <- createDataPartition(dataset$tmg, p = 0.7, list = FALSE)
  train_data <- dataset[splitIndex,] %>% as.data.frame()
  test_data <- dataset[-splitIndex,] %>% as.data.frame()
} else {
  train_data <- dataset %>% as.data.frame()
  test_data <- dataset %>% as.data.frame()
}

ctrl <- trainControl(
  method = "cv",
  number = n_folds,
  returnResamp = 'final',
  savePredictions = 'final',
  classProbs = FALSE,
  verboseIter = F,
  allowParallel = T
)

train_catboost <- function() {
  # catboostgrid <- expand.grid(
  #   depth = c(2, 6, 8, 10),          # Maximum depth of trees. Deeper trees can model more complex relationships, but risk overfitting and require more data and time to train.
  #   learning_rate = c(0.01, 0.1), # Learning rate, or shrinkage factor. This parameter scales the contribution of each tree. Lower values can achieve better performance but require more trees.
  #   iterations = c(100, 200),     # Maximum number of trees to be built, or the number of boosting steps. More iterations lead to a more complex model, but also increase the risk of overfitting and the time to train the model.
  #   l2_leaf_reg = c(1e-6, 1, 3),        # L2 regularization term for the cost function. This parameter applies a penalty for complexity in the structure of the individual trees. Higher values make the model more conservative.
  #   rsm = c(0.8, 0.9, 1),              # Fraction of features to be used for each tree, a technique to reduce overfitting and speed up training.
  #   border_count = c(10, 20, 255)      # Number of splits considered for each feature. Higher values can lead to finer splits, but are more computationally expensive.
  # )
  model <- caret::train(
    x = train_data %>% select(-tmg),
    y = train_data$tmg,
    method = catboost.caret,
    trControl = ctrl,
    # tuneGrid = catboostgrid, # for catboost
    # tunelength = 2,
    # tuneLength = 100,
    logging_level = "Silent",
    preProcess = c("center", "scale") # Standardization
  )
  return(model)
}

train_svm <- function() {
  svm_grid <- expand.grid(
    C = 10^seq(0, 4, length = 20),
    sigma = 10^seq(-7, -1, length = 20)
  )
  model <- caret::train(
    # x = train_data %>% select(-tmg),
    # y = as.factor(train_data$tmg),
    tmg ~ .,
    data = train_data,
    method = "svmRadial",
    trControl = ctrl,
    tuneLength = 100,
    # tuneGrid = svm_grid,
    # preProcess = c("center", "scale"), # Standardization
    verbose = 100
  )
  return(model)
}

train_glment <- function() {
  # glmnet_grid <- expand.grid(
  #   alpha = 0.1 + seq(-0.5, 0.5, length = 20),
  #   lambda = 0.0001168715 + seq(-0.0005, 0.0005, length = 20)
  # )
  model <- caret::train(
    x = train_data %>% select(-tmg),
    y = train_data$tmg,
    method = "glmnet",
    trControl = ctrl,
    # tuneGrid = glmnet_grid,
    tuneLength = 100,
    preProcess = c("center", "scale"), # Standardization
    verbose = 100
  )
  return(model)
}

train_ranger <- function() {
  # ranger_grid <- expand.grid(
  #   mtry = c(2, 3, 4, 5, 6, 7, 8, 9, 10),
  #   min.node.size = c(1, 3, 5, 7, 9, 11, 13, 15),
  #   splitrule = c("variance", "extratrees")
  # )
  model <- caret::train(
    x = train_data %>% select(-tmg),
    y = train_data$tmg,
    method = "ranger",
    trControl = ctrl,
    # tuneGrid = ranger_grid,
    tuneLength = 100,
    preProcess = c("center", "scale"), # Standardization
    importance = "impurity", # permutation
    verbose = 100,
    num.thread = 16
  )
  return(model)
}


save_hyperparameters_plot <- function() {
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
    labs(title = paste0(model_name, ": RMSE by hyperparameters"), x = "Hyperparameters", y = "RMSE") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  ggsave(paste0(model_name, "_hyperparameters.png"), plot, width = 1000 + 20 * num_rows, height = 2000, units = "px", limitsize = FALSE)
}

save_hyperparameter_heatmap <- function() {
  print(paste0("Plotting hyperparameters grid for ", model_name, " because it has ", length(hyper), " hyperparameters", "(", hyper[[1]], ", ", hyper[[2]], ")"))
  x <- model$results[[hyper[[1]]]]
  y <- model$results[[hyper[[2]]]]
  min_rmse <- min(model$results$RMSE)
  rmse_50 <- quantile(model$results$RMSE, 0.5)
  # Export csv
  write.csv(model$results, paste0(model_name, "_hyperparameters_grid.csv"), row.names = FALSE)
  # Heatmap of RMSE by hyperparameters
  plot <- ggplot(model$results, aes(x = as.factor(x), y = as.factor(y), fill = RMSE)) +
    geom_tile() +
    scale_fill_gradientn(colours = c("black", "blue", "green", "yellow", "red", "white"), limits = c(min_rmse, rmse_50)) +
    # Add pink circle point at minimum
    geom_point(aes(x = as.factor(x[which.min(model$results$RMSE)]), y = as.factor(y[which.min(model$results$RMSE)])), color = "pink", size = 2) +
    # Add red circle point at maximum
    geom_point(aes(x = as.factor(x[which.max(model$results$RMSE)]), y = as.factor(y[which.max(model$results$RMSE)])), color = "magenta", size = 2) +
    labs(title = paste0(model_name, ": RMSE by hyperparameters"), x = hyper[[1]], y = hyper[[2]]) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    scale_x_discrete(labels = function(x) sprintf("%.3e", as.numeric(x))) +
    scale_y_discrete(labels = function(y) sprintf("%.3e", as.numeric(y)))

  unique_count_hyper1 <- length(unique(x))
  unique_count_hyper2 <- length(unique(y))
  ggsave(paste0(model_name, "_hyperparameters_grid.png"), plot, width = 1000 + 30 * unique_count_hyper1, height = 1000 + 30 * unique_count_hyper2, units = "px", limitsize = FALSE)
}

save_importance <- function() {
  importance_results <- as.data.frame(varImp(model, scale = FALSE)$importance)
  print(importance_results)
  plot <- importance_results %>%
    mutate(Variables = row.names(.)) %>%
    arrange(desc(Overall)) %>%
    head(20) %>%
    ggplot(aes(x = reorder(Variables, Overall), y = Overall)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = paste0(model_name, ": Variable Importance"), x = "Variables", y = "Importance") +
    theme_bw()
  ggsave(paste0(model_name, "_importance.png"), plot, width = 2000, height = 2000, units = "px", limitsize = FALSE)
  write.csv(importance_results, paste0(model_name, "_importance.csv"))
}

save_execution_result <- function() {
  predictions <- predict(model, test_data)
  RMSE_full <- caret::RMSE(predictions, test_data$tmg)
  RMSE_fold <- mean(model$resample$RMSE)
  print("Hyperparameters:")
  hyperparameters_rmse <- data.frame(
    bestTune = model$bestTune,
    RMSE_fold = RMSE_fold,
    RMSE_full = RMSE_full
  )
  print(hyperparameters_rmse)
  write.csv(hyperparameters_rmse, paste0(model_name, "_hyperparameters_rmse.csv"), row.names = FALSE)
  results <- data.frame(Reference = test_data$tmg, Predicted = as.vector(predictions))
  plot <- ggplot(results, aes(x = Reference, y = Predicted)) +
    geom_point(color = 'blue') +
    geom_abline(intercept = 0, slope = 1, color = 'red') +
    ggtitle(paste0("Model: ", model_name, " - RMSE-fold: ", format(RMSE_fold, scientific=F, nsmall=4), " - RMSE-full: ", format(RMSE_full, scientific=F, nsmall=4))) +
    xlab("Reference Values") +
    ylab("Predicted Values") +
    theme_bw()
  ggsave(paste0(model_name, "_results.png"), plot, width = 2000, height = 2000, units = "px", limitsize = FALSE)
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

num_rows <- model$results %>% nrow()
print(paste0("Saving results... (", num_rows, " rows)"))

save_hyperparameters_plot()

print("Considering heatmap... ")
# If only 2 hyperparameters, plot the results in a grid, as a heatmap
if (length(hyper) == 2) {
  save_hyperparameter_heatmap()
} else {
  print(paste0("Not plotting hyperparameters grid for ", model_name, " because it has ", length(hyper), " hyperparameters"))
}


save_importance()

save_execution_result()
