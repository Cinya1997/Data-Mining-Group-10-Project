# load the required libraries
#install.packages("ranger")
#install.packages("caret")
#install.packages("ROSE")  # used for oversampling
#install.packages("ggplot2")
#install.packages("lattice")
library(ranger)
library(caret)
library(ROSE)

train_data <- read.csv("sampled_data.csv")
train_data$TARGET <- as.factor(train_data$TARGET)

# 1.data preprocessing,one-hot encoding
# check the structure of the data
str(train_data)

# transform categorical variables to factors
train_data$NAME_CONTRACT_TYPE  <- as.factor(train_data$NAME_CONTRACT_TYPE)
train_data$CODE_GENDER         <- as.factor(train_data$CODE_GENDER)
train_data$NAME_INCOME_TYPE    <- as.factor(train_data$NAME_INCOME_TYPE)
train_data$NAME_EDUCATION_TYPE <- as.factor(train_data$NAME_EDUCATION_TYPE)
train_data$NAME_FAMILY_STATUS  <- as.factor(train_data$NAME_FAMILY_STATUS)

# one-hot encoding using dummyVars
dummy_model <- dummyVars(~ ., data = train_data[, !names(train_data) %in% "TARGET"])
train_data_encoded <- predict(dummy_model, newdata = train_data)

# transform the encoded data to a data frame, including the target variable
train_data_numeric <- data.frame(train_data_encoded, TARGET = train_data$TARGET)

# check the structure of the numeric data
str(train_data_numeric)


# 2.split data into train 70%, validation 15%, and test sets 15%
# set seed for reproducibility
set.seed(123)

# split the data into train 70%, validation 15%, and test sets 15%
trainIndex <- createDataPartition(train_data_numeric$TARGET, p = 0.7, list = FALSE)
train_data <- train_data_numeric[trainIndex, ]
temp_data <- train_data_numeric[-trainIndex, ]

valIndex <- createDataPartition(temp_data$TARGET, p = 0.5, list = FALSE)
validation_data <- temp_data[valIndex, ]
test_data <- temp_data[-valIndex, ]

# check the dimensions of the datasets
cat("Number of samples in the training set: ", nrow(train_data), "\n")
cat("Number of samples in the validation set: ", nrow(validation_data), "\n")
cat("Number of samples in the test set: ", nrow(test_data), "\n")


# 3.non-oversampled model,ranger
# train a ranger model on the non-oversampled data using the ranger function
model_ranger <- ranger(TARGET ~ ., 
                       data = train_data, 
                       num.trees = 200, 
                       mtry = 5,            
                       min.node.size = 10,   
                       probability = TRUE)

# predict on the validation set
predictions_val <- predict(model_ranger, data = validation_data)$predictions[, 2]
predicted_class_val <- ifelse(predictions_val > 0.3, "1", "0")

# ensure the predicted factor levels are the same as the reference factor
levels_order <- levels(validation_data$TARGET)  # get the levels order of the reference factor
predicted_class_val <- factor(predicted_class_val, levels = levels_order)  # set the levels of the predicted factor

# compute the confusion matrix and performance metrics
conf_matrix_val <- confusionMatrix(as.factor(predicted_class_val), validation_data$TARGET)

# output Sensitivity, Accuracy, Specificity, TP, FP, TN, FN
conf_matrix_val$table  # TP, FP, TN, FN
conf_matrix_val$byClass["Sensitivity"]  
conf_matrix_val$byClass["Specificity"]  
conf_matrix_val$overall["Accuracy"]  


# 4.oversampled model,ranger
# oversample the training data using ROSE
train_data_oversample <- ROSE(TARGET ~ ., data = train_data, seed = 123)$data

# train a ranger model on the oversampled data
model_ranger_oversample <- ranger(TARGET ~ ., data = train_data_oversample, num.trees = 500,mtry = 3, min.node.size = 10, probability = TRUE)

# predict on the validation set
predictions_val_oversample <- predict(model_ranger_oversample, data = validation_data)$predictions[, 2]
predicted_class_val_oversample <- ifelse(predictions_val_oversample > 0.3, "1", "0")

# ensure the predicted factor levels are the same as the reference factor
levels_order <- levels(validation_data$TARGET)  # get the levels order of the reference factor
predicted_class_val_oversample <- factor(predicted_class_val_oversample, levels = levels_order)  # set the levels of the predicted factor

# compute the confusion matrix and performance metrics
conf_matrix_val_oversample <- confusionMatrix(as.factor(predicted_class_val_oversample), validation_data$TARGET)

# output Sensitivity, Accuracy, Specificity, TP, FP, TN, FN
conf_matrix_val_oversample$table  # TP, FP, TN, FN
conf_matrix_val_oversample$byClass["Sensitivity"]  
conf_matrix_val_oversample$byClass["Specificity"]  
conf_matrix_val_oversample$overall["Accuracy"]  

# 5.predict on the test set for final evaluation
predictions_test <- predict(model_ranger, data = test_data)$predictions[, 2]
predicted_class_test <- ifelse(predictions_test > 0.5, "1", "0")

# ensure the predicted factor levels are the same as the reference factor
levels_order <- levels(test_data$TARGET) 
predicted_class_test <- factor(predicted_class_test, levels = levels_order) 

# compute the confusion matrix and performance metrics
conf_matrix_test <- confusionMatrix(as.factor(predicted_class_test), test_data$TARGET)

# output Sensitivity, Accuracy, Specificity, TP, FP, TN, FN
conf_matrix_test$table  # TP, FP, TN, FN
conf_matrix_test$byClass["Sensitivity"]  
conf_matrix_test$byClass["Specificity"]  
conf_matrix_test$overall["Accuracy"]  

thresholds <- seq(0.1, 0.9, by = 0.05)
results <- data.frame(Threshold = numeric(), Sensitivity = numeric(), Specificity = numeric(), Accuracy = numeric())

for (threshold in thresholds) {
  predicted_class_val_oversample <- ifelse(predictions_val_oversample > threshold, "1", "0")
  predicted_class_val_oversample <- factor(predicted_class_val_oversample, levels = levels_order)
  
  conf_matrix_val_oversample <- confusionMatrix(as.factor(predicted_class_val_oversample), validation_data$TARGET)
  
  results <- rbind(results, data.frame(
    Threshold = threshold,
    Sensitivity = conf_matrix_val_oversample$byClass["Sensitivity"],
    Specificity = conf_matrix_val_oversample$byClass["Specificity"],
    Accuracy = conf_matrix_val_oversample$overall["Accuracy"]
  ))
}

print(results)

#ROC
library(pROC)
predictions_val <- predict(model_ranger, data = validation_data)$predictions[, 2]
roc_random_nosample <- roc(validation_data$TARGET, predictions_val)

predictions_test_oversample <- predict(model_ranger_oversample, data = test_data)$predictions[, 2]
roc_random <- roc(test_data$TARGET, predictions_test_oversample)
