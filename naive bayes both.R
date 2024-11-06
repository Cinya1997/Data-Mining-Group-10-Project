
  
library(caret)
library(dplyr)
library(ggplot2)
library(lattice)
library(e1071)
library(DMwR2)

data <- read.csv("sampled_data.csv")

# View data format
# according to data format convern the data format
data$TARGET <- as.factor(data$TARGET)

data$NAME_CONTRACT_TYPE <- as.factor(data$NAME_CONTRACT_TYPE)
data$CODE_GENDER <- as.factor(data$CODE_GENDER)
data$NAME_INCOME_TYPE <- as.factor(data$NAME_INCOME_TYPE)
data$NAME_EDUCATION_TYPE <- as.factor(data$NAME_EDUCATION_TYPE)
data$NAME_FAMILY_STATUS <- as.factor(data$NAME_FAMILY_STATUS)
data$REGION_RATING_CLIENT <- as.factor(data$REGION_RATING_CLIENT)
# Because we plan to use multiple models for comparison, we choose the three-part method. At the same time, due to the imbalance of the target of the data, the data set is divided into 70%, 15%, and 15%.

set.seed(123)

# train data
train_index <- createDataPartition(data$TARGET, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
temp_data <- data[-train_index, ]

#valid data
valid_index <- createDataPartition(temp_data$TARGET, p = 0.5, list = FALSE)
valid_data <- temp_data[valid_index, ]

#test data
test_data <- temp_data[-valid_index, ]

# Normalize numeric variables
num_vars <- c("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH")
train_means <- apply(train_data[num_vars], 2, mean)
train_sds <- apply(train_data[num_vars], 2, sd)


train_data[num_vars] <- scale(train_data[num_vars], center = train_means, scale = train_sds)
valid_data[num_vars] <- scale(valid_data[num_vars], center = train_means, scale = train_sds)
test_data[num_vars] <- scale(test_data[num_vars], center = train_means, scale = train_sds)

# In order to take categorical variables into account, we need to perform one-hot encoding on the categorical variables and exclude the TARGET column.
dummies <- dummyVars(~ . - TARGET, data = train_data)
train_data_encoded <- predict(dummies, newdata = train_data) %>% as.data.frame()

# add TARGET
train_data_encoded$TARGET <- train_data$TARGET

# Perform the same process on the validation set and the test set.
valid_data_encoded <- predict(dummies, newdata = valid_data) %>% as.data.frame()
valid_data_encoded$TARGET <- valid_data$TARGET

test_data_encoded <- predict(dummies, newdata = test_data) %>% as.data.frame()
test_data_encoded$TARGET <- test_data$TARGET

# Make sure TARGET is a factor in the training, validation, and test sets
train_data_encoded$TARGET <- as.factor(train_data_encoded$TARGET)
valid_data_encoded$TARGET <- as.factor(valid_data_encoded$TARGET)
test_data_encoded$TARGET <- as.factor(test_data_encoded$TARGET)
#Load necessary libraries
library(e1071) # For Naive Bayes library(caret) # For evaluation metrics

#Assuming your data preprocessing has been completed and train_data_encoded is ready
# 1. Function to Randomly Oversample Minority Class
oversample_data <- function(data, target_column) {
  # Separate majority and minority classes
  majority_class <- data[data[[target_column]] == levels(data[[target_column]])[which.max(table(data[[target_column]]))], ]
  minority_class <- data[data[[target_column]] == levels(data[[target_column]])[which.min(table(data[[target_column]]))], ]
  
  # Calculate the difference in class counts
  n_majority <- nrow(majority_class)
  n_minority <- nrow(minority_class)
  
  # Randomly sample with replacement from the minority class to balance it
  oversampled_minority <- minority_class[sample(1:n_minority, n_majority, replace = TRUE), ]
  
  # Combine the majority class with the oversampled minority class
  oversampled_data <- rbind(majority_class, oversampled_minority)
  
  return(oversampled_data)
}

# Apply random oversampling on the training data
train_data_oversampled <- oversample_data(train_data_encoded, "TARGET")

# Verify class distribution after oversampling
cat("\nClass Distribution After Random Oversampling:\n")
## 
## Class Distribution After Random Oversampling:
print(table(train_data_oversampled$TARGET))
## 
##     0     1 
## 25730 25730
# Prepare the training data for Naive Bayes
x_train_oversampled <- train_data_oversampled[, -ncol(train_data_oversampled)]  # Exclude TARGET column
y_train_oversampled <- as.factor(train_data_oversampled$TARGET)

# Train Naive Bayes on the oversampled dataset
set.seed(123)
nb_model_oversampled <- naiveBayes(x = x_train_oversampled, y = y_train_oversampled)

# Predict on validation set
x_valid <- valid_data_encoded[, -ncol(valid_data_encoded)]  # Exclude TARGET column
y_valid <- as.factor(valid_data_encoded$TARGET)

valid_predictions_oversampled <- predict(nb_model_oversampled, newdata = x_valid)

# Confusion Matrix and Accuracy for the validation set
cat("\nRandom Oversampled Naive Bayes Results:\n")
## 
## Random Oversampled Naive Bayes Results:
print(confusionMatrix(as.factor(valid_predictions_oversampled), y_valid))
## Confusion Matrix and Statistics

#NON OVER- SAMPLED

library(caret)
library(dplyr)
library(ggplot2)
library(lattice)
library(e1071)
library(DMwR2)

# load data
data <- read.csv("sampled_data.csv")

# View data format
# according to data format convern the data format
data$TARGET <- as.factor(data$TARGET)

data$NAME_CONTRACT_TYPE <- as.factor(data$NAME_CONTRACT_TYPE)
data$CODE_GENDER <- as.factor(data$CODE_GENDER)
data$NAME_INCOME_TYPE <- as.factor(data$NAME_INCOME_TYPE)
data$NAME_EDUCATION_TYPE <- as.factor(data$NAME_EDUCATION_TYPE)
data$NAME_FAMILY_STATUS <- as.factor(data$NAME_FAMILY_STATUS)
data$REGION_RATING_CLIENT <- as.factor(data$REGION_RATING_CLIENT)

# Because we plan to use multiple models for comparison, we choose the three-part method. At the same time, due to the imbalance of the target of the data, the data set is divided into 70%, 15%, and 15%.

set.seed(123)

# train data
train_index <- createDataPartition(data$TARGET, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
temp_data <- data[-train_index, ]

#valid data
valid_index <- createDataPartition(temp_data$TARGET, p = 0.5, list = FALSE)
valid_data <- temp_data[valid_index, ]

#test data
test_data <- temp_data[-valid_index, ]

# Normalize numeric variables
num_vars <- c("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH")
train_means <- apply(train_data[num_vars], 2, mean)
train_sds <- apply(train_data[num_vars], 2, sd)


train_data[num_vars] <- scale(train_data[num_vars], center = train_means, scale = train_sds)
valid_data[num_vars] <- scale(valid_data[num_vars], center = train_means, scale = train_sds)
test_data[num_vars] <- scale(test_data[num_vars], center = train_means, scale = train_sds)

# In order to take categorical variables into account, we need to perform one-hot encoding on the categorical variables and exclude the TARGET column.
dummies <- dummyVars(~ . - TARGET, data = train_data)
train_data_encoded <- predict(dummies, newdata = train_data) %>% as.data.frame()

# add TARGET
train_data_encoded$TARGET <- train_data$TARGET

# Perform the same process on the validation set and the test set.
valid_data_encoded <- predict(dummies, newdata = valid_data) %>% as.data.frame()
valid_data_encoded$TARGET <- valid_data$TARGET

test_data_encoded <- predict(dummies, newdata = test_data) %>% as.data.frame()
test_data_encoded$TARGET <- test_data$TARGET

# Make sure TARGET is a factor in the training, validation, and test sets
train_data_encoded$TARGET <- as.factor(train_data_encoded$TARGET)
valid_data_encoded$TARGET <- as.factor(valid_data_encoded$TARGET)
test_data_encoded$TARGET <- as.factor(test_data_encoded$TARGET)

# Load necessary libraries
library(e1071)  # For Naive Bayes
library(caret)  # For evaluation metrics

# Assuming your data preprocessing has been completed and train_data_encoded is ready

# 1. Function to Randomly Oversample Minority Class
oversample_data <- function(data, target_column) {
  # Separate majority and minority classes
  majority_class <- data[data[[target_column]] == levels(data[[target_column]])[which.max(table(data[[target_column]]))], ]
  minority_class <- data[data[[target_column]] == levels(data[[target_column]])[which.min(table(data[[target_column]]))], ]
  
  # Calculate the difference in class counts
  n_majority <- nrow(majority_class)
  n_minority <- nrow(minority_class)
  
  # Randomly sample with replacement from the minority class to balance it
  oversampled_minority <- minority_class[sample(1:n_minority, n_majority, replace = TRUE), ]
  
  # Combine the majority class with the oversampled minority class
  oversampled_data <- rbind(majority_class, oversampled_minority)
  
  return(oversampled_data)
}

# Apply random oversampling on the training data
train_data_oversampled <- oversample_data(train_data_encoded, "TARGET")

# Verify class distribution after oversampling
cat("\nClass Distribution After Random Oversampling:\n")
## 
## Class Distribution After Random Oversampling:
print(table(train_data_oversampled$TARGET))
## 
##     0     1 
## 25730 25730
# Prepare the training data for Naive Bayes
x_train_oversampled <- train_data_oversampled[, -ncol(train_data_oversampled)]  # Exclude TARGET column
y_train_oversampled <- as.factor(train_data_oversampled$TARGET)

# Train Naive Bayes on the oversampled dataset
set.seed(123)
nb_model_oversampled <- naiveBayes(x = x_train_oversampled, y = y_train_oversampled)

# Predict on validation set
x_valid <- valid_data_encoded[, -ncol(valid_data_encoded)]  # Exclude TARGET column
y_valid <- as.factor(valid_data_encoded$TARGET)

valid_predictions_oversampled <- predict(nb_model_oversampled, newdata = x_valid)

# Confusion Matrix and Accuracy for the validation set
cat("\nRandom Oversampled Naive Bayes Results:\n")
## 
## Random Oversampled Naive Bayes Results:
print(confusionMatrix(as.factor(valid_predictions_oversampled), y_valid))

# none-oversample
# Prepare the training data for Naive Bayes (non-oversampled)
x_train2 <- train_data_encoded[, -ncol(train_data_encoded)]  # Exclude TARGET column
y_train2 <- as.factor(train_data_encoded$TARGET)

# Train Naive Bayes on the original dataset (non-oversampled)
set.seed(123)
nb_model2 <- naiveBayes(x = x_train2, y = y_train2)

# Predict on validation set
x_valid2 <- valid_data_encoded[, -ncol(valid_data_encoded)]  # Exclude TARGET column
y_valid2 <- as.factor(valid_data_encoded$TARGET)

valid_predictions2 <- predict(nb_model2, newdata = x_valid2)

# Confusion Matrix and Accuracy for the validation set (non-oversampled)
cat("\nNon-Oversampled Naive Bayes Results:\n")
print(confusionMatrix(as.factor(valid_predictions2), y_valid))


#ROC
library(pROC)
valid_predictions_oversampled <- predict(nb_model_oversampled, newdata = x_valid, type = "raw")
roc_naive <- roc(y_valid, valid_predictions_oversampled[, 2])

valid_predictions_raw <- predict(nb_model2, newdata = x_valid2, type = "raw")
roc_naive_nosample <- roc(y_valid2, valid_predictions_raw[, 2])
