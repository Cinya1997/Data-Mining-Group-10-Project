library(xgboost)
library(caret)
library(dplyr)
library(ggplot2)
library(lattice)
library(e1071)
library(DMwR2)
library(gbm)

# load data
data <- read.csv("sampled_data.csv")

# View data format
# according to data format convern the data format

data$NAME_CONTRACT_TYPE <- as.factor(data$NAME_CONTRACT_TYPE)
data$CODE_GENDER <- as.factor(data$CODE_GENDER)
data$NAME_INCOME_TYPE <- as.factor(data$NAME_INCOME_TYPE)
data$NAME_EDUCATION_TYPE <- as.factor(data$NAME_EDUCATION_TYPE)
data$NAME_FAMILY_STATUS <- as.factor(data$NAME_FAMILY_STATUS)
data$REGION_RATING_CLIENT <- as.factor(data$REGION_RATING_CLIENT)

# train data

set.seed(123)
train_index <- createDataPartition(data$TARGET, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
temp_data <- data[-train_index, ]

cat("Unique values in train_data$TARGET after partitioning:", unique(train_data$TARGET), "\n")


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

cat("Unique values in train_data$TARGET after partitioning:", unique(train_data$TARGET), "\n")

#  Make sure TARGET is a character before converting to a factor
train_data_encoded$TARGET <- as.numeric(train_data_encoded$TARGET)
valid_data_encoded$TARGET <- as.numeric(valid_data_encoded$TARGET)
test_data_encoded$TARGET <- as.numeric(test_data_encoded$TARGET)

cat("Unique values in train_data$TARGET after partitioning:", unique(train_data$TARGET), "\n")

# Make sure TARGET is a factor in the training, validation, and test sets
train_data_encoded$TARGET <- as.numeric(as.character(train_data_encoded$TARGET))
valid_data_encoded$TARGET <- as.numeric(as.character(valid_data_encoded$TARGET))
test_data_encoded$TARGET <- as.numeric(as.character(test_data_encoded$TARGET))

cat("Unique values in train_data$TARGET after partitioning:", unique(train_data_encoded$TARGET), "\n")
cat("Unique values in TARGET column of train_data_encoded after final correction:", unique(train_data_encoded$TARGET), "\n")

# Prepare x_train and y_train
x_train <- as.matrix(train_data_encoded[, -ncol(train_data_encoded)])  # Exclude TARGET column
y_train <- train_data_encoded$TARGET  # This should now contain only 0 and 1

valid_data_encoded$TARGET <- as.numeric(valid_data_encoded$TARGET)
x_valid <- as.matrix(valid_data_encoded[, -ncol(valid_data_encoded)])
y_valid <- valid_data_encoded$TARGET

test_data_encoded$TARGET <- as.numeric(test_data_encoded$TARGET)
x_test <- as.matrix(test_data_encoded[, -ncol(test_data_encoded)])
y_test <- test_data_encoded$TARGET



# build the model
gbdt_model_non_oversampled <- gbm(
  formula = TARGET ~ .,
  distribution = "bernoulli", 
  data = train_data_encoded,
  n.trees = 100,               
  interaction.depth = 3,       
  shrinkage = 0.1,             
  n.minobsinnode = 10,         
  verbose = FALSE              
)

# preodict the model
x_valid <- as.data.frame(x_valid)
x_test <- as.data.frame(x_test)

valid_predictions_prob <- predict(gbdt_model_non_oversampled, newdata = x_valid, n.trees = 100, type = "response")
valid_predictions_non_oversampled <- ifelse(valid_predictions_prob > 0.5, 1, 0)
cat("\nNon-Oversampled GBDT Results:\n")
print(confusionMatrix(as.factor(valid_predictions_non_oversampled), as.factor(y_valid)))

test_predictions_prob <- predict(gbdt_model_non_oversampled, newdata = x_test, n.trees = 100, type = "response")
test_predictions_non_oversampled <- ifelse(test_predictions_prob > 0.5, 1, 0)
cat("\nNon-Oversampled GBDT Results:\n")
print(confusionMatrix(as.factor(test_predictions_non_oversampled), as.factor(y_test)))

#ROC
library(pROC)
roc_gradient_nosample <- roc(y_test, test_predictions_prob)



