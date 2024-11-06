# Load required libraries
library(xgboost)
library(caret)
library(dplyr)
library(ggplot2)
library(lattice)
library(e1071)
library(smotefamily)
library(gbm)

# Load data
data <- read.csv("sampled_data.csv")

# View data format and convert data types
data$NAME_CONTRACT_TYPE <- as.factor(data$NAME_CONTRACT_TYPE)
data$CODE_GENDER <- as.factor(data$CODE_GENDER)
data$NAME_INCOME_TYPE <- as.factor(data$NAME_INCOME_TYPE)
data$NAME_EDUCATION_TYPE <- as.factor(data$NAME_EDUCATION_TYPE)
data$NAME_FAMILY_STATUS <- as.factor(data$NAME_FAMILY_STATUS)
data$REGION_RATING_CLIENT <- as.factor(data$REGION_RATING_CLIENT)

# Split data into training, validation, and test sets
set.seed(123)
train_index <- createDataPartition(data$TARGET, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
temp_data <- data[-train_index, ]

valid_index <- createDataPartition(temp_data$TARGET, p = 0.5, list = FALSE)
valid_data <- temp_data[valid_index, ]
test_data <- temp_data[-valid_index, ]

# Normalize numeric variables
num_vars <- c("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH")
train_means <- apply(train_data[num_vars], 2, mean)
train_sds <- apply(train_data[num_vars], 2, sd)

train_data[num_vars] <- scale(train_data[num_vars], center = train_means, scale = train_sds)
valid_data[num_vars] <- scale(valid_data[num_vars], center = train_means, scale = train_sds)
test_data[num_vars] <- scale(test_data[num_vars], center = train_means, scale = train_sds)

# One-hot encode categorical variables
dummies <- dummyVars(~ . - TARGET, data = train_data)
train_data_encoded <- predict(dummies, newdata = train_data) %>% as.data.frame()
train_data_encoded$TARGET <- train_data$TARGET

valid_data_encoded <- predict(dummies, newdata = valid_data) %>% as.data.frame()
valid_data_encoded$TARGET <- valid_data$TARGET

test_data_encoded <- predict(dummies, newdata = test_data) %>% as.data.frame()
test_data_encoded$TARGET <- test_data$TARGET

# Convert TARGET to numeric for encoding consistency
train_data_encoded$TARGET <- as.numeric(as.character(train_data_encoded$TARGET))
valid_data_encoded$TARGET <- as.numeric(as.character(valid_data_encoded$TARGET))
test_data_encoded$TARGET <- as.numeric(as.character(test_data_encoded$TARGET))

# Prepare x_train and y_train
x_train <- as.matrix(train_data_encoded[, -ncol(train_data_encoded)])  # Exclude TARGET column
y_train <- train_data_encoded$TARGET

x_valid <- as.matrix(valid_data_encoded[, -ncol(valid_data_encoded)])
y_valid <- valid_data_encoded$TARGET

x_test <- as.matrix(test_data_encoded[, -ncol(test_data_encoded)])
y_test <- test_data_encoded$TARGET

# Build the model without oversampling for comparison
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

# Predict the model without oversampling
x_valid <- as.data.frame(x_valid)
x_test <- as.data.frame(x_test)

valid_predictions_prob <- predict(gbdt_model_non_oversampled, newdata = x_valid, n.trees = 100, type = "response")
valid_predictions_non_oversampled <- ifelse(valid_predictions_prob > 0.5, 1, 0)
cat("\nNon-Oversampled GBDT Results on Validation Set:\n")
## 
## Non-Oversampled GBDT Results on Validation Set:
print(confusionMatrix(as.factor(valid_predictions_non_oversampled), as.factor(y_valid)))
 

test_predictions_prob <- predict(gbdt_model_non_oversampled, newdata = x_test, n.trees = 100, type = "response")
test_predictions_non_oversampled <- ifelse(test_predictions_prob > 0.5, 1, 0)
cat("\nNon-Oversampled GBDT Results on Test Set:\n")
## 
## Non-Oversampled GBDT Results on Test Set:
print(confusionMatrix(as.factor(test_predictions_non_oversampled), as.factor(y_test)))
        
             
# Apply SMOTE using smotefamily
smote_result <- SMOTE(train_data_encoded[, -ncol(train_data_encoded)], 
                      train_data_encoded$TARGET, 
                      K = 5, dup_size = 2)

# Combine the oversampled data and labels
train_data_oversampled <- as.data.frame(smote_result$data)
names(train_data_oversampled)[ncol(train_data_oversampled)] <- "TARGET"  # Set the target column name

# Convert TARGET back to numeric if needed
train_data_oversampled$TARGET <- as.numeric(as.character(train_data_oversampled$TARGET))

# Prepare x_train and y_train for the oversampled data
x_train_oversampled <- as.matrix(train_data_oversampled[, -ncol(train_data_oversampled)])  # Exclude TARGET column
y_train_oversampled <- train_data_oversampled$TARGET

# Build the model with the oversampled training data
gbdt_model_oversampled <- gbm(
  formula = TARGET ~ .,
  distribution = "bernoulli", 
  data = train_data_oversampled,
  n.trees = 100,               
  interaction.depth = 3,       
  shrinkage = 0.1,             
  n.minobsinnode = 10,         
  verbose = FALSE              
)

# Predict on the validation and test sets using the oversampled model
valid_predictions_prob_oversampled <- predict(gbdt_model_oversampled, newdata = x_valid, n.trees = 100, type = "response")
valid_predictions_oversampled <- ifelse(valid_predictions_prob_oversampled > 0.5, 1, 0)
cat("\nOversampled GBDT Results on Validation Set:\n")
## 
## Oversampled GBDT Results on Validation Set:
print(confusionMatrix(as.factor(valid_predictions_oversampled), as.factor(y_valid)))

test_predictions_prob_oversampled <- predict(gbdt_model_oversampled, newdata = x_test, n.trees = 100, type = "response")
test_predictions_oversampled <- ifelse(test_predictions_prob_oversampled > 0.5, 1, 0)
cat("\nOversampled GBDT Results on Test Set:\n")
## 
## Oversampled GBDT Results on Test Set:
print(confusionMatrix(as.factor(test_predictions_oversampled), as.factor(y_test)))
             
#ROC
roc_gradient <- roc(y_test, test_predictions_prob_oversampled)

