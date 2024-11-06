
library(caret)
library(dplyr)
library(ggplot2)
library(lattice)
library(e1071)
library(DMwR2)

# load data
data <- read.csv("sampled_data.csv")

# Format columns as factors
data$TARGET <- as.factor(data$TARGET)
data$NAME_CONTRACT_TYPE <- as.factor(data$NAME_CONTRACT_TYPE)
data$CODE_GENDER <- as.factor(data$CODE_GENDER)
data$NAME_INCOME_TYPE <- as.factor(data$NAME_INCOME_TYPE)
data$NAME_EDUCATION_TYPE <- as.factor(data$NAME_EDUCATION_TYPE)
data$NAME_FAMILY_STATUS <- as.factor(data$NAME_FAMILY_STATUS)
data$REGION_RATING_CLIENT <- as.factor(data$REGION_RATING_CLIENT)

# Split the data into training, validation, and test sets
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

# One-hot encoding for categorical variables
dummies <- dummyVars(~ . - TARGET, data = train_data)
train_data_encoded <- predict(dummies, newdata = train_data) %>% as.data.frame()
train_data_encoded$TARGET <- train_data$TARGET

valid_data_encoded <- predict(dummies, newdata = valid_data) %>% as.data.frame()
valid_data_encoded$TARGET <- valid_data$TARGET

test_data_encoded <- predict(dummies, newdata = test_data) %>% as.data.frame()
test_data_encoded$TARGET <- test_data$TARGET

# Remove near-zero variance predictors
nzv <- nearZeroVar(train_data_encoded, saveMetrics = TRUE)
train_data_encoded <- train_data_encoded[, !nzv$zeroVar & !nzv$nzv]
valid_data_encoded <- valid_data_encoded[, !nzv$zeroVar & !nzv$nzv]
test_data_encoded <- test_data_encoded[, !nzv$zeroVar & !nzv$nzv]

# Train the SVM model
set.seed(123)
library(e1071)
svm_model <- svm(TARGET ~ ., data = train_data_encoded, probability = TRUE)
test_prob <- predict(svm_model, newdata =test_data_encoded, probability = TRUE)
prob_values <- attr(test_prob, "probabilities")

# Make predictions on the validation set
test_predictions <- predict(svm_model, newdata = test_data_encoded)

# Create a confusion matrix for the validation set
conf_matrix <- confusionMatrix(test_predictions, test_data_encoded2$TARGET)

# Display the confusion matrix and statistics
print(conf_matrix)

#ROC
library(pROC)
positive_prob <- prob_values[, "1"]
roc_SVM_nosample <- roc(test_data_encoded2$TARGET, positive_prob)

