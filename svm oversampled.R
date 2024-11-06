
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

# Oversample the minority class in the training set
minority_class <- train_data %>% filter(TARGET == "1")
majority_class <- train_data %>% filter(TARGET == "0")

# Calculate how many more samples are needed to balance the classes
n_minority <- nrow(minority_class)
n_majority <- nrow(majority_class)
n_needed <- n_majority - n_minority

# Randomly sample with replacement from the minority class
set.seed(123)
oversampled_minority <- minority_class[sample(1:n_minority, n_needed, replace = TRUE), ]

# Combine the oversampled minority class with the original majority class
train_data_balanced <- rbind(majority_class, minority_class, oversampled_minority)

# Normalize numeric variables
num_vars <- c("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH")
train_means <- apply(train_data_balanced[num_vars], 2, mean)
train_sds <- apply(train_data_balanced[num_vars], 2, sd)

train_data_balanced[num_vars] <- scale(train_data_balanced[num_vars], center = train_means, scale = train_sds)
valid_data[num_vars] <- scale(valid_data[num_vars], center = train_means, scale = train_sds)
test_data[num_vars] <- scale(test_data[num_vars], center = train_means, scale = train_sds)

# One-hot encoding for categorical variables
dummies <- dummyVars(~ . - TARGET, data = train_data_balanced)
train_data_encoded <- predict(dummies, newdata = train_data_balanced) %>% as.data.frame()
train_data_encoded$TARGET <- train_data_balanced$TARGET

valid_data_encoded <- predict(dummies, newdata = valid_data) %>% as.data.frame()
valid_data_encoded$TARGET <- valid_data$TARGET

test_data_encoded <- predict(dummies, newdata = test_data) %>% as.data.frame()
test_data_encoded$TARGET <- test_data$TARGET

# Remove near-zero variance predictors
nzv <- nearZeroVar(train_data_encoded, saveMetrics = TRUE)
train_data_encoded <- train_data_encoded[, !nzv$zeroVar & !nzv$nzv]
valid_data_encoded <- valid_data_encoded[, !nzv$zeroVar & !nzv$nzv]
test_data_encoded <- test_data_encoded[, !nzv$zeroVar & !nzv$nzv]

# Train the SVM model on the oversampled training data
set.seed(123)
svm_model <- train(
  TARGET ~ ., data = train_data_encoded,
  method = "svmRadial",
 # trControl = trainControl(method = "cv", number = 3),
  tuneLength = 1,
 probability = TRUE
)

# Make predictions on the validation set
valid_predictions <- predict(svm_model, newdata = valid_data_encoded)

# Create a confusion matrix for the validation set
conf_matrix <- confusionMatrix(valid_predictions, valid_data_encoded$TARGET)

# Display the confusion matrix and statistics
print(conf_matrix)

#ROC
test_prob <- predict(svm_model, newdata = test_data_encoded, type = "prob")
true_labels <- test_data_encoded$TARGET
roc_curve <- roc(true_labels, test_prob[, "1"]) 