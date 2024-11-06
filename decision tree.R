# load the required libraries
#install.packages("DMwR2")
#install.packages("ROSE")
#install.packages("ggplot2")
#install.packages("lattice")
library(caret)       # for data partitioning
library(rpart)       # for decision tree model
library(ROSE)        # for rose oversampling
library(dplyr)       # for data manipulation



# read the sampled data
data <- read.csv("sampled_data.csv")

# ensure the target variable is a factor
data$TARGET <- as.factor(data$TARGET)

# set seed for reproducibility
set.seed(42)

# split the data into train 70%, validation 15%, and test sets 15%
trainIndex <- createDataPartition(data$TARGET, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
temp_data <- data[-trainIndex, ]

validationIndex <- createDataPartition(temp_data$TARGET, p = 0.5, list = FALSE)
validation_data <- temp_data[validationIndex, ]
test_data <- temp_data[-validationIndex, ]

# check the dimensions of the datasets
dim(train_data)   # 70% of the dataset
dim(validation_data)  # 15% of the dataset
dim(test_data)    # 15% of the dataset

# 1.non-oversampled
# train a decision tree model on the non-oversampled data
decision_tree_model <- rpart(TARGET ~ ., 
                             data = train_data, 
                             method = "class", 
                             weights = weights,
                             control = rpart.control(minsplit = 20, maxdepth = 10))



weights <- ifelse(train_data$TARGET == "1", 13, 1)
decision_tree_model <- rpart(TARGET ~ ., data = train_data, method = "class", weights = weights)


# use the model to make predictions on the validation set
val_predictions <- predict(decision_tree_model, validation_data, type = "class")

# evaluate the model performance on the validation set
conf_matrix_val <- confusionMatrix(val_predictions, as.factor(validation_data$TARGET))

# predict on the test set for final evaluation
test_predictions <- predict(decision_tree_model, test_data, type = "class")
conf_matrix_test <- confusionMatrix(test_predictions, as.factor(test_data$TARGET))

# compute metrics: accuracy, sensitivity, specificity, TP, FP, TN, FN
accuracy <- conf_matrix_test$overall['Accuracy']
sensitivity <- conf_matrix_test$byClass['Sensitivity']
specificity <- conf_matrix_test$byClass['Specificity']
TP <- conf_matrix_test$table[2, 2]
FP <- conf_matrix_test$table[1, 2]
TN <- conf_matrix_test$table[1, 1]
FN <- conf_matrix_test$table[2, 1]

# print the results
print(paste("non-oversampled accuracy : ", accuracy))
print(paste("non-oversampled sensitivity : ", sensitivity))
print(paste("non-oversampled specificity : ", specificity))
print(paste("TP: ", TP, " FP: ", FP, " TN: ", TN, " FN: ", FN))

# 2.oversampling using rose
# transform categorical variables to factors
train_data$NAME_CONTRACT_TYPE <- as.factor(train_data$NAME_CONTRACT_TYPE)
train_data$CODE_GENDER <- as.factor(train_data$CODE_GENDER)
train_data$NAME_INCOME_TYPE <- as.factor(train_data$NAME_INCOME_TYPE)
train_data$NAME_EDUCATION_TYPE <- as.factor(train_data$NAME_EDUCATION_TYPE)
train_data$NAME_FAMILY_STATUS <- as.factor(train_data$NAME_FAMILY_STATUS)

# check the structure of the data
str(train_data)

library(rpart)

weights <- ifelse(train_data$TARGET == "1", 15, 1)
decision_tree_model <- rpart(TARGET ~ ., 
                             data = train_data, 
                             method = "class", 
                             weights = weights,
                             control = rpart.control(minsplit = 20, maxdepth = 20))

# oversample the minority class using ROSE
train_data_rose <- ROSE(TARGET ~ ., data = train_data, seed = 42)$data

# train a decision tree model on the ROSE oversampled data
decision_tree_model_rose <- rpart(TARGET ~ ., data = train_data_rose, method = "class")

# evaluate the model performance on the validation set
val_predictions_rose <- predict(decision_tree_model_rose, validation_data, type = "class")
conf_matrix_val_rose <- confusionMatrix(val_predictions_rose, as.factor(validation_data$TARGET))

# predict on the test set for final evaluation
test_predictions_rose <- predict(decision_tree_model_rose, test_data, type = "class")
conf_matrix_test_rose <- confusionMatrix(test_predictions_rose, as.factor(test_data$TARGET))

# compute metrics: accuracy, sensitivity, specificity, TP, FP, TN, FN
accuracy_rose <- conf_matrix_test_rose$overall['Accuracy']
sensitivity_rose <- conf_matrix_test_rose$byClass['Sensitivity']
specificity_rose <- conf_matrix_test_rose$byClass['Specificity']
TP_rose <- conf_matrix_test_rose$table[2, 2]
FP_rose <- conf_matrix_test_rose$table[1, 2]
TN_rose <- conf_matrix_test_rose$table[1, 1]
FN_rose <- conf_matrix_test_rose$table[2, 1]

# print the results
print(paste("Accuracy of ROSE oversampling : ", accuracy_rose))
print(paste("Sensitivity of ROSE oversampling: ", sensitivity_rose))
print(paste("Specificity of ROSE oversampling: ", specificity_rose))
print(paste("TP: ", TP_rose, " FP: ", FP_rose, " TN: ", TN_rose, " FN: ", FN_rose))

# ROC
# NONE OVERSAMPLE
test_probabilities <- predict(decision_tree_model, test_data, type = "prob")[, 2]  
roc_DecisionTree_nosample <- roc(as.numeric(as.factor(test_data$TARGET)) - 1, test_probabilities)

#OVERSAMPLE
test_probabilities_rose <- predict(decision_tree_model_rose, test_data, type = "prob")[, 2]  
roc_DecisionTree <- roc(as.numeric(as.factor(test_data$TARGET)) - 1, test_probabilities_rose)
