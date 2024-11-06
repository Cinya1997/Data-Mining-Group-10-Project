# main.R

library(knitr)
library(pROC)

purl("XGBOOT.Rmd", output = "XGBOOT.R")
purl("Logistic Regression.Rmd", output = "Logistic Regression.R")
purl("K-Nearest Neighbors (KNN).Rmd", output = "K-Nearest Neighbors (KNN).R")



# load source 
source("XGBOOT.R")  
source("Logistic Regression.R")  
source("K-Nearest Neighbors (KNN).R")  
source("Neural Network.R")  
source("decision tree.R")  
source("data mining gradient boosted non oversampling.R")  
source("gradient boosted oversampled.R")  
source("naive bayes both.R")  

source("svm oversampled.R")  
source("svm non oversampled.R")  


# Function to calculate record counts and rejection percentages
get_summary <- function(data, dataset_name) {
  total_records <- nrow(data)
  rejected_count <- sum(data$TARGET == 1)
  rejected_percentage <- (rejected_count / total_records) * 100
  
  return(data.frame(
    Dataset = dataset_name,
    Total_Records = total_records,
    Rejected_Count = rejected_count,
    Rejected_Percentage = rejected_percentage
  ))
}

# Create summaries for each dataset
original_train_summary <- get_summary(train_data, "Original Train Data")
original_valid_summary <- get_summary(valid_data, "Original Validation Data")
original_test_summary <- get_summary(test_data, "Original Test Data")

oversampled_train_summary <- get_summary(train_data_oversampled, "Oversampled Train Data")
oversampled_valid_summary <- get_summary(valid_data_encoded2, "Oversampled Validation Data")
oversampled_test_summary <- get_summary(test_data_encoded2, "Oversampled Test Data")

# Combine all summaries into one data frame
summary_results <- rbind(
  original_train_summary,
  original_valid_summary,
  original_test_summary,
  oversampled_train_summary,
  oversampled_valid_summary,
  oversampled_test_summary
)

# Print the summary results
print(summary_results)

#---------------- Data distribution-oversample---------------------------------------------------------------------
# training SET
train_distribution <- table(train_data$TARGET)
train_percentage <- prop.table(train_distribution) * 100
cat("Train Data Distribution:\n")
print(train_distribution)
cat("Percentage:\n")
print(train_percentage)

# valid set
valid_distribution <- table(valid_data$TARGET)
valid_percentage <- prop.table(valid_distribution) * 100
cat("\nValidation Data Distribution:\n")
print(valid_distribution)
cat("Percentage:\n")
print(valid_percentage)

# test set
test_distribution <- table(test_data$TARGET)
test_percentage <- prop.table(test_distribution) * 100
cat("\nTest Data Distribution:\n")
print(test_distribution)
cat("Percentage:\n")
print(test_percentage)



# ROC-AUC 
#------------------------------Oversample-----------------------------------------------------------------------
# XGBOOT 
plot(roc_xgboost, col = "blue", main = "ROC Curve for Different Models", lwd = 2)
auc_xgboost <- auc(roc_xgboost)
print(auc_xgboost)

#Logistic Regression
plot(roc_LogReg, col = "red",add = TRUE, lwd = 2)
auc_LogReg <- auc(roc_LogReg)
print(auc_LogReg)

#KNN
plot(roc_KNN, col = "orange",add = TRUE, lwd = 2)
auc_KNN <- auc(roc_KNN)
print(auc_KNN)

# Neural Network
plot(roc_Neural, col = "lightblue",add = TRUE, lwd = 2)
auc_Neural <- auc(roc_Neural)
print(auc_Neural)

# Decision Tree
plot(roc_DecisionTree, col = "pink",add = TRUE, lwd = 2)
auc_DecisionTree <- auc(roc_DecisionTree)
print(auc_DecisionTree)

# Random forest
plot(roc_random, col = "purple",add = TRUE, lwd = 2)
auc_random <- auc(roc_random)
print(auc_random)

# Gradient Boosted 
plot(roc_gradient, col = "green",add = TRUE, lwd = 2)
auc_gradient <- auc(roc_gradient)
print(auc_gradient)

# Naive Bayes  
plot(roc_naive, col = "#FFFF00",add = TRUE, lwd = 2)
auc_naive <- auc(roc_naive)
print(auc_naive)

# SVM  
plot(roc_SVM, col = "grey",add = TRUE, lwd = 2)
auc_SVM <- auc(roc_SVM)
print(auc_SVM)


# 添加图例
legend("bottomright", legend = c("XGBoost", "Logistic Regression", "KNN","Neural Network","Decision Tree","Random forest","Gradient Boosted","Naive Bayes","SVM"),
       col = c("blue", "red", "orange","lightblue","pink","purple","green","#FFFF00","grey"), lwd = 2)

#----NON-OVERSAMPLE---------------------------------------------------------------------------------------------------
# ROC-AUC 
# XGBOOT
plot(roc_xgboost_nonsample, col = "blue", main = "ROC Curve for Different Models", lwd = 2)
auc_xgboost_nosample <- auc(roc_xgboost_nonsample)
print(auc_xgboost_nosample)

#Logistic Regression
plot(roc_LogReg_nosample, col = "red", add = TRUE, lwd = 2)
auc_LogReg_nosample <- auc(roc_LogReg_nosample)
print(auc_LogReg_nosample)

#KNN
plot(roc_kNN_nosample, col = "orange",add = TRUE, lwd = 2)
auc_KNN_nosample <- auc(roc_kNN_nosample)
print(auc_KNN_nosample)

# Neural Network
plot(roc_Neural_nosample, col = "lightblue",add = TRUE, lwd = 2)
auc_Neural_nosample <- auc(roc_Neural_nosample)
print(auc_Neural_nosample)

# Decision Tree
plot(roc_DecisionTree_nosample, col = "pink",add = TRUE, lwd = 2)
auc_DecisionTree_nosample <- auc(roc_DecisionTree_nosample)
print(auc_DecisionTree_nosample)

# Random forest
plot(roc_random_nosample, col = "purple",add = TRUE, lwd = 2)
auc_random_nosample <- auc(roc_random_nosample)
print(auc_random_nosample)

# Gradient Boosted 
plot(roc_gradient_nosample, col = "green",add = TRUE, lwd = 2)
auc_gradient_nosample <- auc(roc_gradient_nosample)
print(auc_gradient_nosample)
# Naive Bayes  
plot(roc_naive_nosample, col = "#FFFF00",add = TRUE, lwd = 2)
auc_naive_nosample <- auc(roc_naive_nosample)
print(auc_naive_nosample)

# SVM  
plot(roc_SVM_nosample, col = "grey",add = TRUE, lwd = 2)
auc_SVM_nosample <- auc(roc_SVM_nosample)
print(auc_SVM_npsample)


# 添加图例
legend("bottomright", legend = c("XGBoost", "Logistic Regression", "KNN","Neural Network","Decision Tree","Random forest","Gradient Boosted","Naive Bayes","SVM"),
       col = c("blue", "red", "orange","lightblue","pink","purple","green","#FFFF00","grey"), lwd = 2)