library(caret)
library(dplyr)
library(ggplot2)
library(lattice)
library(tidyverse)
library(janitor)

#install.packages("tidyverse")
#install.packages("janitor")


# read the data
data <- read_csv("sampled_data.csv")

data$TARGET <- as.factor(data$TARGET)
data$NAME_CONTRACT_TYPE <- as.factor(data$NAME_CONTRACT_TYPE)
data$CODE_GENDER <- as.factor(data$CODE_GENDER)
data$NAME_INCOME_TYPE <- as.factor(data$NAME_INCOME_TYPE)
data$NAME_EDUCATION_TYPE <- as.factor(data$NAME_EDUCATION_TYPE)
data$NAME_FAMILY_STATUS <- as.factor(data$NAME_FAMILY_STATUS)
data$REGION_RATING_CLIENT <- as.factor(data$REGION_RATING_CLIENT)

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

train_data_encoded<- clean_names(train_data_encoded)
valid_data_encoded <- clean_names(valid_data_encoded)
test_data_encoded <- clean_names(test_data_encoded)

# Make sure TARGET is a factor in the training, validation, and test sets
train_data_encoded$target <- as.factor(train_data_encoded$target)
valid_data_encoded$target <- as.factor(valid_data_encoded$target)
test_data_encoded$target <- as.factor(test_data_encoded$target)

library(ROSE)
# Over-sampling the train_data,use ROSE function
train_data_oversampled <- ROSE(target ~ ., data = train_data_encoded, seed = 123)$data
test_data_oversampled <- ROSE(target ~ ., data = test_data_encoded, seed = 123)$data

library(nnet)
library(neuralnet)
netmode=nnet(target~.,data=train_data_oversampled,size=12,rang=0.1,decay=5e-4,maxit=100)
predict1 <- predict(netmode,test_data_oversampled)
predict1 <- ifelse(predict1[,1] > 0.5,1,0)
confusionMatrix(factor(test_data_oversampled$target),factor(predict1))

netmode2=nnet(target~.,data=train_data_encoded,size=12,rang=0.1,decay=5e-4,maxit=100)
predict2 <- predict(netmode2,test_data_encoded)
predict2 <- ifelse(predict2[,1] > 0.5,1,0)
confusionMatrix(factor(test_data_encoded$target),factor(predict2))

#ROC
roc_Neural <- roc(test_data_oversampled$target, predict1)
roc_Neural_nosample <- roc(test_data_encoded$target, predict2)

