#Removing any pre-existing objects
rm(list=ls())

#Setting the directory
print("Setting the current working directory")
setwd("C:/Users/Akash/Desktop/Works/data frames/Edwisor/Santander project")

#loading the libraries
print("Loading the libraries")
x = c("randomForest", "ggplot2", "reshape2", "rpart", "ROSE", "dplyr", "ROCR","sampling", "caret", "inTrees", "e1071")


#loading all packages at once and then removing x as its of no use
lapply(x, require, character.only = TRUE)
rm(x)

#----------------------------------------------------------------------------------
#loading the data
print("Loading the data")
train= read.csv("train.csv")
print("Check first few rows of the data")
head(train)

#checking distribution of target variable in the train dataset
print("Distribution of target variable in the training dataset")
table(train$target)

#----------------------------------------------------------------------------------

#Missing Value Analysis
print("Missing Value Analyssis")

#checking missing values
missing_val= data.frame(apply(train,2,function(x){sum(is.na(x))}))

missing_val$columns= row.names(missing_val)
row.names(missing_val)= NULL

#renaming first var name as Missing_percentage
names(missing_val)[1]= "Missing_percentage"

#sorting the dataframe in descending order according to Missing_percentage variable
missing_val= missing_val[order(-missing_val$Missing_percentage),]

#rearranging the columns
missing_val= missing_val[,c(2,1)]

print("Sum of Percentage of Missing values in all variables is:")
print(sum(missing_val$Missing_percentage))
print("We can see that there are no missing values in any of the variables in the dataset")
#----------------------------------------------------------------------------------

#Saving column names in variables
#storing all variable names in col_names variable

col_names= colnames(train)
col_names= as.list(col_names)

#storing continuous variables in cont_names variable

cont_names= colnames(train)
cont_names= as.list(col_names)
cont_names[1:2] <- NULL   #removing columns other than continuous type


#------------------------------------------------------------------------------------

#Outlier Analysis
print("Outlier Analysis")
outlier_data= train[,grep("^[var]", names(train), value=TRUE)]
outlier_count=list()
for (i in cont_names){
  val= outlier_data[,i][outlier_data[,i]%in% boxplot.stats(outlier_data[,i])$out]
  outlier_count[i]= (length(val))
}

#Storing variable names and outlier count in dataframe outlier_count
outlier_count= do.call(rbind, Map(data.frame, "Features"= cont_names, "OutlierCount"= outlier_count))

#sorting the Outlier Count column in descending order
outlier_count <- outlier_count[order(-outlier_count$OutlierCount),]

#top 50 features acc to outlier count
outlier_count<- head(outlier_count,n=50)

#printing top 50 outlier variables with their outlier count
print("Top 50 outlier variables with highest outliers count are:")
print(outlier_count)

#storing top 50 features names(according to outlier count) in feature_names variable
feature_names= outlier_count[,1]

#plotting box plot of top 50 variables
#selecting only those features from outlier_data which contains highest count of outliers
top_outliers= outlier_data[,feature_names]
meltTrain= melt(top_outliers)
meltTrain= subset(meltTrain, variable=feature_names)
p <- ggplot(meltTrain, aes(factor(variable), value)) 
p + geom_boxplot() + facet_wrap(~variable, scale="free")


#----------------------------------------------------------------
#plotting histogram of few variables

for (col in 1:200) {
  hist(outlier_data[,col])
}

# NOTE: We can see that each variable is normally distributed, hence we can impute/replace outliers with Mean

#Imputing Outliers with mean
#Replacing all outliers with NA

for (i in cont_names){
  val = outlier_data[,i][outlier_data[,i] %in% boxplot.stats(outlier_data[,i])$out]
  outlier_data[,i][outlier_data[,i]%in% val] = NA
}


#filling NAs with mean
for(i in 1:ncol(outlier_data)){
  outlier_data[is.na(outlier_data[,i]), i] <- mean(outlier_data[,i], na.rm = TRUE)
}

outlier_data$target= train$target

#rearranging the columns(keeping target column first)
outlier_data= outlier_data[,c(201,1:200)]

#calculating important features through correlation

print("Feature Selection")

print("Calculating Important Features through Correlation and they are:")
cor_df= cor(train[,2:202])
cor_target = abs(cor_df[,"target"])
relevant_features = cor_target[cor_target>0.02]
relevant_features= names(relevant_features)

#printing the names of important variables
print(relevant_features)

#taking out the df with relevant features
relevant_feature_df= outlier_data[,relevant_features]


#making data ready for the model
#converting target column to factor type
relevant_feature_df$target= as.factor(relevant_feature_df$target)


#dividing data into train and test data using stratified sampling
set.seed(1234)


#DOING UNDERSAMPLING
data_balanced_under <- ovun.sample(target ~ ., data = relevant_feature_df, method = "under",N = 100000)$data

stratas= strata(data_balanced_under, c("target"), size=c(20000,20000), method="srswor")
data_balanced_under= getdata(data_balanced_under, stratas)
data_balanced_under= data_balanced_under[,1:124]


set.seed(1234)
train.under_index= createDataPartition(data_balanced_under$target, p=0.80, list= FALSE)

#making training and test data
under_train_sample= data_balanced_under[train.under_index,]
under_test_sample= data_balanced_under[-train.under_index,]

#Checking distribution of classes in the target variable
print("Distribution of category in target variable after Undersampling")
table(data_balanced_under$target)

#DOING OVERSAMPLING
data_balanced_over <- ovun.sample(target ~ ., data = relevant_feature_df, method = "over",N = 300000)$data


stratas= strata(data_balanced_over, c("target"), size=c(100000,100000), method="srswor")
data_balanced_over= getdata(data_balanced_over, stratas)
data_balanced_over= data_balanced_over[,1:124]

print("Distribution of category in target variable after Oversampling")
table(data_balanced_over$target)

set.seed(1234)
train.over_index= createDataPartition(data_balanced_over$target, p=0.80, list= FALSE)

#making training and test data
over_train_sample= data_balanced_over[train.over_index,]
over_test_sample= data_balanced_over[-train.over_index,]


#-----------------------------------------------------------------------------------------
#Decision tree
#using rpart function
#undersampling
print("Implementing Decision Tree for Undersampled Data")

library(rpart.plot)

model= rpart(target~., data=under_train_sample, method= "class")

rpart.plot(model)


#predicting class labels
pred_label= predict(model, under_test_sample[,1:123], type="class")

#predicting class prob
pred_prob= predict(model, under_test_sample[,1:123], type="prob")

#AUC
pred_prob$max= pmax(pred_prob[,1], pred_prob[,2])

pred_ROCR <- prediction(pred_prob$max, under_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Decision Tree(Undersampled)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]

#evaluating results
CM= table(under_test_sample[, 124], predicted= pred_label)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]

#Evaluation Metrics 
print("Evaluation Metrics of Decision Tree- Under Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))
print("We can see that the AUC score is very less, Accuracy of the model is also around 60% and
      the FNR and FPR are also high so we will try Oversampling the data and the running the model over it")


#----------------------------------------------------------------------------------------------
#doing over sampling
library(rpart.plot)

print("Implementing Decision Tree for Oversampled Data")

model= rpart(target~., data=over_train_sample, method= "class")

rpart.plot(model)


#predicting
pred_label= predict(model, over_test_sample[,1:123], type="class")
pred_prob= predict(model, over_test_sample[,1:123], type="prob")


#AUC metric
pred_prob$max= pmax(pred_prob[,1], pred_prob[,2])

pred_ROCR <- prediction(pred_prob$max, over_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Decision Tree(OverSampled)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
auc_ROCR

#evaluating results
CM= table(over_test_sample[, 124], predicted= pred_label)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]

#Evaluation Metrics 
print("Evaluation Metrics of Decision Tree- Over Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))
print("We can see that the AUC score is still very less, Accuracy of the model is also around 60% and
      the FNR and FPR are still high so we will try Random Forest which is collection of Decision Trees")


print("-------------------------------------------------------------------------------------------------------------------------------")

#------------------------------------------------------------------------------------------

#Random Forest
print("Implementing Random Forest for Undersampled Data")

#undersampling
RF_model= randomForest(target~., under_train_sample, importance=TRUE, ntree=50, nodesize=5)

pred_prob= predict(RF_model, under_test_sample , type = "prob")

pred_prob= data.frame(pred_prob)

#extracting rules from random forest
#transform rf object to intrees format

treeList= RF2List(RF_model)

#extracting rules
exec= extractRules(treeList, under_train_sample[,-124])

#visualizing a rule
exec[1:2,]


#make rules more readable(this will give colnames for var)
readableRules= presentRules(exec, colnames(under_train_sample))

#get rule metrics
ruleMetric= getRuleMetric(exec, under_train_sample[,-124], under_train_sample$target)

#seeing the rule metric data
ruleMetric[1:2, ]

#predicting test data
RF_predictions= predict(RF_model, under_test_sample[,-124])

#calculating AUC
library(ROCR)
pred_prob$max= pmax(pred_prob[,1], pred_prob[,2])

pred_ROCR <- prediction(pred_prob$max, under_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Random Forest(Undersampling)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]


#evaluating metrics
CM= table(under_test_sample$target, RF_predictions)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]


#Evaluation Metrics 
print("Evaluation Metrics of Random Forest - Under Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))


#NOTE-We are getting good accuracy and other classification metrics but we are getting less AUC score
print("We are getting good accuracy and other classification metrics but we are getting less AUC score, so we will try oversampling the train data 
      and running the model over it")


#Random Forest-OVERSAMPLING
print("Implementing Random Forest for Oversampled Data")
RF_model= randomForest(target~., over_train_sample, importance=TRUE, nodesize=5, ntree=50)

pred_prob= predict(RF_model, over_test_sample , type = "prob")

pred_prob= data.frame(pred_prob)

#extracting rules from random forest
#transform rf object to intrees format

treeList= RF2List(RF_model)

#extracting rules
exec= extractRules(treeList, over_train_sample[,-124])

#visualizing a rule
exec[1:2,]


#make rules more readable(this will give colnames for var)
readableRules= presentRules(exec, colnames(over_train_sample))

#get rule metrics
ruleMetric= getRuleMetric(exec, over_train_sample[,-124], over_train_sample$target)

#seeing the rule metric data
ruleMetric[1:2, ]

#predicting test data
RF_predictions= predict(RF_model, over_test_sample[,-124])

#calculating AUC
library(ROCR)
pred_prob$max= pmax(pred_prob[,1], pred_prob[,2])

pred_ROCR <- prediction(pred_prob$max, over_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Random Forest(Oversampling)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]


#evaluating metrics
CM= table(over_test_sample$target, RF_predictions)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]

#Evaluation Metrics 
print("Evaluation Metrics of Random Forest - Over Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))

print("We can see that the model seems overfitted on the model and is giving unexpected results. Even on specifying nodesize the model remain overfitted,
      so we will try different algorithm that is Logistic Regression")


print("-------------------------------------------------------------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------
#logistic regression  (logistic reg will only take balanced dataset)
print("Implementing Logistic Regression on Under Sampled dataset")

#UNDERSAMPLING
logit_model= glm(target~., data= under_train_sample, family= "binomial")

#summary of the model
summary(logit_model)

#predicting probabilities
logit_predictions_pred= predict(logit_model, newdata= under_test_sample[,1:123], type = "response") 

#conv prob to 1,0
logit_predictions_label= ifelse(logit_predictions_pred > 0.5, 1, 0)

#calculating AUC
library(ROCR)

pred_ROCR <- prediction(logit_predictions_pred, under_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Logistic Regression(Undersampled)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]

#evaluate the performance of classification
CM= table(under_test_sample$target, logit_predictions_label)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]

print("Evaluation Metrics of Logistic Regression - Under Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))


print("We can see that the logistic regression model is giving AUC score of around 0.80 which is a good score. Also, the Accuracy score, Specificity
      and Sensitivity is high and FNR, FPR are low, which is great")



#OVERSAMPLING
print("Implementing Logistic Regression on Over Sampled dataset")

logit_model= glm(target~., data= over_train_sample, family= "binomial")

#summary of the model
summary(logit_model)

logit_predictions_pred= predict(logit_model, newdata= over_test_sample[,1:123], type = "response") 

#conv prob to 1,0
logit_predictions_label= ifelse(logit_predictions_pred > 0.5, 1, 0)

#calculating AUC
library(ROCR)

pred_ROCR <- prediction(logit_predictions_pred, over_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Logistic Regression(Oversampled)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]


#evaluate the performance of classification
CM= table(over_test_sample$target, logit_predictions_label)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]

print("Evaluation Metrics of Logistic Regression - Over Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))

print("We can see that the logistic regression model(trained on Oversampled Data) is giving slightly better results thatn earlier Logistic Regression model
      which was being trained on Undersampled Data")

print("-------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------------
#Naive bayes
#UNDERSAMPLING

print("Implementing Naive Bayes on Under Sampled dataset")

NB_model= naiveBayes(target~., data= under_train_sample)

#calculating the probabilities
NB_predictions= predict(NB_model, under_test_sample[,1:123], type="raw")

#converting prob into labels
NB_predictions_label= predict(NB_model, under_test_sample[,1:123], type="class")

CM= table(observed= under_test_sample[,124], predicted= NB_predictions_label)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]

#calculating AUC
library(ROCR)
NB_predictions$max= pmax(NB_predictions[,1], NB_predictions[,2])

pred_ROCR <- prediction(NB_predictions$max, under_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Naive Bayes(Undersampled)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]

print("Evaluation Metrics of Naive Bayes - Under Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))

print("We can see that the Naive Bayes model(trained on Undersampled Data) is giving high Accuracy score, Specificity
      and Sensitivity is high and FNR, FPR are low, which is great but it's also giving less AUC score")



#OVERSAMPLING

print("Implementing Naive Bayes on Over Sampled dataset")

NB_model= naiveBayes(target~., data= over_train_sample)

#calculating the probabilities
NB_predictions= predict(NB_model, over_test_sample[,1:123], type="raw")

#converting prob into labels
NB_predictions_label= predict(NB_model, over_test_sample[,1:123], type="class")

#calculating AUC
library(ROCR)
NB_predictions$max= pmax(NB_predictions[,1], NB_predictions[,2])

pred_ROCR <- prediction(NB_predictions$max, over_test_sample$target)

roc_ROCR <- performance(pred_ROCR, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, main = "ROC curve for Naive Bayes(Oversampled)", colorize = T)
abline(a = 0, b = 1)

auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]


CM= table(observed= over_test_sample[,124], predicted= NB_predictions_label)
TN= CM[1,1]
FN= CM[2,1]
TP= CM[2,2]
FP= CM[1,2]

print("Evaluation Metrics of Naive Bayes - Over Sampled Data")
print(CM)
print(paste0("AUC score is: ", auc_ROCR))
print(paste0("Accuracy of the model is:", ((TP + TN) / (TP + TN + FP + FN))*100))
print(paste0("Classification Error Rate is:", ((FP + FN) / (TP + TN + FP + FN))*100))
print(paste0("Sensitivity is:", (TP / (FN + TP))*100))
print(paste0("Specificity is:", (TN / (TN + FP))*100))
print(paste0("False Negative Rate:", ((FN*100)/(FN+TP))))
print(paste0("False Positive Rate:", (FP / (TN + FP))*100))
print(paste0("Precision is:", (TP / (TP + FP))*100))

print("Just like Naive Bayes trained on Undersampled Data We can see that the Naive Bayes model(trained on Oversampled Data) is giving high
Accuracy score, Specificity and Sensitivity is high and FNR, FPR are low, which is great but it's also giving less AUC score")

print("-------------------------------------------------------------------------------------------------------------------------------")

#Applying logistic Regression on test data

#logistic regression on test data

#selecting the relevant features from the test data so that we can run the trained model on it
print("After checking all the above models we can see that Logistic Regression trained on Oversampled Data it is giving better result than others.")
print("Applying Logistic Regression(Trained on Oversampled Data) on the test data")
relevant_features_test= relevant_features[-1]
test=read.csv("test.csv")
test_file_copy<-data.frame(test)
test_file_copy= test_file_copy[,relevant_features_test]

#predicting the values
logit_predictions_test= predict(logit_model, newdata= test_file_copy, type = "response") 

#conv prob to 1,0
logit_predictions_label_test= ifelse(logit_predictions_test > 0.5, 1, 0)

#adding predicted values to the test dataset
test$target= logit_predictions_label_test

#writing the test file with the predicted values
print("Saving the test file with predictions in the directory")
write.csv(test, "C:/Users/Akash/Desktop/Works/data frames/Edwisor/Santander project/Test_file_with_predictions.csv",row.names = FALSE)
print("Test file Saved with predicted values")
