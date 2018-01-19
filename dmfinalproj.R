#Installing and loading  required packages
install.packages("VIM")
install.packages("mlbench")
install.packages("ggplot2")
install.packages("vcd")
install.packages("reshape2")
install.packages("randomForest")
install.packages("rpart")
install.packages("RGtk2")
install.packages("stringi")
install.packages("caret")
install.packages("rattle")
install.packages('rpart.plot')
install.packages('RColorBrewer')
install.packages('clusterSim')
install.packages('recommenderlab')
library(VIM)
require(mlbench)
require(ggplot2)
require(vcd)
require(reshape2)
library(class)
library(randomForest)
require(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(clusterSim)
library(recommenderlab)

#Reading input csv files
census_train_data <- read.csv("censustrain.csv")

#Giving the names to the attributes to census data csv files
colnames(census_train_data) <- c( "Age","Workclass","fnlwgt","Education", "Education-num", "Marital-status",
                                 
                                 "Occupation", "Relationship", "Race", "Sex", "Capital-gain", "Capital-loss", 
                                 
                                 "Hours-per-week", "Native-country", "Income")

#Plotting the box plots for each attribute
boxplot(census_train_data, ylim = c(0, 100), main = "Type, range and distribution of attributes before cleaning and normalization of train data",  ylab = "Number", las = 2, cex.axis = 0.7)

#Replacing all the ? in census test data with NAs
for (i in 1:32561) 
{
  for (j in 1:14)
  { 
    if (census_train_data[i,j] == " ?")
    {
      census_train_data[i,j] <- NA
    }
  }
}

summary(census_train_data)

#Generate barplots for number of missing values and their proportion against variables
# Sum up the missing values for each variable
var_missing_census_data<- sapply(census_train_data,function(x)sum(is.na(x)))

# Ordering the variables as per number of missing values
var_missing_census_data<- var_missing_census_data[order(var_missing_census_data)]
#Creating dataframe and plotting the variables as per missing values
missing_census_data<- data.frame(variable=names(var_missing_census_data),missing=var_missing_census_data,missing.prop=var_missing_census_data/dim(census_train_data)[1],stringsAsFactors=FALSE)
missing_census_data$variable<- factor(missing_census_data$variable,levels=missing_census_data$variable,ordered=FALSE)
ggplot(data=missing_census_data,aes(x=variable,y=missing)) + geom_bar(stat="identity", color="white", fill="palegreen") + labs(x="Attribute",y="Number of missing values in training data") + ggtitle("Plot for Number of Missing Values per attribute for 32561 training records")+ theme(axis.text.x=element_text(angle=45, hjust=1))


#Imputing the missing values using KNN imputations
Imputed_census_train_data <-kNN(census_train_data, variable = c("Native-country", "Workclass", "Occupation"), k = 7)
Imputed_census_train_data <- Imputed_census_train_data[,-c(16:18)]
summary(Imputed_census_train_data)

#Preprocessing the data
preProcValues <- preProcess(Imputed_census_train_data, method = c("center", "scale"))
Imputed_census_train_data <- predict(preProcValues, Imputed_census_train_data)
boxplot(Imputed_census_train_data, ylim = c(0, 100), main = "Type, range and distribution of attributes after preprocessing of train data",  ylab = "Number", las = 2, cex.axis = 0.7)
#Giving the names to the attributes
colnames(Imputed_census_train_data) <- c("Age","Workclass","fnlwgt","Education","Educationnum","Maritalstatus",
                      
                     "Occupation","Relationship","Race","Sex","Capitalgain","Capitalloss", 
                      
                     "Hoursperweek","Nativecountry","Income")

set.seed(3033)

#Partioning the data in 70% training and 30% testing
intrain <- createDataPartition(y = Imputed_census_train_data$Income, p= 0.7, list = FALSE)
training <- Imputed_census_train_data[intrain,]
testing <- Imputed_census_train_data[-intrain,]

#Checking if the training data is balanced or not
cols <- c("chartreuse","chocolate")
plot(training$Income, col = cols, border = c( "Red" , "Blue"), ylab = "Number of Instances",
     xlab = "Income Class", main = "Number of Instances of each Income class in Training Data")

#Up sampling the training data as the data is imbalanced
up_training <- upSample(x = training[, -ncol(training)], y = training$Income) 

#Plotting the training data again after cleaning and normalization
boxplot(up_training, ylim = c(0, 100), main = "Type, range and distribution of attributes after cleaning and normalization of train data",  ylab = "Number", las = 2, cex.axis = 0.7)

#Capturing the test data and the classes of the test data separately (to use these as ground truth
#while checking the accuracy of the models)
TestData <- testing[,1:14]
TestClasses <- testing[,15]
 
#conversion of Income variable to factor variable
training[["Income"]] = factor(training[["Income"]])

#Creating the control for cross-validations
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE)

colnames(up_training) <- c("Age","Workclass","fnlwgt","Education", "Educationnum", "Maritalstatus",
                      
                      "Occupation", "Relationship", "Race", "Sex", "Capitalgain", "Capitalloss", 
                      
                      "Hoursperweek", "Nativecountry", "Income")

#Creating prediction model using random forest
model_rf <- randomForest(Income ~ ., data=up_training, ntree=500, importance=TRUE, trControl = trctrl)
#model_rf <- randomForest(Income ~ ., data=up_training, ntree=500, importance=TRUE)

#plotting the random forest model
layout(matrix(c(1,2),nrow=1),
       width=c(4,1)) 
par(mar=c(5,4,4,0))
plot(model_rf, log="y")
par(mar=c(5,0,4,2))
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(model_rf$err.rate),col=1:4,cex=0.8,fill=1:4)

#Plotting Variable Importance
varImpPlot(model_rf)

#Running the random forest model on test data to get the corresponding Income labels
pred_rf <- predict(model_rf,TestData)

#Checking the performance of model by getting the confusion matrix for random forest method
confusionMatrix(pred_rf, TestClasses)

#Plotting confusion mapping after using Random Forest algorithm without cross-validations
TrueClass <- factor(c('<=50K', '<=50K', '>50K', '>50K'))
PredictedClass <- factor(c('<=50K', '>50K', '<=50K', '>50K'))
Y      <- c(7400, 16, 1777, 575)
df_rf_wocv <- data.frame(TrueClass, PredictedClass, Y)

ggplot(data =  df_rf_wocv, mapping = aes(x = TrueClass, y = PredictedClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "cornsilk3", high = "darkolivegreen1") +
  theme_bw() + theme(legend.position = "none") + ggtitle("Confusion Matrix after applying model from Random Forest algorithm obtained without cross-validations on test data")

#Plotting confusion mapping after using Random Forest algorithm with cross-validations
TrueClass <- factor(c('<=50K', '<=50K', '>50K', '>50K'))
PredictedClass <- factor(c('<=50K', '>50K', '<=50K', '>50K'))
Y      <- c(7404, 12, 1783, 569)
df_rf <- data.frame(TrueClass, PredictedClass, Y)

ggplot(data =  df_rf, mapping = aes(x = TrueClass, y = PredictedClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "cornsilk3", high = "darkolivegreen1") +
  theme_bw() + theme(legend.position = "none") + ggtitle("Confusion Matrix after applying model from Random Forest algorithm with cross-validations on test data")


#Creating prediction model using decision tree with cross-validations
model_tree<-rpart(Income ~ .,data=up_training, method="class")
#model_tree<-rpart(Income ~ .,data=up_training, method="class", xval = 5)

#Plotting the decision tree
fancyRpartPlot(model_tree)

#Running the random forest model on test data to get the corresponding Income labels
pred_tree <- predict(model_tree,TestData, type="class")

#Running the random forest model on test data to get the corresponding Income labels
confusionMatrix(pred_tree, TestClasses)

#Plotting confusion mapping after using Decision tree algorithm without cross-validations
TrueClass <- factor(c('<=50K', '<=50K', '>50K', '>50K'))
PredictedClass <- factor(c('<=50K', '>50K', '<=50K', '>50K'))
Y      <- c(5758, 1658, 1950, 402)
df_tree <- data.frame(TrueClass, PredictedClass, Y)

ggplot(data =  df_tree, mapping = aes(x = TrueClass, y = PredictedClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "cornsilk3", high = "darkolivegreen1") +
  theme_bw() + theme(legend.position = "none") + ggtitle("Confusion Matrix after applying model from Decision Tree algorithm without cross-validations on test data")

#Plotting confusion mapping after using Decision tree algorithm without cross-validations
TrueClass <- factor(c('<=50K', '<=50K', '>50K', '>50K'))
PredictedClass <- factor(c('<=50K', '>50K', '<=50K', '>50K'))
Y      <- c(5758, 1658, 1950, 402)
df_tree <- data.frame(TrueClass, PredictedClass, Y)

ggplot(data =  df_tree, mapping = aes(x = TrueClass, y = PredictedClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "cornsilk3", high = "darkolivegreen1") +
  theme_bw() + theme(legend.position = "none") + ggtitle("Confusion Matrix after applying model from Decision Tree algorithm without cross-validations on test data")

#Writing the results which are more accurate
write.csv(pred_rf, file = "GuptaClassification1.csv", row.names = FALSE)


