library(caret)        #For Cross Validation
library(e1071)        #Required for SVM algo
library(magrittr)     #needed for one hot encoding
library(ggcorrplot)   #For data Visualization 
library(mice)         #For Multiple Imputation
library(nnet)         #Multiple Regression
library(faraway)      #For Checking VIF Scores

mydata<-read.csv("house-data.csv",header = TRUE)
mydata
summary(mydata)
View(mydata)


# Data Preprocessing ------------------------------------------------------

mydata[sapply(mydata, is.character)] <- lapply(mydata[sapply(mydata, is.character)], as.factor)
summary(mydata)
drop <- c("Id","Alley","PoolQC","Fence","MiscFeature")#Deleting Columns as More than 1100 Values are missing
mydata = mydata[,!(names(mydata) %in% drop)]

View(mydata)

# MICE Imputation to Deal With Missing Data---------------------------------------------------------

#Stage 1 of MICE
imputation<-mice(mydata,seed=23345,method = "cart",m=5)
# print(imputation)
# imputation$imp
# stripplot(imputation,pch=20,cex=1.2)

#Stage 2 of MICE
model.fit<-with(imputation,lm(SalePrice~LotFrontage,data = mydata))
summary(model.fit)

#Stage 3 of MICE
pooled.model<-pool(model.fit)
summary(pooled.model)
final_data<-complete(imputation,3)
View(final_data)


# Data Visualization ------------------------------------------------------
x<-round(cor((final_data[sapply(final_data, is.numeric)])),digits = 4)#corelation of only numeric features

ggcorrplot(cor(x)) #Cor-relation of Heat map 
View(x)

hist(final_data$SalePrice,data = final_data,labels=TRUE,xlab = "Sales Price")
hist(final_data$OverallCond,data=final_data,labels = TRUE,xlab = "Overall Condition")
boxplot(final_data$LotFrontage~final_data$Street,main="Box Plot for Street with Linear feet of street connected to property",ylab="Linear feet of street connected to property",las=1,xlab = "Street(Type of Road Access)")

anv<-aov(final_data$OverallCond~final_data$HouseStyle+final_data$Neighborhood
         +final_data$Foundation+final_data$Heating
         +final_data$KitchenQual+final_data$GarageType, data = final_data)
summary(anv)
summary(final_data)


# Applying Backward Stepwise Regression for Feature Selection -------------
fs<-expression(null.model1<-lm(SalePrice~1),model2<-step(null.model1,
                                                         scope = ~LotFrontage + LotArea + Street + Utilities+
                                                           LotConfig + Neighborhood + Condition1 + Condition2 + BldgType +HouseStyle +
                                                           OverallQual + OverallCond + YearBuilt +RoofStyle +RoofMatl + Exterior1st + 
                                                           MasVnrArea + ExterQual+ExterCond+Foundation + BsmtCond+BsmtQual + TotalBsmtSF + Heating +X1stFlrSF + 
                                                           X2ndFlrSF + LowQualFinSF+GrLivArea+FullBath+BedroomAbvGr + KitchenAbvGr + KitchenQual + Functional + 
                                                           Fireplaces + GarageArea + GarageCond+PavedDrive+PoolArea + MiscVal+MoSold + YrSold+SaleType + 
                                                           SaleCondition))
#step2
step.fit<-with(imputation,fs)
step.fit.models<-lapply(step.fit$analyses, formula)
step.fit.features<-lapply(step.fit.models, terms)
feature.frequency<-unlist(lapply(step.fit.features,labels))
feature.frequency
sort(table(feature.frequency))
#Selecting features which appeared for more than 3 times out of 5 Imputation.
f_data<-subset(final_data,select = c(SalePrice,BedroomAbvGr,BldgType,BsmtQual,
                                     Condition1,Condition2,Exterior1st,ExterQual,Fireplaces,
                                     Functional,GarageArea,KitchenAbvGr,KitchenQual,
                                     LotArea,MasVnrArea,MoSold,Neighborhood,OverallCond,
                                     OverallQual,PoolArea,RoofMatl,SaleCondition,
                                     SaleType,Street,TotalBsmtSF,X1stFlrSF,
                                     X2ndFlrSF,YearBuilt))


View(f_data)


# Dummy Encoding to deal with Categorical Data --------------------------------------------------------

 dummy<-dummyVars("~ .",data=f_data)
 newdata<-data.frame(predict(dummy,newdata=f_data))
 mydataone<-as.data.frame(newdata)
 View(mydataone)
 data_for_house_price<-mydataone

# Replacing Values in OverallCondition Variable ---------------------------

 mydataone$OverallCond<-replace(mydataone$OverallCond, mydataone$OverallCond>=7 & mydataone$OverallCond<=10, "Good")
 mydataone$OverallCond<-replace(mydataone$OverallCond, mydataone$OverallCond>=4 & mydataone$OverallCond<=6, "Average")
 mydataone$OverallCond<-replace(mydataone$OverallCond, mydataone$OverallCond>=1 & mydataone$OverallCond<=3, "Poor")
 
View(mydataone)




# Multinomial Logistic Regression Using 10-fold Cross Validation for Prediction of Overall Condition of house-----------------------------------------

mydataone$OverallCond<-as.factor(mydataone$OverallCond)
train.index <- createDataPartition(mydataone[,"OverallCond"],p=0.8,list=FALSE)
mydataone.trn <- mydataone[train.index,]
mydataone.tst <- mydataone[-train.index,]

mydataone.trn$OverallCond<-relevel(mydataone.trn$OverallCond, ref = "Good")

ctrl  <- trainControl(method  = "cv",number  = 10) 
fit.cv <- train(OverallCond ~ ., data = mydataone.trn, method = "multinom",
                trControl = ctrl)

#Confusion Marix of Train Model
pred1 <- predict(fit.cv,mydataone.trn)
mat2<-table(pred1,mydataone.trn$OverallCond)
mat2
sum(diag(mat2))/sum(mat2)


#Confusion Matrix of Test Model  
mydataone.tst$OverallCond<-relevel(mydataone.tst$OverallCond, ref = "Good")
pred <- predict(fit.cv,mydataone.tst)
mat1<-table(pred,mydataone.tst$OverallCond)
mat1
sum(diag(mat1))/sum(mat1)






#SVM for predicting overall condition of house-------------------------------------
svm_model<-svm(mydataone.trn$OverallCond~.,data=mydataone.trn)
summary(svm_model)

#confusion matrix and mis-classification error of model
pred<-predict(svm_model,mydataone.tst)
confusionMatrix(table(mydataone.tst[,"OverallCond"],pred))
mat1<-table(pred,mydataone.tst$OverallCond)
sum(diag(mat1))/sum(mat1)


#KNN for Predicting Overall Condition of House-----------------------------------------------------
fit.cv <- train(OverallCond ~ ., data = mydataone.trn, method = "knn",
  trControl = ctrl,tuneGrid =data.frame(k=10))
pred <- predict(fit.cv,mydataone.tst)
confusionMatrix(table(mydataone.tst[,"OverallCond"],pred))
print(fit.cv)



#Question 3
#Use of Stratified Sampling to Separate Train and Test Data.
library(rsample)
stratify<-initial_split(data_for_house_price,prop = 0.75)
stratify
#Creating Training Data Set
st_train<-training(stratify)
#Creating Testing Data Set
st_test<-testing(stratify)

#Defining RSQUARE function to calculate R2 for Test data.
RSQUARE = function(y_actual,y_predict){
  cor(y_actual,y_predict)^2
}

# Linear Regression to predict house prices using Stratified Sampling-------------------------------
model<-lm(st_train$SalePrice~.,data = st_train)
summary(model)
pred_1<-predict(model,st_test)
RSQUARE(st_test$SalePrice,pred_1)

#SVM to Predict House Prices using Stratified Sampling------------------------------------------------
model_svm<-svm(st_train$SalePrice~.,data = st_train,kernel="linear",cost=1.0)
summary(model_svm)
pred_svm<-predict(model_svm,st_test)
RSQUARE(st_test$SalePrice,pred_svm)



# Separating Train and Test Data using Random Sampling  -------------------
tr<-sample(1:nrow(data_for_house_price),size = 1095)#keeping same size as that of Stratified
random_train<-data_for_house_price[tr,]
random_test<-data_for_house_price[-random_train,]

#Applying Linear Regression on Random Samples
random_model1<-lm(random_train$SalePrice~.,data=random_train)
summary(random_model1)
pred_model1<-predict(random_model1,random_test)
RSQUARE(random_test$SalePrice,pred_model1)

#Applying SVM on random samples
random_model2<-svm(random_train$SalePrice~.,data=random_train,kernel="linear",cost=1.0)
summary(random_model2)
pred_model2<-predict(random_model2,random_test)
RSQUARE(random_test$SalePrice,pred_model2)


#Question 4: Cluster Analysis  

library(factoextra) 
library(cluster)
#Load in the data.
house_data<-read.csv("house-data.csv", header=T)
#Select required rows of said data.
hd_original <- house_data[,c(13, 51)]
#Remove rows with missing values.
hd <- na.omit(hd_original)
#Scale each variable to have a mean of 0 and sd of 1.
hd <- scale(hd)
#Create a plot of the number of clusters vs. the total within the sum of squares.
fviz_nbclust(hd, kmeans, method = "wss") 
#This outputs Additional Figure 1
#From the graph this produces, we can see there is a slight bend in the graph at 5, so we set K, number of clusters, to 5.

#Make the results reproducible by setting a seed.
set.seed(1)
#Perform k-means clustering with k = 5 clusters.
km <- kmeans(hd, centers = 5, nstart = 25)
#View the results.
km
#Plot results of final k-means model.
fviz_cluster(km, data = hd)
#Now, use the aggregate() function to find the mean of the variables in each cluster.
aggregate(hd_original, by=list(cluster=km$cluster), mean)






