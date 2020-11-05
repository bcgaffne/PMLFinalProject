---
title: "Practical Machine Learning - Final Project"
author: "ME"
date: "October 26, 2020"
output: html_document
---

## Project Description  

From Project Description:  
*Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).*  
*The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.*

###0.Libraries
```{r Libraries}
        library(dplyr)
        library(caret)
        
```

###1.Get Data

```{r dataDownload}
        training<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

        testing<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

```

###3.Data Cleaning

There are various columns containing empty values and N/As. There are also columns with near zero variance which will have very little impact on modelling results. There columns will be removed as part of the data cleaning process.  
```{r DataCleaning}
        train_rmNA<-training[,colSums(is.na(training))==0]
        NZV<-nearZeroVar(train_rmNA)
        train_noNZV<-train_rmNA[,-NZV]

```

In additions, variables 1:6  (names, timestamps etc.) will be removed to simplify the modelling process.

```{r DataCleaning 2}

        train_Clean<-train_noNZV[,-c(1:6)]

```

###2.Splitting the Data  

Here we will split the data into training and test sets. 75% of the cleaned dataset will be used for training and the remaining 25% will be used for testing accuracy of the final model.  

```{r DataSplit}
#na remove

        
        inTrain<-createDataPartition(y=train_Clean$classe,p=0.75,list=F)
        train_data<-train_Clean[inTrain,]
        test_data<-train_Clean[-inTrain,]
        

```

###3.Training the Model 

Here we will start training some models to predict class. We will start with Random Forests as in a lot of cases the random forest algorithm makes good predictions relative to others. Random Forests also require less pre-processing relative to other models. 

We can adjust the train control argument in the train function to add K-Fold cross validation. Here we add 3 folds. 

```{r RandomForest}
        set.seed(123)
        rfMdl<-train(classe~.,data=train_data,method="rf",trControl=trainControl(method="cv",number = 3))
        rfMdl

```

The random forest model appears to show 98% accuracy on the training set (2% in sample error rate). We will now test this model against the test dataset.

###4. Testing the Model

```{r Testing RF Model}
        
        Class_predict<-predict(rfMdl,newdata=test_data)
        rf_CM<-confusionMatrix(Class_predict,as.factor(test_data$classe))
        rf_CM

```

The random forest model predicts the test datset with 99% accuracy, resulting in an out-of-Sample error rate of 1%. This is a very good result. Hence, we can use this model to predict on the blind test (i.e. quiz)

###5.Quiz Prediction
```{r Quiz Prediction}

        Quiz_prediction<-predict(rfMdl,newdata = testing)
        Quiz_prediction

```