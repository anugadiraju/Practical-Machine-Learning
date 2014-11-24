Practical Machine Learning course Project
========================================================

Background - Machine learning course project
---------------------------------------------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a 
large amount of data about personal activity relatively inexpensively. In this project, 
our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

#### Data 
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:    https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#### Course Project Requirements
The goal of our project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. 

The sections below describe the following:
*how the model was built
*how cross validation was used 
*how the out of sample error was estimated 
*And what the final inferences were

Getting and cleaning Data
--------------------------


```r
   library(data.table)
    require(Hmisc)
```

```
## Loading required package: Hmisc
## Loading required package: grid
## Loading required package: lattice
## Loading required package: survival
## Loading required package: splines
## Loading required package: Formula
## 
## Attaching package: 'Hmisc'
## 
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
    library(caret)
```

```
## Loading required package: ggplot2
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```r
    library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:Hmisc':
## 
##     combine
```

```r
    options(warn=-1)
    setwd("C:")
    if(!file.exists("data")) {
        dir.create("data")
    }
```

Get training data.

```r
    fileUrl2 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" 
    download.file(fileUrl2,destfile="./data/fitnessdata.csv")
    fitDT <- fread("./data/fitnessdata.csv",header=T,na.strings=c("NA","Not available"),colClasses="character")
    fitDT <- as.data.frame(fitDT)
```
    
We examine the data set and columns. We will remove columns not used in prediction, such as user_name,V1,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window and num_window. We willfurther clean, by getting rid of variables starting with kurtosis, skewness, max and min as they are empty or contain NA values. We will also remove  all columns starting with amplitude, std, var,and avg. Use nearZeroVar to further remove variables with zero or nearzero variance.(nearZeroVar didn't indicate any more removal for variables.)
   
  

```r
      cleanFitDT <- fitDT[, -which(names(fitDT) %in% c("user_name","V1","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window"))]
    
    cleanFitDT <- cleanFitDT[, - grep("^kurtosis|^skewness|^max|^min",colnames(cleanFitDT))]
    
    cleanFitDT <- cleanFitDT[,-grep("^amplitude|^std|^var|^avg",colnames(cleanFitDT))]
```

```r
 x <-  nearZeroVar(cleanFitDT,saveMetrics = TRUE)
```

   
 Get and cleanup test data similar to training data above.It is important to apply same preprocessing methodology to test set as well.

```r
   fileUrl1 <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" 
   download.file(fileUrl1,destfile="./data/fittestdata.csv")
   fitTestDT <- fread("./data/fittestdata.csv",header=T,na.strings=c("NA","Not available"),colClasses="character")
   fitTestDT <- as.data.frame(fitTestDT)
   cleanTDT <- fitTestDT[, -which(names(fitTestDT) %in% c("user_name","V1","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window"))]
    cleanTDT <- cleanTDT[, - grep("^kurtosis|^skewness|^max|^min",colnames(cleanTDT))]
   cleanTDT <- cleanTDT[,-grep("^amplitude|^std|^var|^avg",colnames(cleanTDT))]
```
  
Building a Model
------------------

   ### Create training, test and validation data splits (or partitions)
   
   

```r
   trainIndex = createDataPartition(y=cleanFitDT$classe,p=0.3,list=FALSE)
   training = cleanFitDT[trainIndex,]
   totalTesting = cleanFitDT[-trainIndex,]
   
   valIndex = createDataPartition(y=totalTesting$classe,p=0.5,list=FALSE)
   testing = totalTesting[-valIndex,]
   valid = totalTesting[valIndex,]
```

### Using Cross Validation and Estimating Error

   Do cross validation and estimate error, using validation data set asisde specifically for this purpose. Cross validation step is used for two main purposes:
   We use code similar to the one below to test different models to compare the error rate against each other and then choose the final model. In this case randomforest performed the best.
   We then use similar cross validation function with different parameters for the chosen model to get an optimum fit.


```r
   k = 10   # number of folds
    confusionFrame <- as.data.frame(matrix(ncol=5,nrow =k))
    names(confusionFrame) = c("A","B","C","D","E")
     ## a data frame of 5 columns (one for each category abcde) and 10 rows - 1 for each fold) to hold class error estimates
    n = floor(nrow(valid)/k)
    for(i in 1:k) {
        index1 = ((i-1)*n +1)
        index2 = (i*n)
        subset = index1:index2
        cvTrain <- valid[-subset,]
        cvTest <- valid[subset,]
        fitRF = randomForest(as.factor(classe) ~.,data=cvTrain,importance=TRUE,ntree=200)
        predictRF <- predict(fitRF,cvTest)
        confusion1 <- fitRF$confusion
        confusionFrame[i,] <- confusion1[,"class.error"]
        #print(paste("Classification Error for", i))
        #print(confusion1[,"class.error"])
    }
    print(paste("Average Classification Error for RandomForest with 10-fold Cross validation:"))
```

```
## [1] "Average Classification Error for RandomForest with 10-fold Cross validation:"
```

```r
    print(colMeans(confusionFrame))
```

```
##           A           B           C           D           E 
## 0.005265104 0.023811753 0.022674768 0.018485351 0.009675402
```
    
### Fitting a randomforest model

   The above error rate indicates that randomforest model we used is a pretty good fit. Use the randomforest model on on train data and create the model. 
   print the model, look at the variable importance and plot the variable importance. The variables roll_belt, yaw_belt,magnet_dumbel_z,  pitch_belt are on the top of our variable importance list, indicating that these are good predictors.
   

```r
   set.seed(325)
   rf1 <- randomForest(as.factor(classe) ~.,data=training,importance=TRUE,ntree=200)
   varImpPlot(rf1)
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) 

```r
   rf1
```

```
## 
## Call:
##  randomForest(formula = as.factor(classe) ~ ., data = training,      importance = TRUE, ntree = 200) 
##                Type of random forest: classification
##                      Number of trees: 200
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 2%
## Confusion matrix:
##      A    B    C   D    E class.error
## A 1667    6    1   0    0 0.004181601
## B   26 1102   10   1    1 0.033333333
## C    0   19 1005   1    2 0.021421616
## D    3    0   32 926    4 0.040414508
## E    0    1    6   5 1071 0.011080332
```

### Predicting on the model
   
   Predict the outcome on the modle using the testing partition of data. We can guess if our predictions are right,
   
   

```r
    predictRF <- predict(rf1,testing)
    testing$predictRight <- predictRF == testing$classe
    badpredictions <- testing$predictRight[testing$predictRight == FALSE]
```


### Final Results and analysis
  
  plot the confusionMtrix from randome forest rf1

   The y-axis shows the predicted class for all items, and the x-axis shows the actual class. The tiles are coloured according to the frequency of the intersection of the two classes thus the diagonal represents where we predict the actual class. The colour represents the relative frequency of that observation in our data; Any row of tiles (save for the diagonal) represents instances where we falsely identified items as belonging  to the specified class. 
   In the rendered plot we can see our predictions were very accurate.
   

```r
   onfusion1 <- rf1$confusion

   plotC <- ggplot(as.data.frame(as.table(confusion1)))
   
   plotC + geom_tile(aes(x=Var1, y=Var2, fill=Freq)) + scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + scale_fill_gradient(breaks=seq(from=-.5, to=4, by=.2)) + labs(fill="Normalized\nFrequency")
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png) 

  
   
### Now predict using the real test data - 

```r
  predictTest <- predict(rf1,cleanTDT)
  #pml_write_files(predictTest)
```

  



