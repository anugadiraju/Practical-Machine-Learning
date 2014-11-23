
## this function assists in programming assignment submission. This creates 20 different output files from the input
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

## 
## This function is used to do a 10-fold cross validation to estimate the errors and further refine our randomforest model.
cross_validate_rf <- function(valid) {
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
       # errorVector[i] <- roc.area(as.matrix(cvTest),predictRF)
        print(paste("Classification Error for", i))
       print(confusion1[,"class.error"])
    }
    print(paste("Average Classification Error for RandomForest with 10-fold Cross validation:"))
    print(colMeans(confusionFrame))
}

predict1 <- function() {
    library(data.table)
    require(Hmisc)
    library(caret)
    library(randomForest)
    options(warn=-1)
    setwd("C:/Users/agadiraju/Documents/RWorkingDir/Practical-Machine-Learning")
    if(!file.exists("data")) {
        dir.create("data")
    }
    
    ## Get and clean data
    
   ## Get training data.
   fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" 
   download.file(fileUrl2,destfile="./data/fitnessdata.csv")
    fitDT <- fread("./data/fitnessdata.csv",header=T,na.strings=c("NA","Not available"),colClasses="character")
    fitDT <- as.data.frame(fitDT)
    #names(fitDT)
    
    ## remove columns not used in prediction, such as user_name from fitDT and gather them in cleanFitDT  keep or loose it?
    
    cleanFitDT <- fitDT[, -which(names(fitDT) %in% c("user_name","V1","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window"))]
    
    ## further clean, by getting rid of variables starting with kurtosis, skewness, max and min as they are empty or contain NA values.
    cleanFitDT <- cleanFitDT[, - grep("^kurtosis|^skewness|^max|^min",colnames(cleanFitDT))]
    
    ## further remove  all amplitude, std, var,and avg
    cleanFitDT <- cleanFitDT[,-grep("^amplitude|^std|^var|^avg",colnames(cleanFitDT))]
   
   ##  read and cleanup testDT similar to training data above.
   fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" 
   download.file(fileUrl1,destfile="./data/fittestdata.csv")
   fitTestDT <- fread("./data/fittestdata.csv",header=T,na.strings=c("NA","Not available"),colClasses="character")
   fitTestDT <- as.data.frame(fitTestDT)
   #names(fitTestDT)
   
   ## remove useless columns, such as user_name from fitDT and gather them in cleanFitDT -  num_window ?? keep or loose it?
   
   cleanTDT <- fitTestDT[, -which(names(fitTestDT) %in% c("user_name","V1","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window"))]
   
   ## further clean, by getting rid of variables starting with kurtosis, skewness, max and min as they are empty or contain NA values.
   cleanTDT <- cleanTDT[, - grep("^kurtosis|^skewness|^max|^min",colnames(cleanTDT))]
   
   ## further remove  all amplitude, std, var,and avg
   cleanTDT <- cleanTDT[,-grep("^amplitude|^std|^var|^avg",colnames(cleanTDT))]
   
   ## Use nearZeroVar to further remove variables with zero or nearzero variance.
   ## nearZeroVar didn't indicate any more removal for variables.
   ## spliting 10% and 90% into train and predict sets didn't really help with performance,
   ## caret train still couldn't model this.
   
   x <-  nearZeroVar(cleanFitDT,saveMetrics = TRUE)
   
   ## create training and test data and validation data splits (or partitions)
   
   trainIndex = createDataPartition(y=cleanFitDT$classe,p=0.3,list=FALSE)
   training = cleanFitDT[trainIndex,]
   totalTesting = cleanFitDT[-trainIndex,]
   
   valIndex = createDataPartition(y=totalTesting$classe,p=0.5,list=FALSE)
   testing = totalTesting[-valIndex,]
   valid = totalTesting[valIndex,]
   
   ## cross validation and estimate error, using validation data set asisde specifically for this purpose.
   
   cross_validate_rf(valid)
    
   set.seed(325)

   ## Now use the randomforest model on on train data and create the model. 
   ## print the model, look at the variable importance and plot the variable importance.
   rf1 <- randomForest(as.factor(classe) ~.,data=training,importance=TRUE,ntree=200)
   rf1$importance
   varImpPlot(rf1)
   rf1

   ## now predict on the testing partition of data.
   predictRF <- predict(rf1,testing)


   ## Now we can guess if our predictions are right 
   ##  table(prediction, testing$classe)
   ## testing$predictRight <- predict == testing$classe
   print(rf1)

   ## plot the confusionMtrix from randome forest rf1
   # The y-axis shows the predicted class for all items, and the x-axis shows the actual class. The tiles are coloured according to the frequency of the intersection of the two classes thus the diagonal represents where we predict the actual class. 
   # The colour represents the relative frequency of that observation in our data; 
   # given some classes occur more frequently we normalize the values before plotting.
   # Any row of tiles (save for the diagonal) represents instances where we falsely identified items as belonging 
   # to the specified class. In the rendered plot we can se were often identified for items 
   # belonging to all other classes.

   confusion1 <- rf1$confusion

   plotC <- ggplot(as.data.frame(as.table(confusion1)))
   
   plotC + geom_tile(aes(x=Var1, y=Var2, fill=Freq)) + scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + scale_fill_gradient(breaks=seq(from=-.5, to=4, by=.2)) + labs(fill="Normalized\nFrequency")
   

   testing$predictRight <- predictRF == testing$classe
   badpredictions <- (testing$predictRight == FALSE)
   badpredictions <- testing$predictRight[testing$predictRight == FALSE]

  ## Now predict using the real test data - 
  predictTest <- predict(rf1,cleanTDT)
  pml_write_files(predictTest)
}
   
