# Answer for the Machine Learning Assignment.

This purpose of this report is produce a prediction model with high accuracy.

Proir uploading the data, let's upload the required library for the codes that we may use in this report.


```r
## Library loading & packages ####
library(rpart) ; library(randomForest) ; library(caret)
```

Then downloading the training and testing data from the provided url


```r
trainurl<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testurl<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainurl,des="training.csv",mode="wb")
download.file(testurl,des="testing.csv",mode="wb")
```

Reading the data into memory, and split the data into training and testing dataset which will be used for cross validation.


```r
training<-read.csv("training.csv",header=TRUE)
testing<-read.csv("testing.csv",header=TRUE)
inTrain<-createDataPartition(y=training$classe,p=.6,list=FALSE)
training.1<-training[inTrain,]
testing.1<-training[-inTrain,]
```

Before applying any prediction algorithm, we should process the dataset and ensuring the relevant features / variables.
1- Removing the 1st column "X"

```r
# Remove the 1st column
training.1<-training.1[,-1]
```

2- Removing the features with non zero variance 


```r
# Remove near zero Var
training.nzv<-nearZeroVar(training.1,saveMetrics=TRUE)$nzv
training.2<-training.1[!training.nzv]
```

3- Removing the features that have too many missing data. 


```r
# Remove the the features that has too many missing data NA. i.e more 90% are missing

getFractionMissing <- function(df = rawActitivity) {
  colCount <- ncol(df)
  returnDf <- data.frame(index=1:ncol(df),
                         columnName=rep("undefined", colCount),
                         FractionMissing=rep(-1, colCount),
                         stringsAsFactors=FALSE)
  for(i in 1:colCount) {
    colVector <- df[,i]
    missingCount <- length(which(colVector == "") * 1)
    missingCount <- missingCount + sum(is.na(colVector) * 1)
    returnDf$columnName[i] <- as.character(names(df)[i])
    returnDf$FractionMissing[i] <- missingCount / length(colVector)
  }
  
  return(returnDf)
}


list.Fraction.Missing<-getFractionMissing(training.2)
training.col.names.wo.missing<-list.Fraction.Missing[list.Fraction.Missing$FractionMissing<.95,]$columnName
training.3<-training.2[training.col.names.wo.missing]
```


Refelect the above changes in the training dataset on both testing dataset i.e the one which will be used for cross validation and the 2nd which will be used for the final validation.


```r
# Apply all preprocessing on the testing datasets
col.names.all<-colnames(training.3)
col.names.wo.classe<-colnames(training.3[,-58])
testing.1<-testing.1[col.names.all]
testing<-testing[col.names.wo.classe]

# binding one row from training into testing dataset the removing it again from the testing dataset in order to ensure the testing dataset has exactly the same class of all features in the training dataset as well as all factor variables has the same no. of lvels between testing and training dataset. This
testing <- rbind(training.3[1, -58] , testing) #note row 2 does not mean anything, this will be removed right.. now:
testing <- testing[-1,]
```

Now, it is time to apply the prediction algorithms. We will start with Decision Tree.


```r
#  Apply Decision tree algorithm

model.fit.1<-rpart(classe ~ ., data=training.3, method="class")
pred.mod.fit.1.1<-predict(model.fit.1, testing.1 , type = "class")
confusionMatrix(pred.mod.fit.1.1, testing.1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2139   67    6    0    0
##          B   60 1240   69   65    0
##          C   33  203 1274  153   56
##          D    0    8   12  872   88
##          E    0    0    7  196 1298
## 
## Overall Statistics
##                                         
##                Accuracy : 0.8696        
##                  95% CI : (0.862, 0.877)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.8351        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9583   0.8169   0.9313   0.6781   0.9001
## Specificity            0.9870   0.9693   0.9313   0.9835   0.9683
## Pos Pred Value         0.9670   0.8647   0.7411   0.8898   0.8648
## Neg Pred Value         0.9835   0.9566   0.9847   0.9397   0.9773
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2726   0.1580   0.1624   0.1111   0.1654
## Detection Prevalence   0.2819   0.1828   0.2191   0.1249   0.1913
## Balanced Accuracy      0.9727   0.8931   0.9313   0.8308   0.9342
```

```r
# fancyRpartPlot(model.fit.1)
```

The accuracy from the decision tree algorithm is 0.8696151

So, let's try Random Forest algorithm.


```r
# Apply Random Forest algorithm

model.fit.2 <- randomForest(classe ~. , data=training.3)
pred.mod.fit.2.1<- predict(model.fit.2, testing.1, type = "class")
confusionMatrix(pred.mod.fit.2.1, testing.1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1517    1    0    0
##          C    0    0 1366    4    0
##          D    0    0    1 1282    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9993   0.9985   0.9969   0.9986
## Specificity            0.9998   0.9998   0.9994   0.9995   1.0000
## Pos Pred Value         0.9996   0.9993   0.9971   0.9977   1.0000
## Neg Pred Value         1.0000   0.9998   0.9997   0.9994   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1933   0.1741   0.1634   0.1835
## Detection Prevalence   0.2846   0.1935   0.1746   0.1638   0.1835
## Balanced Accuracy      0.9999   0.9996   0.9990   0.9982   0.9993
```

The accuracy from the Random Forest algorithm is 0.9988529
So, we should expect a similar out of sample error when we test this prediction into the provided test dataset.


