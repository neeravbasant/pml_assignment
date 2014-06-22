Practicle Machine Learning Writeup Assignment
========================================================

### Brief Introduction to this report.

There are devices such as *Jawbone Up, Nike FuelBand, and Fitbit* with which we can now possible to collect a large amount of data about personal activity relatively inexpensively.
In this report my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the manner in which 6 participants did the exercise. This report also contains how I built my model, how I used cross validation and what I think the expected out of sample error is. At the end of this report there is a prediction for 20 different test cases.

Data for this report can be found in reference section.

#### Loading required library

```r
library(RANN)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```


#### Loading Data(Test and Train data)

```r
pml.train <- read.csv("pml-training.csv")
pml.test <- read.csv("pml-testing.csv")
```

#### Data Prepration

splitting trainig dataset into train and test with a ratio of 7/3 of 10.


```r
set.seed(1234)
trainingIndex <- createDataPartition(pml.train$classe, list = FALSE, p = 0.7)
train = pml.train[trainingIndex, ]
test = pml.train[-trainingIndex, ]
```

Filter the numeric features and outcome. 

```r
num_idx = which(lapply(train, class) %in% c("numeric"))
```

Fixing the  missing values. This includes the imputation method usin K-nearest neighbour model.


```r
preModel <- preProcess(train[, num_idx], method = c("knnImpute"))
ptraining <- cbind(train$classe, predict(preModel, train[, num_idx]))
ptesting <- cbind(test$classe, predict(preModel, test[, num_idx]))
```

Rename first Label to classe


```r
names(ptraining)[1] <- "classe"
names(ptesting)[1] <- "classe"
```


Make test set for submission


```r
prtesting <- predict(preModel, pml.test[, num_idx])
```

### Modeling with Random Forest Model


```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rf_model <- randomForest(classe ~ ., ptraining)
```


### In-sample accuracy

```r
training_pred <- predict(rf_model, ptraining)
print(confusionMatrix(training_pred, ptraining$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

The in-sample accuracy is 100%.

### Out-of-sample accuracy

```r
testing_pred <- predict(rf_model, ptesting)
```


Confusion Matrix: 

```r
print(confusionMatrix(testing_pred, ptesting$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1666   11    2    2    2
##          B    5 1116   13    2    2
##          C    1   10 1000    5    7
##          D    1    0    9  952    3
##          E    1    2    2    3 1068
## 
## Overall Statistics
##                                         
##                Accuracy : 0.986         
##                  95% CI : (0.983, 0.989)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.982         
##  Mcnemar's Test P-Value : 0.48          
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.995    0.980    0.975    0.988    0.987
## Specificity             0.996    0.995    0.995    0.997    0.998
## Pos Pred Value          0.990    0.981    0.978    0.987    0.993
## Neg Pred Value          0.998    0.995    0.995    0.998    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.283    0.190    0.170    0.162    0.181
## Detection Prevalence    0.286    0.193    0.174    0.164    0.183
## Balanced Accuracy       0.996    0.988    0.985    0.992    0.993
```


The cross validation accuracy is  98.6%, which should be sufficient for predicting the 20 test observations.

### Test Set Prediction Results

Applying model to the test data.

```r
answers <- predict(rf_model, prtesting)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


Writing outputs to 20 files.


```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
pml_write_files(answers)
```


## References
[1] http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises  
[2] https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
[3] https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
