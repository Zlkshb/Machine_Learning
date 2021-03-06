# Multiple Linear Regression

# add dataset into R workspace
dataset = read.csv('50_Startups.csv')

# Encoding categorical the dataset
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))

# Splitting dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(formula = Profit ~ .,
               data = training_set)

y_pred1 = predict(regressor, newdata = test_set)

regressor = lm(formula = Profit ~ R.ACY.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)


regressor = lm(formula = Profit ~ R.ACY.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)


regressor = lm(formula = Profit ~ R.ACY.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)


regressor = lm(formula = Profit ~ R.ACY.D.Spend,
               data = dataset)
summary(regressor)
# since the most significance value only R.D.Spent then, equation
# can be changed into

regressor = lm(formula = Profit ~ R.ACY.D.Spend,
               data = training_set)

y_pred2 = predict(regressor, newdata = test_set)

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
