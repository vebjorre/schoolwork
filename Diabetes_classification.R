#Small project done in TMA4268 Statistical learning
#Using different statistical methods to determine whether women have diabetes or not

#Importing data set
flying=dget("https://www.math.ntnu.no/emner/TMA4268/2019v/data/flying.dd") 
ctrain=flying$ctrain
ctest=flying$ctest
# transform(ctrain, diabetes = as.factor(diabetes))
# transform(ctest, diabetes = as.factor(diabetes))

#Summary of training data:
summary(ctrain)

library(tidyverse)
library(caret)
library(pROC)

#===============================================================================================

#K nearest neighbours classification:

set.seed(0)
#CV to find best K
train.control = trainControl(method = "cv", number=5)
model = train(ctrain[,-1], as.factor(ctrain[,1]), "knn", trControl = train.control)
# model = train(diabetes ~., data=ctrain, method="knn", trControl=train.control)
print(model)
#Training model with K=5
KNNclass = class::knn(train=ctrain, cl=ctrain$diabetes, test=ctest, k=5, prob=TRUE)

KNNprobdiabetes = attributes(KNNclass)$prob
KNNprob = ifelse(KNNclass == "0", 1-KNNprobdiabetes, KNNprobdiabetes)

misclass = table(KNNclass, ctest$diabetes)
#misclassification rate:
1 - sum(diag(misclass))/sum(misclass)

KNN.roc = roc(ctest$diabetes, KNNprob)
KNN.auc = auc(ctest$diabetes, KNNprob)

ggroc(KNN.roc) + ggtitle("ROC curve") + annotate("text", x = 0.25, y = 0.3, 
            label = paste("AUC = ", as.character(round(KNN.auc, 4))))

#===============================================================================================

#Decision tree:
library(tree)
set.seed(2)
tree.dia = tree(formula = as.factor(diabetes) ~ ., ctrain, split="gini")

plot(tree.dia)
text(tree.dia, pretty=1, cex=.8)

summary(tree.dia)

yhat.tree = predict(tree.dia, ctest, type = "class")
response.test = ctest$diabetes
misclass = table(yhat.tree, response.test)
print(misclass)
#Misclassification rate:
1 - sum(diag(misclass))/sum(misclass)

#CV to find best number of splits in tree
cv.dia = cv.tree(tree.dia, FUN = prune.misclass)
plot(cv.dia$size, cv.dia$dev, type = "b")

#Prune tree down to 10 splits
prune.dia = prune.misclass(tree.dia, best = 10)
summary(prune.dia)
plot(prune.dia)
text(prune.dia, pretty = 1, cex=0.8)

yhat.prune = predict(prune.dia, ctest, type="class")
prob.prune = prune.dia$yprob
misclass.prune = table(yhat.prune, response.test)
print(misclass.prune)
#Misclassification rate:
1 - sum(diag(misclass.prune))/sum(misclass.prune)

prob.prune = predict(prune.dia, ctest) 
prune.roc = roc(ctest$diabetes, prob.prune[,1], auc=TRUE) 
prune.auc = prune.roc$auc

ggroc(prune.roc) + ggtitle("ROC curve") + annotate("text", x = 0.25, y = 0.3, 
                                                   label = paste("AUC = ", as.character(round(prune.auc, 4))))

#===============================================================================================

#Support vector machine:
#NB! Should have scaled the data here.
library(e1071)

svmfit_linear = svm(diabetes ~ ., data = ctrain, kernel = "linear", cost = 1, 
                    scale = FALSE, type="C-classification")
summary(svmfit_linear)

set.seed(1)
cv_linear = tune(svm, as.factor(diabetes) ~., data=ctrain, 
                 kernel="linear", type="C-classification",
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(cv_linear)
bestmod_linear = cv_linear$best.model

yhat.linear = predict(bestmod_linear, ctest, type="class")
table(predict = yhat.linear, truth=ctest$diabetes)

set.seed(1)
cv_radial = tune(svm, as.factor(diabetes) ~ npreg + glu + bmi + ped, data=ctrain, 
                 kernel="radial", type="C-classification",
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(cv_radial)
bestmod_radial = cv_radial$best.model

yhat.radial = predict(bestmod_radial, ctest, type="class")
table(predict = yhat.radial, truth=ctest$diabetes)

#===============================================================================================

#Feed forward neural network with one hidden layer
#(Not a deliberate decision)

library(nnet)
library(NeuralNetTools)
set.seed(0)

train.data = ctrain[,-1]
train.target = as.factor(ctrain[,1])
test.data = ctest[,-1]
test.target = as.factor(ctest[,1])

#CV to decide number of nodes in hidden layer (less than 5 could be better):
grid = c(5,10,15,20,25,30,50)
k = 5
indices = sample(1:nrow(train.data))
folds = cut(indices, breaks = k, labels=FALSE)
mismat = matrix(NA, ncol=k, nrow=length(grid))
for (j in 1:k){
  thistrain = (1:dim(train.data)[1])[folds != j]
  thisvalid = (1:dim(train.data)[1])[folds == j]
  mean = apply(train.data[thistrain, ], 2, mean)
  std = apply(train.data[thistrain, ], 2, sd)
  new = scale(train.data, center=mean, scale=std)
  for (i in 1:length(grid)) {
    thissize = grid[i]
    
    fit = nnet(train.target ~., data=new, size=thissize, maxit=5000)
    pred = predict(fit, newdata=new[thisvalid, ], type="class")
    misclass = table(pred, new[thisvalid, 1])
    miss_rate = sum(diag(misclass))/sum(misclass)
    mismat[i,j] = miss_rate
  }
}


average.misclass = apply(mismat, 1, sum)/k
plot.window(xlim=c(0,50),ylim=c(0,100))
plot(grid,average.misclass, type="l")

set.seed(0)
mean = apply(train.data, 2, mean)
std = apply(train.data, 2, sd)
train.new = scale(train.data, center=mean, scale=std)
fit5 = nnet(ctrain[,1] ~., data=train.new, size=5, maxit=5000, linout=FALSE, entropy=TRUE)
plotnet(fit5)

test.new = scale(test.data, center=mean, scale=std)

pred = predict(fit5, newdata=test.new, type="class")
misclass.5 = table(pred, ctest[,1])

misclass.5
miss_rate = 1 - sum(diag(misclass.5))/sum(misclass.5)
miss_rate
