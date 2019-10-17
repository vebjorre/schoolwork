#Small project done in TMA4268 Statistical learning
#Using different statistical methods to determine whether women have diabetes or not

#Load packages used in project:
library(tidyverse)
library(caret)
library(pROC)
library(ggplot2)
library(GGally)
library(corrplot)
library(MASS)
library(e1071)
library(tree)
library(nnet)
library(NeuralNetTools)

#Import data set
flying=dget("https://www.math.ntnu.no/emner/TMA4268/2019v/data/flying.dd")
ctrain=flying$ctrain
ctest=flying$ctest
#Summary of training data:
summary(ctrain)


#ctrain.tmp<-ctrain
#ctrain.tmp$diabetes<-as.factor(ctrain.tmp$diabetes)
#ctrain.tmp$npreg<-as.factor(ctrain.tmp$npreg)
#ggpairs(ctrain.tmp,cardinality_threshold = 16)
ggpairs(ctrain)
corrplot(cor(ctrain))


#Scale and center data
mean <- apply(ctrain[,-1], 2, mean) #not diabetes, in column 1                                 
std <- apply(ctrain[,-1], 2, sd)
train.x <- data.frame(scale(ctrain[,-1], center = mean, scale = std))
test.x <- data.frame(scale(ctest[,-1], center = mean, scale = std)) # normalized based on the whole training set

train.y <- as.factor(ctrain[,1])
test.y <- as.factor(ctest[,1])
train.xy <- data.frame(train.x,diabetes=train.y)
test.xy <- data.frame(test.x,diabetes=test.y)

resmat=matrix(ncol=2,nrow=8)
# misclassification rate test data, AUC
# for all the methods considered here
colnames(resmat)=c("Misclassification rate","AUC")
rownames(resmat)=c("KNN","LDA","logistic regression","classification tree","pruned tree","SVC","SVM","NN")

#===============================================================================================
#Need factors to have valid var.names for trainControl()
knn.train.y <- as.factor(ctrain[,1]) 
knn.test.y <- as.factor(ctest[,1])
knn.train.y <- factor(knn.train.y, labels=c("O","I")) 
knn.test.y <- factor(knn.test.y, labels=c("O", "I"))
knn.train.xy <- data.frame(train.x,diabetes=knn.train.y)
knn.test.xy <- data.frame(test.x,diabetes=knn.test.y)

#K nearest neighbours classification:

set.seed(0)
#CV to find best K
train.control <- trainControl(method = "cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary)
#train knn model
knn.mod <- train(diabetes ~ ., data=knn.train.xy, "knn", trControl = train.control,tuneLength=50)
print(knn.mod)
#test with test data
knn.res <- predict(knn.mod,newdata=test.x)
knn.prob <- predict(knn.mod,newdata=test.x, type="prob")
knn.conf <- confusionMatrix(knn.res,knn.test.y)$table
knn.missrate <- 1 - sum(diag(knn.conf))/sum(knn.conf)
#plot ROC
knn.roc <- roc(knn.test.y, knn.prob[,2])
ggroc(knn.roc)+ggtitle("ROC curve - K-Nearest-Neighbours") + annotate("text", x = 0.25, y = 0.3, label = paste("AUC = ", as.character(round(auc(knn.roc), 4))))

resmat[1,1] = knn.missrate
resmat[1,2] = knn.roc$auc
#===============================================================================================

#train lda model
lda.mod <- lda(diabetes~npreg+glu+bp+skin+bmi+ped+age, data=train.xy,prior=c(0.5,0.5))
lda.res <- predict(lda.mod, newdata = test.x)
lda.conf <- confusionMatrix(lda.res$class,test.y)$table
lda.missrate <- 1 - sum(diag(lda.conf))/sum(lda.conf)

lda.roc = roc(test.y,lda.res$posterior[,2],legacy.axes=TRUE,auc=TRUE)
ggroc(lda.roc)+ggtitle("ROC curve - Linear discriminant analysis") + annotate("text", x = 0.25, y = 0.3, label = paste("AUC = ", as.character(round(auc(lda.roc), 4))))

resmat[2,1] = lda.missrate
resmat[2,2] = lda.roc$auc
#===============================================================================================

#train logistic regression model
logist.mod=glm(diabetes~., data=train.xy,family="binomial")
summary(logist.mod)
logist.prob <- predict(logist.mod, newdata = test.x,type="response")
logist.res <- factor(ifelse(logist.prob>=0.5,1,0))
logist.conf <- confusionMatrix(test.y,logist.res)$table
logist.missrate <- 1 - sum(diag(logist.conf))/sum(logist.conf)

# logist.roc = roc(test.y,logist.res,legacy.axes=TRUE)
logist.roc <- roc(test.y,logist.prob)
ggroc(logist.roc)+ggtitle("ROC curve - Logistic regression") + annotate("text", x = 0.25, y = 0.3, label = paste("AUC = ", as.character(round(auc(logist.roc), 4))))

resmat[3,1] = logist.missrate
resmat[3,2] = logist.roc$auc
#===============================================================================================

#Classification tree:

set.seed(2)
tree.mod <- tree(diabetes ~ ., train.xy, split="gini")

plot(tree.mod)
text(tree.mod, pretty=1, cex=.8)

summary(tree.mod)

tree.res <- predict(tree.mod, newdata = test.x, type="class")
tree.conf <- confusionMatrix(tree.res, test.y)$table
print(tree.conf)
#Misclassification rate:
tree.missrate <- 1 - sum(diag(tree.conf))/sum(tree.conf)

resmat[4,1] = tree.missrate
#===============================================================================================

#CV to find best number of splits in tree
#Not completely correct to use cv on already scaled data
tree.cv <- cv.tree(tree.mod, FUN = prune.misclass)
plot(tree.cv$size, tree.cv$dev, type = "b")

#Prune tree down to 10 splits
pruned.mod <- prune.tree(tree.mod, best = 10)
summary(pruned.mod)
plot(pruned.mod)
text(pruned.mod, pretty = 1, cex=0.8)

pruned.res <- predict(pruned.mod, newdata=test.x, type="class")
pruned.conf <- confusionMatrix(pruned.res, test.y)$table

#Misclassification rate:
pruned.missrate <- 1 - sum(diag(pruned.conf))/sum(pruned.conf)

resmat[5,1] = pruned.missrate
#===============================================================================================

#Support vector classifier:

set.seed(1)
svc.cv <- tune(svm, diabetes ~., data=train.xy, 
                 kernel="linear", type="C-classification",
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100), gamma=c(0.01,1,5,10)))
svc.mod <- svc.cv$best.model

svc.res <- predict(svc.mod, test.x, type="class")
svc.conf <- confusionMatrix(svc.res, test.y)$table
svc.missrate <- 1 - sum(diag(svc.conf))/sum(svc.conf)

svm.cv <- tune(svm, as.factor(diabetes) ~ ., data=train.xy, 
                 kernel="radial", type="C-classification",
                 ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100),gamma=c(0.01,1,5,10)))
summary(svm.cv)
svm.mod <- svm.cv$best.model

svm.res <- predict(svm.mod, test.x, type="class")
svm.conf <- confusionMatrix(svm.res,test.y)$table
svm.missrate <- 1 - sum(diag(svm.conf))/sum(svm.conf)

resmat[6,1] = svc.missrate
resmat[7,1] = svm.missrate

#===============================================================================================

#Feed forward neural network with one hidden layer
#No hidden layers are better in this situation -> logistic regression
#nnet allows only zero or one hidden layer

#Unscaled data:
train.data <- ctrain[,-1]
train.target <- as.factor(ctrain[,1])
test.data <- ctest[,-1]
test.target <- as.factor(ctest[,1])

#CV to decide number of nodes in hidden layer:

set.seed(0)
num.nodes <- c(0:10,15,20,25,30,50)
# num.nodes <- c(0:20)
k <- 5
indices <- sample(1:nrow(train.data))
folds <- cut(indices, breaks = k, labels=FALSE)
mismat <- matrix(NA, ncol=k, nrow=length(num.nodes))
for (j in 1:k){
  thistrain <- (1:dim(train.data)[1])[folds != j]
  thisvalid <- (1:dim(train.data)[1])[folds == j]
  mean <- apply(train.data[thistrain, ], 2, mean)
  std <- apply(train.data[thistrain, ], 2, sd)
  new = scale(train.data, center=mean, scale=std)
  for (i in 1:length(num.nodes)) {
    thissize = num.nodes[i]
    
    fit = nnet(formula=train.target ~ npreg+glu+bp+skin+bmi+ped+age, data=new, size=thissize, maxit=100, skip=TRUE)
    pred = predict(fit, newdata=new[thisvalid, ], type="class")
    misclass = table(pred, new[thisvalid, 1])
    miss_rate = sum(diag(misclass))/sum(misclass)
    mismat[i,j] = miss_rate
  }
}


average.misclass <- apply(mismat, 1, sum)/k
plot.window(xlim=c(0,50),ylim=c(0,100))
plot(num.nodes,average.misclass, type="l")

nnet.mod <- nnet(formula=diabetes~npreg+glu+bp+skin+bmi+ped+age, data=train.xy, size=0, linout=FALSE, skip=TRUE, maxit=5000)

nnet.res <- predict(nnet.mod, newdata=test.x, type="class")
nnet.conf <- confusionMatrix(factor(nnet.res), test.y)$table
nnet.missrate = 1 - sum(diag(nnet.conf))/sum(nnet.conf)

nnet.prob <- predict(nnet.mod, newdata=test.x)
nnet.roc <- roc(test.y, as.numeric(nnet.prob), legacy.axes=TRUE)
ggroc(nnet.roc)+ggtitle("ROC curve - Neural network") + annotate("text", x = 0.25, y = 0.3, label = paste("AUC = ", as.character(round(auc(nnet.roc), 4))))

resmat[8,1] = nnet.missrate
resmat[8,2] = nnet.roc$auc

print(resmat)

plot(knn.roc)
plot(lda.roc,add=TRUE)
plot(logist.roc,add=TRUE)
plot(nnet.roc,add=TRUE)
