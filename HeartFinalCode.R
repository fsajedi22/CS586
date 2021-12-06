
# Francesca Sajedi CS586 - CKD Data set
# After looking into each data set more - 
# I decided to focus my time on CKD rather than touch on each data set
#libraries
library(ggplot2)
library(dplyr)
library(readr)
library(corrplot)
library(caret)
library(pbkrtest)
library(ROCR)
library(tree)
library(randomForest)
library(rstanarm)
library(pROC)
library(tidyr)
library(rpart)
library(rpart.plot)

#Data set - change path to where you saved data
heart <- read.csv("/Users/fsajedi/Desktop/heart.csv", 
                  header = FALSE, 
                  sep = ",")
# Manipulating the data
colnames(heart) <- c("Age","Gender","CP","TBps",
                     "Chol","Fbs","Recg","Thalach","Exang","Op",
                     "Slope","Ca","Thal","Heart")

heart[heart == "?"] <- NA
str(heart)
heart$CP=as.numeric(as.character(heart$CP))
heart$TBps=as.numeric(as.character(heart$TBps))
heart$Chol=as.numeric(as.character(heart$Chol))
heart$Fbs=as.numeric(as.character(heart$Fbs))
heart$Recg=as.numeric(as.character(heart$Recg))
heart$Thalach=as.numeric(as.character(heart$Thalach))
heart$Exang=as.numeric(as.character(heart$Exang))
heart$Op=as.numeric(as.character(heart$Op))
heart$Slope=as.numeric(as.character(heart$Slope))
heart$Ca=as.numeric(as.character(heart$Ca))
heart$Thal=as.numeric(as.character(heart$Thal))
heart$Heart=as.numeric(as.character(heart$Heart))

#Cleaning the data
heart$Age[which(is.na(heart$Age))]= mean(heart$Age, na.rm = TRUE)
heart$Gender[which(is.na(heart$Gender))]= mean(heart$Gender, na.rm = TRUE)
heart$CP[which(is.na(heart$CP))]= mean(heart$CP, na.rm = TRUE)
heart$TBps[which(is.na(heart$TBps))]= mean(heart$TBps, na.rm = TRUE)
heart$Chol[which(is.na(heart$Chol))]= mean(heart$Chol, na.rm = TRUE)
heart$Fbs[which(is.na(heart$Fbs))]= mean(heart$Fbs, na.rm = TRUE)
heart$Recg[which(is.na(heart$Recg))]= mean(heart$Recg, na.rm = TRUE)
heart$Thalach[which(is.na(heart$Thalach))]= mean(heart$Thalach, na.rm = TRUE)
heart$Exang[which(is.na(heart$Exang))]= mean(heart$Exang, na.rm = TRUE)
heart$Op[which(is.na(heart$Op))]= mean(heart$Op, na.rm = TRUE)
heart$Slope[which(is.na(heart$Slope))]= mean(heart$Slope, na.rm = TRUE)
heart$Ca[which(is.na(heart$Ca))]= mean(heart$Ca, na.rm = TRUE)
heart$Thal[which(is.na(heart$Thal))]= mean(heart$Thal, na.rm = TRUE)
heart$Heart[which(is.na(heart$Heart))]= mean(heart$Heart, na.rm = TRUE)

#Summarizing Data
summary(heart)

boxplot(heart, main = "Range of Values for all Parameters")
str(heart)
# Transforming data and graphing
heart$Heart[heart$Heart == "2"] <- "1"
heart$Heart[heart$Heart == "3"] <- "1"
heart$Heart[heart$Heart == "4"] <- "1"
heart$Heart=as.numeric(as.character(heart$Heart))


preprocessParamsh <- preProcess(heart, method=c("scale"))
print(preprocessParamsh)
transformedh <- predict(preprocessParamsh, heart)

boxplot(transformedh, main = "Range of Values for all Parameters")
preprocessParamsh2 <- preProcess(heart, method=c("center"))
print(preprocessParamsh2)
transformedh2 <- predict(preprocessParamsh2,heart)
boxplot(transformedh2, main = "Range of Values for all Parameters 2")

preprocessParamsh3 <- preProcess(heart, method=c("center", "scale"))
print(preprocessParamsh3)
transformedh3 <- predict(preprocessParamsh,heart)
boxplot(transformedh3, main = "Range of Values for all Parameters 3")

preprocessParamsh4 <- preProcess(heart, method=c("range"))
print(preprocessParamsh4)
transformedh4 <- predict(preprocessParamsh, heart)
boxplot(transformedh4, main = "Range of Values for all Parameters",sub = "Scaled based on Range")
summary(transformedh4)
str(heart)
heart = transformedh4
########################################################################
########################################################################

cp_meanh <- heart %>% group_by(Heart) %>% summarise(Plas = round(mean(CP),2))


heart$Heart[heart$Heart > 0] <- 1
fate =barplot(table(heart$Heart),
              main="Heart Disease Yes (1), No (0)", sub = "Outcome of Patient Data",col="blue")

fate

ins_meanh <- heart %>% group_by(Heart) %>% summarise(Plas = round(mean(Chol),2))

#Relationship between Heart & Chol levels
Chol_ = ggplot(data=heart,aes(Heart,Chol)) +
  geom_boxplot(aes(fill=Heart) ,col = "blue") + stat_boxplot(geom = "errorbar", col = "blue") + 
  ggtitle("Would you use Cholesterol as a parameter?: Affect Cholesterol has on Heart Disease") + 
  xlab("Heart") + ylab("Chol") + guides(fill=F) + 
  geom_text(data = ins_meanh, aes(x=Heart,y=Plas,label=Plas),
            hjust = -1.5,vjust=-0.5) 


Chol_

#ALL
gather(heart, x, y, Age:Thal) %>%
  ggplot(aes(x = y, color = Heart, fill = Heart)) +
  geom_density(alpha = 0.3) +
  facet_wrap( ~ x, scales = "free", ncol = 3)

#Thalach and Op
t_op= ggplot(heart, aes(x = Thalach, y = Op, color = Heart)) +
  geom_point() +
  ylab("Thalach") +
  xlab("OP") +
  ggtitle("Relationship") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

t_op

#Relationship between Age and Heart
age_= ggplot(heart, aes(Age, fill = Heart)) +
  geom_density(, col = "blue") + ylab("Heart Risk") + 
  ggtitle("Age vs.Threat of Heart Disease")

age_


str(heart)
corrplot(cor(heart[,-9]),type = "lower", method = "color", main  = "
         How do the parameters Correlate? ")

#Creating the Training and Test set
set.seed(15689)
indexh <- createDataPartition(heart$Heart,p = 0.7,list = F)
trainh <- heart[indexh,]
testh  <- heart[-indexh,]

########################################################################
########################################################################
#Logistic Regression


m1h <- glm(Heart ~ ., data = trainh, family = binomial(link = "logit"))
summary(m1h)

anova(m1h,testh = "Chisq")
mod_finh <- glm(Heart ~ Age + CP + TBps + Chol + Fbs + Recg + Thalach + Exang +Op + 
                  Slope + Ca + Thal,
                data = trainh, family = binomial(link = "logit"))
summary(mod_finh)

summary(residuals(mod_finh))
par(mfrow=c(2,2))
plot(mod_finh)

testh_pred <- predict(mod_finh,testh, type = "response")
pred_testhh <- as.data.frame(cbind(testh$Heart,testh_pred))
colnames(pred_testhh) <- c("Original","testh_pred")
pred_testhh$outcome <- ifelse(pred_testhh$testh_pred > 0.5, 1, 0)
error <- mean(pred_testhh$outcome != testh$Heart)
print(paste('test Data Accuracy', round(1-error,2)*100,'%'))
testData = as.factor(testh$Heart)
outcomeVal = as.factor(pred_testhh$outcome)
confusionMatrix(testData, outcomeVal)
acc_lgh <- confusionMatrix(testData,outcomeVal)$overall['Accuracy']
par(mfrow=c(1,1))
plot.roc(testh$Heart,testh_pred,percent=TRUE,col="#1c61b6",print.auc=TRUE,
         main = "Area under the curve for Logistic Regression")


########################################################################
########################################################################

#Bayesian Logistic Regression
prior_disth <- student_t(df = 7, location = 0, scale = 2.5)
bayes_modh  <- stan_glm(Heart ~ ., data = trainh,
                        family = binomial(link = "logit"), 
                        prior = prior_disth, prior_intercept = prior_disth,
                        seed = 15689)


posterior_interval(bayes_modh, prob = 0.95)

summary(residuals(bayes_modh))

bayes_resh <- data.frame(residuals(bayes_modh))
bayes_resh$indexh <- seq.int(nrow(bayes_resh)) 


pred <- posterior_linpred(bayes_modh, newdata = testh, transform=TRUE)
pred
fin_predh <- colMeans(pred)
testh_prediction <- as.integer(fin_predh >= 0.5)

outcomeVal2 = as.factor(testh_prediction)
confusionMatrix(testData, outcomeVal2)

acc_bayesh <- confusionMatrix(testData,outcomeVal2)$overall['Accuracy']
acc_bayesh
plot.roc(testh$Heart,fin_predh,percent=TRUE,col="#1c61b6", print.auc=TRUE,
         main = "Area under the curve for Bayesian Logistic Regression")


########################################################################
########################################################################
#Decision Tress

set.seed(42)
fit <- rpart(Heart ~ .,
             data = trainh,
             method = "class",
             control = rpart.control(xval = 10, 
                                     minbucket = 2, 
                                     cp = 0), 
             parms = list(split = "information"))

rpart.plot(fit, extra = 100)
pred_dt_testh <- predict(fit, testh, type = "class")

outcomeVal3 = as.factor(pred_dt_testh)
confusionMatrix(testData, outcomeVal3)





set.seed(15689)
m_dth <- tree(Heart ~ ., data = trainh)
pred_dth <- predict(m_dth, trainh)
confusionMatrix(trainh$Heart,pred_dth)
plot(m_dth)
text(m_dth, pretty = 0)


acc_dth <- confusionMatrix(outcomeVal3,testData)$overall['Accuracy']
acc_dth

########################################################################
########################################################################
#Random Forest
heart$Heart <- as.factor(heart$Heart)
trainh$Heart <- as.factor(trainh$Heart)
testh$Heart <- as.factor(testh$Heart)
str(heart)
str(trainh)

set.seed(42)
model_rfh <- caret::train(Heart ~ .,
                          data = trainh,
                          method = "rf",
                          preProcess = c("scale", "center"),
                          trControl = trainControl(method = "repeatedcv", 
                                                   number = 10, 
                                                   repeats = 10, 
                                                   savePredictions = TRUE, 
                                                   verboseIter = FALSE))
model_rfh$finalModel$confusion
imph <- model_rfh$finalModel$importance

imph[order(imph, decreasing = TRUE), ]
importanceh <- varImp(model_rfh, scale = TRUE)
plot(importanceh, main = "Importance of Parameters")
confusionMatrix(predict(model_rfh, testh), testh$Heart)
acc_rfh <- confusionMatrix(predict(model_rfh, testh), testh$Heart)$overall['Accuracy']
acc_rfh

########################################################################
########################################################################
#Extreme gradient boosting


model_xgbh <- caret::train(Heart ~ .,
                           data = trainh,
                           method = "xgbTree",
                           preProcess = c("scale", "center"),
                           trControl = trainControl(method = "repeatedcv", 
                                                    number = 10, 
                                                    repeats = 10, 
                                                    savePredictions = TRUE, 
                                                    verboseIter = FALSE))
importance <- varImp(model_xgbh, scale = TRUE)
plot(importance, main = "Importance of Parameters")

confusionMatrix(predict(model_xgbh, testh), testh$Heart)
acc_xbgh <- confusionMatrix(predict(model_xgbh, testh), testh$Heart)$overall['Accuracy']


########################################################################
########################################################################

#Overall Accuracies 

accuracyh <- data.frame(Model=c("Logistic","Bayesian Logistic","Decision Tree",
                                "Random Forest","Boosting"),
                        Accuracy=c(acc_lgh*100,acc_bayesh*100,acc_dth*100,
                                   acc_rfh*100,acc_xbgh*100))

acc_heart= ggplot(accuracyh,aes(x=Model,y=Accuracy))+geom_bar(stat='identity', color ="blue", fill = "blue")+
  ggtitle('Comparison of Model Accuracy')

acc_heart
