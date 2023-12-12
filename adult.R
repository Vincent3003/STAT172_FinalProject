rm(list = ls())

# load package part
library(rpart)      # fitting classification trees
library(rpart.plot) # plotting classification trees
library(ggplot2)    # quality graphics
library(pROC)       # for creating ROC curves
library(wesanderson)   
library(RColorBrewer)
library(randomForest)
library(dplyr)
library(logistf)
library(glmnet) # for fitting lasso, ridge regressions (GLMs)
library(lubridate) #for easily manipulating dates


# 1. Read in the data and do some initial exploratory work to make sure the data was read in correctly and
#to see if any cleaning is required. (we’ll do this together)
adult <- read.csv("adult.csv", stringsAsFactors = TRUE)
summary(adult)    # look at a summary of the data 
str(adult)        # look at the structure of the data
colnames(adult)   # get the column names of the dataset

################ Clean data ###############
# rename the columns
colnames(adult) <- c("Age", "WorkingClass", "Final_Weight", "Education_Level", "Education_Number", 
                     "Marital_Status", "Occupation", "Relationship", "Race", "Sex", "Capital_Gain", 
                     "Capital_Loss", "Hrs_per_Week", "Native_Country", "Income")

# Change  ? to NA value
adult[adult == "?"] = NA

# Omit missing adult1 records
adult1 <- na.omit(adult)
head(adult1)

# Add one more column for Net Gain
adult <- mutate(adult, Net_Capital = Capital_Gain - Capital_Loss)

adult1 <- mutate(adult1, Net_Capital = Capital_Gain - Capital_Loss)

# create a binary variable for Income
# adult1 <- mutate(adult1, IncomeBin = case_when(Income == ">50K" ~ 1, Income == "<=50K" ~ 0))
# summary(adult1)

#################### DATA MANIPULATION ###################
# Do some grouping for better ploting

# Grouping for Education
adult1 <- mutate(adult1, EducationLevel_clean = Education_Level)
adult1 <- mutate(adult1, 
                EducationLevel_clean = case_when(
                  Education_Level %in% c("10th", "11th", "12th", "HS-grad") ~ "HighSchool",
                  Education_Level %in% c("Masters", "Doctorate", "Bachelors", "Prof-school") ~ "HigherEducation",
                  Education_Level %in% c("1st-4th", "Preschool", "5th-6th") ~ "Elementary",
                  Education_Level %in% c("Some-college", "Assoc-voc", "Assoc-acdm") ~ "Post-High_School",
                  TRUE ~ "MiddleSchool"
                )
)

# Grouping for Occupation
adult1 <- mutate(adult1, Occupation_clean = Occupation)
adult1 <- mutate(adult1, 
                 Occupation_clean = case_when(
                   Occupation %in% c("Exec-managerial", "Prof-specialty") ~ "Professional",
                   Occupation %in% c("Machine-op-inspct", "Craft-repair") ~ "SkilledLabor",
                   Occupation %in% c("Other-service", "Adm-clerical") ~ "Services",
                   Occupation %in% c("Transport-moving", "Handlers-cleaners") ~ "ManualLabor",
                   Occupation %in% c("Farming-fishing", "Tech-support") ~ "Specialized",
                   Occupation %in% c("Protective-serv", "Armed-Forces") ~ "Protective Services",
                   Occupation %in% c("Sales") ~ "Sales",
                   Occupation %in% c("Priv-house-serv") ~ "Domestic",
                   TRUE ~ "Others"
                 )
)

adult1 <- mutate(adult1, WorkingClass_clean = WorkingClass)
adult1 <- mutate(adult1, 
                 WorkingClass_clean = case_when(
                   WorkingClass %in% c("Federal-gov", "Local-gov", "State-gov") ~ "Governemnt",
                   WorkingClass %in% c("Self-emp-not-inc", "Self-emp-inc") ~ "SelfEmployed",
                   WorkingClass %in% c("Never-worked", "Without-pay", "NA") ~ "Others",
                   TRUE ~ "Private"
                 )
)

adult3 <- as.data.frame(unclass(adult1),                     # Convert all columns to factor
                      stringsAsFactors = TRUE)

# Drop columns in a datset
adult2 = subset(adult1, select = -c(Final_Weight, Education_Number, Relationship, Capital_Gain, 
                                    Capital_Loss, Education_Level, Occupation, WorkingClass,
                                    Native_Country)) 
adult2.2 = subset(adult3, select = -c(Final_Weight, Education_Number, Relationship, Capital_Gain, 
                                   Capital_Loss, Education_Level, Occupation, WorkingClass,
                                   Native_Country)) 
# adult2 has chr and int variables only but adult2.2 has only Factor and int

str(adult2)
summary(adult2)
head(adult2)
# Write CSV file into a folder
write.csv(adult2, "adult_1.csv", row.names=FALSE)

################# Visualization ##################
# A basic scatterplot with color depending on Species
ggplot(adult2, aes(x=Age, color=Sex)) + 
  geom_bar(size=6) +
  ggtitle("Different Working Ages between Female and Male")

ggplot(adult2, aes(x = WorkingClass_clean, fill = Income)) + 
  scale_color_brewer(palette="BrBG") +
  geom_bar(position = "fill") + coord_flip() +
  labs(y = "Proportion") + ggtitle("Income of different Working Class ")

ggplot(adult2) +  
  geom_histogram(aes(x = Age, fill = Income), alpha = 0.5, position = "identity") +
  ggtitle("Age Distribution by Income Level")

ggplot(adult2, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") + 
  labs(title = "Age Distribution", x = "Age", y = "Frequency")

ggplot(adult2, aes(x = EducationLevel_clean, fill = Income)) + geom_bar(position = "stack") +
  labs(title = "Income Distribution by Education Level", x = "Education Level", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(adult2, aes(x = Marital_Status, fill = Marital_Status)) + geom_bar() +
  labs(title = "Marital Status Distribution", x = "Marital Status", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(adult2, aes(x = Hrs_per_Week, fill = Income)) + 
  geom_density(position = "dodge") +
  facet_wrap(~ Race) +  
  scale_color_brewer(palette="BrBG") +
  ggtitle("Income of different Working Hours per Week from different race")


######################### DATA PREPARATION Methods ################
# need to 'set the seed'
# setting the seed will result in all of us getting same training/ testing data
RNGkind(sample.kind = "default")
set.seed(2291352)

# create a vector of id's that will be put in the training data
train.ids <- sample(x = 1: nrow(adult2.2), size = floor(0.8 * nrow(adult2.2)))
# create training data set
train.df <- adult2.2[train.ids, ]
# create testing data set
test.df <- adult2.2[-train.ids, ]

##########################  CLASSFICATION TREE FITTING / INTERPRETATION ########################## 
set.seed(172172172)   # for reproducibility
ctree_adult <- rpart(Income ~ ., 
                  data = train.df, 
                  method = 'class', control=rpart.control(cp=0.0001, minsplit=1))

rpart.plot(ctree_adult)   # print the tree graph
printcp(ctree_adult)      # print the tree word version
# This is NOT a pruned tree. This is an overgrown tree

# prunned the tree down
optimalcp <- ctree_adult$cptable[which.min(ctree_adult$cptable[,"xerror"]),"CP"]
ctree_adult$cptable[which.min(ctree_adult$cptable[,"xerror"]),"CP"]
# 'improved' tree where CP is optimized and ctree_adult2 is a new tree
ctree_adult2<-prune(ctree_adult, cp=optimalcp)
rpart.plot(ctree_adult2)

# create ROC curve
pi_hat <- predict(ctree_adult2, test.df, type = "prob")[,">50K"] #choose Y: positive
rocCurve <- roc(response = test.df$Income,#supply truth
                predictor = pi_hat,#supply predicted PROBABILITIES)
                levels = c("<=50K", ">50K")) #(negative, positive)
## Setting direction: controls < cases
plot(rocCurve,print.auc = TRUE, print.thres = TRUE)

pi_star <- coords(rocCurve, "best", ret = "threshold")$threshold[1]
test.df$result_prediction <- as.factor(ifelse(pi_hat > pi_star, ">50K", "<=50K"))

##########################  RANDOM FOREST ########################## 
# BASELINE FOREST #
# you do not have to fit a baseline forest before you tune 
# however its helpful because then you know how long it's going to take
mtry <- c(1:9)
#make room for B, OOB error
keeps <- data.frame(m = rep(NA,length(mtry)),
                    OOB_err_rate = rep(NA, length(mtry)))
for (idx in 1:length(mtry)){
  forest <- randomForest(Income ~ .,
                         ntree = 500,
                         mtry = mtry[idx],
                         data = train.df,
                         type = 'classification')
  #record how many trees we tried
  keeps[idx, "m"] <- mtry[idx]
  #record what our OOB error rate was
  keeps[idx,"OOB_err_rate"] <- mean(predict(forest)!= train.df$Income)
}
ggplot(data = keeps) +
  geom_line(aes(x = m, y = OOB_err_rate)) +
  theme_bw() + labs(x = "m (mtry) value", y = "OOB error rate") +
  scale_x_continuous(breaks=c(1:10))

#My results suggest an m of 2 would be ideal for minimizing OOB error
final_forest <- randomForest(Income ~ .,
                             ntree = 500,
                             mtry = 3,
                             data = train.df,
                             type = 'classification',
                             importance = TRUE)

pi_hat_forest<- predict(final_forest, test.df, type = "prob")[,">50K"]
rocCurve_forest <- roc(response = test.df$Income,#supply truth
                predictor = pi_hat_forest,#supply predicted PROBABILITIES)
                levels = c("<=50K", ">50K")) #(negative, positive)
plot(rocCurve_forest,print.auc = TRUE, print.thres = TRUE)

pi_star_forest <- coords(rocCurve_forest, "best", ret = "threshold")$threshold[1]
test.df$result_pred <- as.factor(ifelse(pi_hat_forest > pi_star_forest, ">50K", "<=50K"))

# Create a bar chart to predict which x variables are important to predict Income
vi <- as.data.frame(varImpPlot(final_forest, type = 1))
vi$Variable <- rownames(vi)
ggplot(data = vi) +
  geom_bar(aes(x = reorder(Variable,MeanDecreaseAccuracy), weight = MeanDecreaseAccuracy),
           position ="identity") +
  coord_flip() +
  labs( x = "Variable Name",y = "Importance")

# adult2 has chr and int variables only but adult2.2 has only Factor and int
adult2$Income_bin <- ifelse(adult2$Income == ">50K", 1, 0)
adult2.2$Income_bin <- ifelse(adult2.2$Income == ">50K", 1, 0)


m1 <- glm(Income_bin ~  Marital_Status , data = adult2, family = binomial)
AIC(m1)   # 27280.32
BIC(m1)   # 27338.52


m2 <- glm(Income_bin ~  Marital_Status + Net_Capital, data = adult2, family = binomial)
AIC(m2)   # 25190.53
BIC(m2)   # 27338.52

m3 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean, data = adult2, 
          family = binomial)
AIC(m3)   # 22319.88
BIC(m3)   # 22419.65

m4 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age, data = adult2, 
          family = binomial)
AIC(m4)   # 22035.06
BIC(m4)   # 22143.15

m5 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Occupation_clean, 
          data = adult2, 
          family = binomial)
AIC(m5)   # 21518.23
BIC(m5)   # 21684.52

m6 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Occupation_clean 
          + Hrs_per_Week, 
          data = adult2, 
          family = binomial)
AIC(m6)   # 21150.87
BIC(m6)   # 21325.47

m7 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Occupation_clean 
          + Hrs_per_Week + Sex, 
          data = adult2, 
          family = binomial)
AIC(m7)   # 21143.01
BIC(m7)   # 21325.92

m8 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Occupation_clean
          + Hrs_per_Week + Sex + WorkingClass_clean, 
          data = adult2, 
          family = binomial)
AIC(m8)   # 21101.64 
BIC(m8)   # 21309.5

m9 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Occupation_clean
          + Hrs_per_Week + Sex + WorkingClass_clean + Race, 
          data = adult2, 
          family = binomial)
AIC(m9)   # 21084.48 
BIC(m9)   # 21325.6

# Whoops, whoops m9 is the smallest AIC
summary(m9)

m_firth <- logistf(Income_bin ~ Marital_Status + Net_Capital + EducationLevel_clean + Age + 
                     Occupation_clean + Hrs_per_Week + Sex + WorkingClass_clean + Race,
                   data = adult2)
summary(m_firth) 
coef(m_firth)
exp(coef(m_firth))

####################### Data Mining ###############################
RNGkind(sample.kind = "default")
set.seed(23591)
train.idx <- sample(x = 1:nrow(adult2), size = floor(.7*nrow(adult2)))
train_data <- adult2[train.idx,]
test.data <- adult2[-train.idx,]

#Fit a traditional logistic regression fit with MLE
lr_mle <- glm(Income_bin ~ Marital_Status + Net_Capital + EducationLevel_clean + Age + 
                Occupation_clean + Hrs_per_Week + Sex + WorkingClass_clean + Race, 
              data = train_data, 
              family = binomial(link = "logit"))

#look at the coefficients on the logistic regression
coef(lr_mle)
exp(coef(lr_mle))
lr_mle %>% coef %>% exp


#build x matrix for the lasso regression
#"one-hot" codes any factors/characters in your data
x <- model.matrix(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + 
                    Occupation_clean + Hrs_per_Week + Sex + WorkingClass_clean + Race
                  , data = train_data)[,-1] 
# the -1 takes out the intercept

#create vector of 0/1 for y
y <- as.vector(train_data$Income_bin)
#alpha = 1 is going to fit a lasso
#alpha = 0 is going to fit a ridge (they shrink differently)

lr_lasso <- glmnet(x=x, y=y, family = binomial(link = logit), alpha = 1)
summary(lr_lasso) 

# the glmnet function fit 100 different logistic regression models;
# one for each different "lambda" value.
# each model has a different coefficient vector, that varies
# because of how big lambda is

#our task: pick the lambda that maximizes out of sample predictive accuracy

lr_lasso$lambda[1] #first lambda tried
lr_lasso$beta[,1] #for that value of lambda, here is the corresponding beta vector
#for a lambda of 0.01, all the coefficients are 0

lr_lasso$lambda[60] #60th value of lambda tried
lr_lasso$beta[,60] #for that value of lambda, here is the corresponding beta vector
#way more numbers! now lambda is 0.0007 (so smaller) and coefficients could grow
#note: as lambda ---> 0, the lasso regression will approach MLE that we fit first.

plot(lr_lasso, xvar="lambda", label = TRUE)

#what have we done/accomplished at this point?
#nothing really. 
#we need to use cross validation measures to tune
#lambda. we will choose the lambda that minimizes
#out of sample error.


lr_lasso_cv = cv.glmnet(x,y, family = binomial(link = "logit"), alpha = 1)

plot(lr_lasso_cv)

lr_lasso_cv$lambda.min #the lambda that minimizes CV error
lr_lasso_cv$lambda.1se # will be a larger penalty (more 0 coefficients)

#see the coefficients for the model that minimizes OOE
lr_lasso_coefs<- coef(lr_lasso_cv, s="lambda.min") %>% as.matrix()
lr_lasso_coefs


#Quantify model performance for the lasso and MLE logistic regression models. Make ROC for both. 
# Choose best model.

#model.matrix code not necessary to make MLE predictions,
#but it is necessary for the lasso
x_test <- model.matrix(Income_bin ~ Marital_Status + Net_Capital + EducationLevel_clean + Age + 
                         Occupation_clean + Hrs_per_Week + Sex + WorkingClass_clean + Race,
                        data = test.data)[,-1]

#you'll see type = "response" on both predict functions
#this makes sure predict() gives us probabilities instead of log odds/ linear predictor scale

test.data <- test.data %>%
  mutate(mle_pred = predict(lr_mle, test.data, type = "response"),
         lasso_pred = predict(lr_lasso_cv, x_test, s = "lambda.min", type = "response")[,1])


cor(test.data$mle_pred, test.data$lasso_pred)
#the predictions from the two logistic regerssion models 
#(MLE vs lasso) are positively correlated but certainly
#not the same

test.data %>%
  ggplot() +
  geom_point(aes(x = mle_pred, y = lasso_pred)) +
  geom_abline(aes(intercept = 0, slope = 1))

#so we know they are different. but which is better?

mle_rocCurve <- roc(response = as.factor(test.data$Income_bin),#supply truth
                      predictor = test.data$mle_pred,#supply predicted PROBABILITIES
                      levels = c("0", "1") #(negative, positive)
)

plot(mle_rocCurve, print.thres = TRUE,print.auc = TRUE)
#AUC = 0.696

lasso_rocCurve   <- roc(response = as.factor(test.data$Income_bin),#supply truth
                        predictor = test.data$lasso_pred,#supply predicted PROBABILITIES
                        levels = c("0", "1") #(negative, positive)
)

plot(lasso_rocCurve, print.thres = TRUE,print.auc = TRUE)
#AUC = 0.774

str(lasso_rocCurve)

ggplot() +
  geom_line(aes(x = 1-mle_rocCurve$specificities, y = mle_rocCurve$sensitivities), colour = "darkorange1") +
  geom_text(aes(x = .75, y = .75, 
                label = paste0("MLE AUC = ",round(mle_rocCurve$auc, 3))), colour = "darkorange1")+
  geom_line(aes(x = 1-lasso_rocCurve$specificities, y = lasso_rocCurve$sensitivities), colour = "cornflowerblue")+
  geom_text(aes(x = .75, y = .65, 
                label = paste0("Lasso AUC = ",round(lasso_rocCurve$auc, 3))), colour = "cornflowerblue") +
  labs(x = "1-Specificity", y = "Sensitivity")

