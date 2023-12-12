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
summary(adult)    # look at a summary of the data 
str(adult)        # look at the structure of the data
colnames(adult)   # get the column names of the dataset

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

# Drop columns in a datset
adult2 = subset(adult1, select = -c(Final_Weight, Education_Number, Relationship, Capital_Gain, 
                                    Capital_Loss, Education_Level, Occupation, WorkingClass,
                                    Native_Country)) 
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
train.ids <- sample(x = 1: nrow(adult2), size = floor(0.8 * nrow(adult2)))
# create training data set
train.df <- adult2[train.ids, ]
# create testing data set
test.df <- adult2[-train.ids, ]

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
test.df$result_pred <- as.factor(ifelse(pi_hat > pi_star, ">50K", "<=50K"))

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
                             mtry = 2,
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


adult2$Income_bin <- ifelse(adult2$Income == ">50K", 1, 0)

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

m5 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Sex, data = adult2, 
          family = binomial)
AIC(m5)   # 21996.68
BIC(m5)   # 22113.08

m6 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Sex + Hrs_per_Week, 
          data = adult2, 
          family = binomial)
AIC(m6)   # 21587.98
BIC(m6)   # 21712.7

m7 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Sex + Hrs_per_Week 
          + WorkingClass_clean, 
          data = adult2, 
          family = binomial)
AIC(m7)   # 21559.17
BIC(m7)   # 21708.82

m8 <- glm(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Sex + Hrs_per_Week +
            WorkingClass_clean + Race, 
          data = adult2, 
          family = binomial)
AIC(m8)   # 21534.11 
BIC(m8)   # 21717.02
# Whoops, whoops m8 is the smallest AIC
summary(m8)

m_firth <- logistf(Income_bin ~  Marital_Status + Net_Capital + EducationLevel_clean + Age + Sex + Hrs_per_Week +
                     WorkingClass_clean + Race, family = binomial,
                   data = adult2)
summary(m_firth) 
coef(m_firth)
exp(coef(m_firth))




