/* import the dataset from https://www.kaggle.com/datasets/sidhus/crab-age-prediction */
proc import out = adult
datafile = "/home/u49582076/sasuser.v94/STAT172_FinalProject/adult_1.csv"
dbms = csv replace;
/* guessingrows = max; */
run;

/* This project targets what factors can affect a consumer's income, 
	especially for who earns more than $50k becasue we want to provide 
	a more effective and personalized skincare routine */

/* this code is showing the frequency of human income depends on their age */
proc freq data = adult;
tables Income*Marital_Status;
run;

/* this code is showing the frequency of human education level depends on their age */
proc freq data = adult;
tables Income*EducationLevel_clean;
run;

proc sgplot data = adult;
scatter x = Hrs_per_Week y = Income ;
run;
proc sgplot data = adult;
histogram Hrs_per_Week;
run;

/*************************** Clean the dataset before cleaning it in R**********************************************************/
/* Plan: transform things like Education_Level or some other x variable into a good category. */
data adult;
set adult;
EducationLevel_clean = Education_Level;
if Education_Level in ("10th", "11th", "12th", "HS-grad") then EducationLevel_clean = "HighSchool";
	else if Education_Level in ("Masters", "Doctorate", "Bachelors", "Prof-school") then EducationLevel_clean = "HigherEducation";
	else if Education_Level in ("1st-4th", "Preschool", "5th-6th") then EducationLevel_clean = "Elementary";
	else if Education_Level in ("Some-college", "Assoc-voc", "Assoc-acdm") then EducationLevel_clean = "Post-High_School";
	else EducationLevel_clean = "MiddleSchool";
run;

data adult;
set adult;
WorkingClass_clean = WorkingClass;
if WorkingClass in ("Federal-gov", "Local-gov", "State-gov") then WorkingClass_clean = "Government";
	else if WorkingClass in ("Self-emp-not-inc", "Self-emp-inc") then WorkingClass_clean = "SelfEmployed";
	else if WorkingClass in ("Never-worked", "Without-pay", "NA") then WorkingClass_clean = "Others";
	else WorkingClass_clean = "Private";
run;

data adult;
set adult;
Occupation_clean = Occupation;
if Occupation in ("Exec-managerial", "Prof-specialty") then Occupation_clean = "Professional";
	else if Occupation in ("Machine-op-inspct", "Craft-repair") then Occupation_clean = "SkilledLabor";
	else if Occupation in ("Other-service", "Adm-clerical") then Occupation_clean = "Services";
	else if Occupation in ("Transport-moving", "Handlers-cleaners") then Occupation_clean = "ManualLabor";
	else if Occupation in ("Farming-fishing", "Tech-support") then Occupation_clean = "Specialized";
	else if Occupation in ("Protective-serv", "Armed-Forces") then Occupation_clean = "Protective Services";
	else if Occupation in ("Sales") then Occupation_clean = "Sales";
	else if Occupation in ("Priv-house-serv") then Occupation_clean = "Domestic";
	else Occupation_clean = "Others";
run;

/*************************************************************************************/

/* Create a binary variable by dividing income less than $50k as 0 and other as 1 */
data adult;
set adult;
if Income = "<=50K" then income_bin = 0;
	else income_bin = 1;
run;

/* GLM:  model */
proc logistic data = adult;
class Marital_Status EducationLevel_clean Occupation_clean Sex WorkingClass_clean Race / param = reference;
model income_bin (event='1') = Marital_Status Net_Capital EducationLevel_clean Age Occupation_clean Hrs_per_Week Sex WorkingClass_clean Race /clparm=both;
run;

/* Ask SAS to look into both confident intervals (Wald and Likelihood) */
proc logistic data = adult;
class WorkingClass_clean EducationLevel_clean sex Race Marital_Status Occupation_clean / param = reference;
model income_bin (event='1') = Marital_Status Net_Capital EducationLevel_clean Age 
                     Occupation_clean Hrs_per_Week Sex WorkingClass_clean Race 
                     /clparm=both firth;
output out = diagrams predicted = pred xbeta = line_pred;
run;


proc sort data = diagrams;
by Hrs_per_Week;
run;

proc sgplot data = diagrams;
scatter x = Hrs_per_Week y = income_bin;
series x = Hrs_per_Week y = pred;
run;


proc sort data = diagrams;
by Age;
run;

proc sgplot data = diagrams;
scatter x = Age y = income_bin;
series x = Age y = pred;
run;
/************************* Continous Part during a moment of Brain-break **************/
/* We are going to fit a GLM wit a Gamma component with a log link */
/* Random Component: Gamma, Link: Log */
proc genmod data = adult;
class WorkingClass_clean EducationLevel_clean sex Race Marital_Status Occupation_clean;
model income_bin = Net_Capital Race Age EducationLevel_clean sex Hrs_per_Week Marital_Status 
	 / dist = gamma link = log; /* we need to specify what we want to use in SAS */
run;
/* AIC = -60595.7769 */

/* Random Component: Inverse Gaussian, Link: Log */
proc genmod data = adult;
class WorkingClass_clean EducationLevel_clean sex Race Marital_Status Occupation_clean;
model income_bin = Net_Capital Race Age EducationLevel_clean sex Hrs_per_Week Marital_Status 
	 / dist = igaussian link = log; /* we need to specify what we want to use in SAS */
run;
/* AIC = -129446.9085 */

/* Random Component: Normal, Link = Log */
proc genmod data = adult;
class WorkingClass_clean EducationLevel_clean sex Race Marital_Status Occupation_clean;
model income_bin = Net_Capital Race Age EducationLevel_clean sex Hrs_per_Week Marital_Status 
	 / dist = normal link = log; /* we need to specify what we want to use in SAS */
run;
/* AIC = 24165.8937	 */

/* Random Component: Normal, Link = Identity */
proc genmod data = adult;
class WorkingClass_clean EducationLevel_clean sex Race Marital_Status Occupation_clean;
model income_bin = Net_Capital Race Age EducationLevel_clean sex Hrs_per_Week Marital_Status 
	 / dist = normal link = identity; /* we need to specify what we want to use in SAS */
run;
/* AIC = 24180.8155	 */

/* Random Component: Gamma, Link: Identity */
proc genmod data = adult;
class WorkingClass_clean EducationLevel_clean sex Race Marital_Status Occupation_clean;
model income_bin = Net_Capital Race Age EducationLevel_clean sex Hrs_per_Week Marital_Status 
	 / dist = gamma link = identity; /* we need to specify what we want to use in SAS */
run;
/* AIC = -60595.7769	 */

/* Random Component: Inverse Gaussian, Link: Identity */
proc genmod data = adult;
class WorkingClass_clean EducationLevel_clean sex Race Marital_Status Occupation_clean;
model income_bin = Net_Capital Race Age EducationLevel_clean sex Hrs_per_Week Marital_Status 
	 / dist = igaussian link = Identity; /* we need to specify what we want to use in SAS */
run;
/* AIC = -129446.9085 */ 
