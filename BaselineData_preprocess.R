#Clear workspace
rm(list=ls())

# Setting the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

Data = read.csv("DeepL_FinalData_addICD9.csv")
Initial_Patients = dim(Data)[1]
Initial_Variables = dim(Data)[2]

Data_filtered = Data

library(ggplot2)
theme_update(plot.title = element_text(hjust = 0.5))
png("hist_age.png", units="px", width=2774, height=1600, res=300)
ggplot(Data, aes(x=AGE)) + 
  #geom_histogram(binwidth=10)+
  geom_histogram(color="darkblue", fill="lightblue")+
  labs(title="Age distribution at ICU admission",x="Age at ICU admission (Years)", y = "No. of patients")+
  theme_classic(base_size = 25)+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_vline(xintercept = 0, colour = 'red')+
  geom_vline(xintercept = 90, colour = 'red')
dev.off()

#Cohort selection
#Remove non adults (age < 15 years at ICU admission)
Data_filtered = Data_filtered[Data_filtered$AGE>=15,]
#Remove  adults with age = 300 years at ICU admission
Data_filtered = Data_filtered[Data_filtered$AGE<=90,]

png("hist_age_filtered.png", units="px", width=2774, height=1600, res=300)
ggplot(Data_filtered, aes(x=AGE)) + 
  # geom_histogram(binwidth=10)+
  geom_histogram(color="darkblue", fill="lightblue")+
  labs(title="Age distribution at ICU admission",x="Age at ICU admission (Years)", y = "No. of patients")+
  theme_classic(base_size = 25)+
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

#Remove stay < 1 day and > 50 days
Data_filtered = Data_filtered[Data_filtered$LOS_ICU>=24,]
Data_filtered = Data_filtered[Data_filtered$LOS_ICU<=24*30,]
#Remove patients with no heart rate data
# Data_filtered = Data_filtered[!is.na(Data_filtered$HeartRate_Mean),]

png("hist_LOS.png", units="px", width=2774, height=1600, res=300)
ggplot(Data, aes(x=LOS_ICU/24)) + 
  #geom_histogram(binwidth=10)+
  geom_histogram(color="darkblue", fill="lightblue")+
  labs(title="LOS distribution",x="Length-of-stay(LOS) (Days)", y = "No. of patients")+
  theme_classic(base_size = 25)+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_vline(xintercept = 1, colour = 'red')+
  geom_vline(xintercept = 30, colour = 'red')
dev.off()

png("hist_LOS_filtered.png", units="px", width=2774, height=1600, res=300)
ggplot(Data_filtered, aes(x=LOS_ICU/24)) + 
  #geom_histogram(binwidth=10)+
  geom_histogram(color="darkblue", fill="lightblue")+
  labs(title="LOS distribution",x="Length-of-stay(LOS) (Days)", y = "No. of patients")+
  theme_classic(base_size = 25)+
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

Filtered_Data = na.omit(Data)

#Variables = data.frame(colnames(Data))

#Checking if NA is present in the data
#Does not work on categorical variables
NA_Variables = t(data.frame(apply(Data, 2, function(x) anyNA(x))))
row.names(NA_Variables) = "Any_NA"

# Get variables which are factors
Factor_names = names(Filter(is.factor, Data_filtered))
Factor_names

#Check levels for each factor variable
for (i in 1:length(Factor_names)){
  NA_Variables[1,Factor_names[i]] = ifelse(any(levels(Data_filtered[,Factor_names[i]])==""),TRUE,FALSE)
}

#Religion and Marital status had blank entries
levels(Data_filtered$RELIGION)
levels(Data_filtered$MARITAL_STATUS)

#Fill blank entries with suitable values
levels(Data_filtered$RELIGION)[levels(Data_filtered$RELIGION)==""] = "NOT SPECIFIED"
levels(Data_filtered$MARITAL_STATUS)[levels(Data_filtered$MARITAL_STATUS)==""] = "UNKNOWN (DEFAULT)"
levels(Data_filtered$MARITAL_STATUS)[levels(Data_filtered$MARITAL_STATUS)=="UNKNOWN (DEFAULT)"] = "UNKNOWN"

#Checking Patients with NA entries
#Data_filtered$Any_NA = apply(Data_filtered, 1, function(x) anyNA(x))

Numeric_names = colnames(Data_filtered)
Numeric_names = setdiff(Numeric_names,Factor_names)
Numeric_names = Numeric_names[which(Numeric_names=="ALBUMIN_max"):which(Numeric_names=="TempC_Min")]

# #Create histogram and boxplots
# for (i in 1:length(Numeric_names)){
#   file_name = paste(Numeric_names[i],".png",sep = "", collapse = NULL)
#   png(file_name, units="px", width=3000, height=1600, res=300)
#   par(mfrow=c(1,2))
#   hist(Data_filtered[,Numeric_names[i]], xlab = Numeric_names[i], main = paste("Histogram of",Numeric_names[i], sep = " ", collapse = NULL))
#   boxplot(Data_filtered[,Numeric_names[i]], ylab = Numeric_names[i], main = paste("Boxplot of",Numeric_names[i], sep = " ", collapse = NULL))
#   dev.off()
# }
NA_percentage = vector(mode = "numeric", length = dim(Data_filtered)[2])
# Count NAs in each column
for (i in 1:length(NA_Variables)){
  NA_percentage[i] = sum(is.na(Data_filtered[,i]))*100/dim(Data_filtered)[1]
}

NA_data = rbind(NA_Variables,NA_percentage)
NA_data = data.frame(t(NA_data))

png("hist_dataloss.png", units="px", width=2774, height=1600, res=300)
ggplot(NA_data, aes(x=NA_percentage)) + 
  #geom_histogram(binwidth=10)+
  geom_histogram(color="darkblue", fill="lightblue")+
  labs(title="Histogram of features with data missing",x="Percentage of data missing", y = "No. of features")+
  theme_classic(base_size = 20)+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_vline(xintercept = 20, colour = 'red')
dev.off()

#Keep variables which has data on more than 80% of patients
Variables_keep = (NA_data[,2] <= 20)

#Final data to be used
Final_data = na.omit(Data_filtered[,Variables_keep])

#Normalizing the data
FinalNumeric_names = colnames(Final_data)
FinalNumeric_names = setdiff(FinalNumeric_names,Factor_names)
FinalNumeric_names
FinalNumeric_names = FinalNumeric_names[which(FinalNumeric_names=="ANIONGAP_max"):which(FinalNumeric_names=="TempC_Min")]
FinalNumeric_names = append("AGE",FinalNumeric_names)
FinalNumeric_names

#Logic mat concatenate function
logmat_concatenate = function(x){
  y = matrix(nrow = dim(x)[1], ncol = 1)
  for (i in 1:dim(x)[1]){
    y[i] = any(x[i,])
  }
  return(y)
}

#Remove patients with negative values
idx = logmat_concatenate(Final_data[,FinalNumeric_names]<0)
Final_data = Final_data[!idx,]

#Chloride can not be >= 160mmol/L
idx = logmat_concatenate(Final_data[,c("CHLORIDE_max","CHLORIDE_min")]>=160)
Final_data = Final_data[!idx,]

y = Final_data[,c("LOS_ICU","icustay_expire_flag")]
#"Primary_ICD9_CODE",
#,'ANIONGAP_max', 'ANIONGAP_min','BICARBONATE_min', 'BICARBONATE_min', 'PLATELET_max', 'PLATELET_max',"ETHNICITY_GROUPED","RELIGION","MARITAL_STATUS"
drops = c("SUBJECT_ID","hadm_id","ICUSTAY_ID","Hospital_expire_flag","LOS_ICU","icustay_expire_flag","elixhauser_SID29", "elixhauser_SID30")
x = Final_data[ , !(names(Final_data) %in% drops)]

FinalNumeric_names = setdiff(FinalNumeric_names,drops)


#Modify column names
colnames(y) = c("Duration","Event")

x_death = x[y$Event==1,]
y_death = y[y$Event==1,]

x_discharge = x[y$Event==0,]
y_discharge = y[y$Event==0,]

n_train_x_death = round(dim(x_death)[1]*0.8, digits = 0)
n_train_x_discharge = round(dim(x_discharge)[1]*0.8, digits = 0)

x_classify_train = rbind(x_death[1:n_train_x_death,],x_discharge[1:n_train_x_discharge,])
x_classify_test = rbind(x_death[-(1:n_train_x_death),],x_discharge[-(1:n_train_x_discharge),])

y_classify_train = rbind(y_death[1:n_train_x_death,],y_discharge[1:n_train_x_discharge,])
y_classify_test = rbind(y_death[-(1:n_train_x_death),],y_discharge[-(1:n_train_x_discharge),])

library(caret)
#Normalizing the data
Standadize = preProcess(x_classify_train[,FinalNumeric_names],method = c("range"),rangeBounds = c(0,1))

x_classify_train[,FinalNumeric_names] = predict(Standadize,x_classify_train[,FinalNumeric_names])
x_classify_test[,FinalNumeric_names] = predict(Standadize,x_classify_test[,FinalNumeric_names])

x_death[,FinalNumeric_names] = predict(Standadize,x_death[,FinalNumeric_names])
x_discharge[,FinalNumeric_names] = predict(Standadize,x_discharge[,FinalNumeric_names])

variable_list = colnames(x)

#Anonymizing function
name_remove = function(x){
  colnames(x) = NULL
  for (i in 1:dim(x)[2]){
    colnames(x)[i] = paste("Var_",i,sep = "", collapse = NULL)
  }
  return(x)
}

# x_classify_train = name_remove(x_classify_train)
# x_classify_test = name_remove(x_classify_test)
# 
# x_death = name_remove(x_death)
# x_discharge = name_remove(x_discharge)

#Save filtered data
#For Classificaation---------------------------------------------------------------------
write.csv(x_classify_train, file = "x_classify_train.csv")
write.csv(x_classify_test, file = "x_classify_test.csv")

write.csv(y_classify_train$Event, file = "y_classify_train.csv")
write.csv(y_classify_test$Event, file = "y_classify_test.csv")

#For Death---------------------------------------------------------------------
write.csv(x_death[1:n_train_x_death,], file = "x_death_train.csv")
write.csv(x_death[-(1:n_train_x_death),], file = "x_death_test.csv")

write.csv(y_death[1:n_train_x_death,1], file = "y_death_train.csv")
write.csv(y_death[-(1:n_train_x_death),1], file = "y_death_test.csv")

#For Discharge---------------------------------------------------------------------
write.csv(x_discharge[1:n_train_x_discharge,], file = "x_discharge_train.csv")
write.csv(x_discharge[-(1:n_train_x_discharge),], file = "x_discharge_test.csv")

write.csv(y_discharge[1:n_train_x_discharge,1], file = "y_discharge_train.csv")
write.csv(y_discharge[-(1:n_train_x_discharge),1], file = "y_discharge_test.csv")

write.csv(y, file = "y.csv")
write.csv(x,file = "x.csv")



write.csv(x_death, file = "x_death.csv")
write.csv(y_death, file = "y_death.csv")
write.csv(x_discharge, file = "x_discharge.csv")
write.csv(y_discharge, file = "y_discharge.csv")

#Remove column names and save data
colnames(y) = c("Duration","Event")
colnames(x) = NULL
for (i in 1:dim(x)[2]){
  colnames(x)[i] = paste("Var_",i,sep = "", collapse = NULL)
}
write.csv(y, file = "y_no_names.csv")
write.csv(x,file = "x_no_names.csv")

x_death = x[y$Event==1,]
y_death = y[y$Event==1,1]

x_discharge = x[y$Event==0,]
y_discharge = y[y$Event==0,1]

write.csv(x_death, file = "x_death_no_names.csv")
write.csv(y_death, file = "y_death_no_names.csv")
write.csv(x_discharge, file = "x_discharge_no_names.csv")
write.csv(y_discharge, file = "y_discharge_no_names.csv")

Subject_ID = Data[,c("SUBJECT_ID","LOS_ICU","icustay_expire_flag")]
save(Subject_ID,file = "Subject_ID.RData")
