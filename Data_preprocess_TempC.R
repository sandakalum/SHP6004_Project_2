#Clear workspace
rm(list=ls())

# Setting the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Read data
Data = read.csv("TempC_48h.csv")

#Convert Charttime into proper Data and Time format
Data$Time = as.POSIXct(Data$Charttime)

Data2 = Data[,c("subject_id","TempC","Time")]

#Unique patient IDs
Patient_ID = unique(Data$subject_id)

#Data frame to store 48hr data
Filtered.data = matrix(nrow = length(Patient_ID),ncol = 26)
Filtered.data[,1] = Patient_ID

names = c("subject_id")
for (i in 1:25){
  names[i+1] = paste("Hour_",as.character(i-1),sep = "", collapse = NULL)
}
colnames(Filtered.data) = names

#Data filtering------------------------------------------------------------------------------
i = 1
for (i in (1:length(Patient_ID))){
  #Data for 1 patient
  idx1 = (Data2$subject_id==Patient_ID[i])
  Patient.data = na.omit(Data2[idx1,])
  
  #Data.points = dim(Patient.data)[1]
  
  #Get starting time
  Start.time = Patient.data[1,"Time"]
  
  #Starting data point entered
  TempC.data = matrix(nrow = 1,ncol = 25)
  idx2 = (Patient.data$Time<=Start.time)
  Hourly.data = Patient.data[idx2,"TempC"]
  #Remove outliers
  # Hourly.data = Hourly.data[(Hourly.data<=150) & (Hourly.data>=40)]
  if (length(Hourly.data) == 0){Hourly.data=0}
  TempC.data[1] = as.numeric(round(mean(Hourly.data),digits = 2))
  
  #Remove used data
  Patient.data = Patient.data[!idx2,]
  
  
  #j = 1
  for (j in 1:24){
    #Next hour end time (add hours to current time)
    Next.time = Start.time + j*60*60
    
    idx2 = (Patient.data$Time<=Next.time)
    Hourly.data = Patient.data[idx2,"TempC"]
    #Remove outliers
    # Hourly.data = Hourly.data[(Hourly.data<=150) & (Hourly.data>=40)]
    Mean.TempC = round(mean(Hourly.data), digits = 2)
    
    if (length(Hourly.data)==0){
      #If no data is available within the hour, use previous reading
      TempC.data[j+1] = TempC.data[j]
    } else {
      TempC.data[j+1] = Mean.TempC
    }
    #Remove used data
    Patient.data = Patient.data[!idx2,]
  }
  Filtered.data[i,2:26] = TempC.data
  if(i %% 1000 == 0) print(i)
}

#summary(Filtered.data)

b = Filtered.data
temp = matrix(nrow = dim(b)[1], ncol = 2)
load("Subject_ID.RData")
for (i in 1:dim(b)[1]){
  temp[i,] = as.numeric(Subject_ID[which(Subject_ID$SUBJECT_ID == b[i,1]),2:3])
}

#q = cbind(Filtered.data[,1],temp)
colnames(temp) = colnames(Subject_ID)[2:3]
Filtered.data = cbind(Filtered.data,temp)

#Remove NAs from ICU_LOS and ICU_Expiary_Flag data
Filtered.data = na.omit(Filtered.data)

write.csv(Filtered.data,file = "TempC_Filtered.csv")
save(Filtered.data,file = "TempC_Filtered.RData")


# b = Filtered.data[(Filtered.data[,1] %in% Subject_ID$SUBJECT_ID),]
# dim(Filtered.data)
# dim(b)


#Removing rows with zero entries
indremoved = which(apply(Filtered.data[,2:26], 1, function(x) any(x == 0)) )
Filtered.data2 = Filtered.data[ -indremoved, ]
write.csv(Filtered.data2,file = "TempC_Filtered_NOzero.csv")
