#Clear workspace
rm(list=ls())

# Setting the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Read data
DiasBP = read.csv("DiasBP_Filtered.csv")[,-1]
HeartRate= read.csv("HeartRate_Filtered.csv")[,-1]
MeanBP = read.csv("MeanBP_Filtered.csv")[,-1]
SpO2 = read.csv("SpO2_Filtered.csv")[,-1]
SysBP = read.csv("SysBP_Filtered.csv")[,-1]
TempC = read.csv("TempC_Filtered.csv")[,-1]

DiasBP = DiasBP[(DiasBP$subject_id %in% HeartRate$subject_id),]
HeartRate = HeartRate[(HeartRate$subject_id %in% DiasBP$subject_id),]

DiasBP = DiasBP[(DiasBP$subject_id %in% MeanBP$subject_id),]
HeartRate = HeartRate[(HeartRate$subject_id %in% MeanBP$subject_id),]
MeanBP = MeanBP[(MeanBP$subject_id %in% DiasBP$subject_id),]

DiasBP = DiasBP[(DiasBP$subject_id %in% SpO2$subject_id),]
HeartRate = HeartRate[(HeartRate$subject_id %in% SpO2$subject_id),]
MeanBP = MeanBP[(MeanBP$subject_id %in% SpO2$subject_id),]
SpO2 = SpO2[(SpO2$subject_id %in% DiasBP$subject_id),]

DiasBP = DiasBP[(DiasBP$subject_id %in% SysBP$subject_id),]
HeartRate = HeartRate[(HeartRate$subject_id %in% SysBP$subject_id),]
MeanBP = MeanBP[(MeanBP$subject_id %in% SysBP$subject_id),]
SpO2 = SpO2[(SpO2$subject_id %in% SysBP$subject_id),]
SysBP = SysBP[(SysBP$subject_id %in% DiasBP$subject_id),]

y_AllData = DiasBP[,c("subject_id","LOS_ICU","icustay_expire_flag")]

rownames(DiasBP) = y_AllData$subject_id

idx = !(names(DiasBP) %in% c("subject_id","LOS_ICU","icustay_expire_flag"))

x_AllData = array(c(unlist(DiasBP[,idx]),unlist(HeartRate[,idx]),unlist(MeanBP[,idx]),unlist(SpO2[,idx]),unlist(SysBP[,idx])), dim = c(dim(DiasBP)[1], (dim(DiasBP)[2]-3), 5), dimnames = list(rownames(DiasBP[,idx]),colnames(DiasBP[,idx]),c("DiasBP","HeartRate","MeanBP","SpO2","SysBP")))

write.csv(x_AllData, file = "x_AllData.csv")
write.csv(y_AllData, file = "y_AllData.csv")
# 
# x_AllData[,,1] = array(unlist(DiasBP[,idx]), dim = c(dim(DiasBP)[1], (dim(DiasBP)[2]-3)), dimnames = list(y_AllData$subject_id,colnames(DiasBP[,idx])))
# x_AllData[,,2] = array(unlist(HeartRate[,idx]), dim = c(dim(DiasBP)[1], (dim(DiasBP)[2]-3)), dimnames = list(y_AllData$subject_id,colnames(DiasBP[,idx])))
# x_AllData[,,3] = array(unlist(MeanBP[,idx]), dim = c(dim(DiasBP)[1], (dim(DiasBP)[2]-3)), dimnames = list(y_AllData$subject_id,colnames(DiasBP[,idx])))
# x_AllData[,,4] = array(unlist(SpO2[,idx]), dim = c(dim(DiasBP)[1], (dim(DiasBP)[2]-3)), dimnames = list(y_AllData$subject_id,colnames(DiasBP[,idx])))
# x_AllData[,,5] = array(unlist(SysBP[,idx]), dim = c(dim(DiasBP)[1], (dim(DiasBP)[2]-3)), dimnames = list(y_AllData$subject_id,colnames(DiasBP[,idx])))
