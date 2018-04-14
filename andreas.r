install.packages("ggplot2")
library("ggplot2")
setwd("C:/Users/andre/Documents/Machine Learning/Projects/Project1/viKnepperDem")
data = read.csv("FINALDATAR.csv")

test1 = data[which(data$Line==1),]
test1 = test1[which(test1$Sum_Duration<200000),]
test2 = data[which(data$Line==2),]
test2 = test2[which(test2$Sum_Duration<200000),]
test_scrap = data[which(data$Scrap<0),]
plot(test1$Scrap[1:length(test1$Staff)])
plot(test1$Staff[1:length(test1$Staff)])
plot(test1$Error_Count[(96):(4*96)])
plot(test1$OutputGood~test1$Error_Count)
max(data$Staff)
