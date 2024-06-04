install.packages("kdensity")
install.packages("KernSmooth")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("MASS")
install.packages("caret")
install.packages("psych")
library("kdensity")
library("KernSmooth")
library("ggplot2")
library("dplyr") 
library("MASS")
library("caret")
library("klaR")
library("psych")
library("ggord")
library("devtools")

fraud <- read.csv('C:/Users/Connor Boos/Downloads/CreditCardFraud_2.csv', sep = ",", dec = ".", header=TRUE)
pca <- read.csv('C:/Users/Connor Boos/Downloads/CreditCardFraud_PCA_Data_3.csv', sep = ",", dec = ".", header=TRUE)
synth <- read.csv('C:/Users/Connor Boos/Downloads/Synthetic Financial Datasets For Fraud Detection_1.csv', sep = ",", dec = ".", header=TRUE)
synth_subset <- sample_n(synth, 10000)
less_fields <- table(synth_subset$amount, synth_subset$newbalanceOrig, synth_subset$isFraud)
less_fields[1:4] <- scale(less_fields[1:4])
apply(less_fields[1:4], 2, mean)

pairs.panels(less_fields[1:4],
             gap = 0,
             bg = c("red", "blue")[less_fields$isFraud],
             pch = 21)

sample = sample_n(synth, 1000)

fraud_flag = synth[synth$isFlaggedFraud == 1,]
fraud_values<- log(fraud_flag$amount)
hist(fraud_values,freq = FALSE)
dens <- density(fraud_values)
# Overlay density curve
lines(dens, col = "red")

plot(synth[,c(3,6)],col=synth[,10])
x <- log(synth$amount)
hist(x,freq = FALSE)
dens <- density(x)
# Overlay density curve
lines(dens, col = "red")

train_indices <- createDataPartition(synth$amount, p = 0.8, list = FALSE)
train_val <- synth[train_indices,]
test_val <- synth[-train_indices,]

preproc.parameter <- train_val %>%  
  preProcess(method = c("center", "scale")) 

train_valtransform <- preproc.parameter %>% predict(train_val) 
test_valtransform <- preproc.parameter %>% predict(test_val) 

model <- lda(amount~., data = train_valtransform)  

predictions <- model %>% predict(test_valtransform) 
mean(predictions$class==test_valtransform$amount) 
model <- lda(amount~., data = train_valtransform)  
model 




kde = kdensity(abs(log(sample$newbalanceOrig)), start = "gumbel", kernel = "gaussian")

fraud_flag = synth[synth$isFlaggedFraud == 1,]
ggplot(fraud_flag,aes(x=amount,y=newbalanceDest,col=isFraud))+geom_point()

plot(sample$newbalanceOrig, sample$newbalanceDest)
ggplot(sample,aes(x=log(amount),y=newbalanceDest,col=isFraud))+geom_point()

##Looking at the different variables
counts <- table(synth$type)
barplot(counts,main="Type Distribution", xlab="Type")

counts1 <- table(synth$type, synth$isFlaggedFraud)
barplot(as.matrix(counts1), legend.text=T, col=c("red" , "green", 
                                                            "blue","yellow","orange"), ylab="count")

# Data for flagged fraud
fraud_flag = synth[synth$isFlaggedFraud == 1,]
counts2 <- table(fraud_flag$type, fraud_flag$isFlaggedFraud)
barplot(as.matrix(counts2), legend.text=T, col=c("red" , "green", 
                                                 "blue","yellow","orange"), ylab="count")

#plot(synth$oldbalanceOrg, synth$newbalanceDest, pch=16,col=synth$isFraud, legend.text=T )
ggplot(synth,aes(x=oldbalanceOrg,y=newbalanceDest,col=isFraud))+geom_point()
#So the flagged fraud is entirely based on transfers

data(synth, package = "HSAUR")
synth <- bkde2D(synth, bandwidth = sapply(synth, ))

x=synth$amount
xsorted=sort(x)
x_i=xsorted
hist(x_i, nclass=308)
n=length(x_i)
h1=.002
t=seq(min(x_i),max(x_i),10)
M=length(t)
fhat1=rep(0,M)
for (i in 1:M){
  fhat1[i]=sum(dnorm((t[i]-x_i)/h1))/(n*h1)}
lines(t, fhat1, lwd=2, col="blue")



types = table()

ncol(types)


types[0][2]

hist(synth$oldbalanceOrg, breaks=10)

plot(kde, main = "Miles per Gallon")
lines(kde, plot_start = TRUE, col = "red")

x <- 2^rnorm(100)
fhat <- kde(x=synth$newbalanceDest, positive=TRUE)
plot(fhat, col=3)
points(c(0.5, 1), predict(fhat, x=c(0.5, 1)))

## large data example on non-default grid
## 151 x 151 grid = [-5,-4.933,..,5] x [-5,-4.933,..,5]
set.seed(8192)
x <- rmvnorm.mixt(10000, mus=c(0,0), Sigmas=invvech(c(1,0.8,1)))
fhat <- kde(x=x, binned=TRUE, compute.cont=TRUE, xmin=c(-5,-5), xmax=c(5,5), bgridsize=c(151,151))
plot(fhat)