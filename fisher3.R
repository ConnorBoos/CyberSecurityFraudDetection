
rm(list = ls())

# Load required libraries
library(glmtoolbox) # For Fisher Scoring algorithm
library(caret)      # For cross-validation
library(ggcorrplot)
library(vtable)


# Step 1: Read in the data, put data in working directory folder
data <- read.csv("CreditCardFraud_2.csv")
data <- na.omit(data)

# Scale the variables
# data$s_distance_from_home = scale(data$distance_from_home)
# data$s_distance_from_last_transaction = scale(data$distance_from_last_transaction)
# data$s_ratio_to_median_purchase_price = scale(data$ratio_to_median_purchase_price)


# Step 2: Data visualization - Data Exploration 
### (Plot1)
par(mfrow = c(2, 4))
hist(data$fraud, col = "slateblue")
hist(data$distance_from_home, col = "slateblue")
hist(data$distance_from_last_transaction, col = "slateblue")
hist(data$ratio_to_median_purchase_price, col = "slateblue")
hist(data$repeat_retailer, col = "slateblue")
hist(data$used_chip, col = "slateblue")
hist(data$used_pin_number, col = "slateblue")
hist(data$online_order, col = "slateblue")

fraud <- data$fraud
distance_from_home <- data$distance_from_home
distance_from_last_transaction <- data$distance_from_last_transaction
ratio_to_median_purchase_price <- data$ratio_to_median_purchase_price
repeat_retailer <- data$repeat_retailer
used_chip <- data$used_chip
used_pin_number <- data$used_pin_number
online_order <- data$online_order

# Setting up the layout
par(mfrow = c(2, 4))

# Creating box plots for each variable with slate blue bars and without outliers
hist(data$fraud, col = "slateblue", main = "Fraud - Response Variable")
hist(data$repeat_retailer, col = "slateblue", main = "Repeat Retailer")
hist(data$used_chip, col = "slateblue", main = "Used Chip")
hist(data$used_pin_number, col = "slateblue", main = "Used Pin Number")
hist(data$online_order, col = "slateblue", main = "Online Order")
boxplot(distance_from_home, main = "Distance from Home", ylab = "Value", outline = FALSE, col = "slateblue")
boxplot(distance_from_last_transaction, main = "Distance from Last Transaction", ylab = "Value", outline = FALSE, col = "slateblue")
boxplot(ratio_to_median_purchase_price, main = "Ratio to Median Purchase Price", ylab = "Value", outline = FALSE, col = "slateblue")

### Plot 2 

# Compute the correlation matrix for the dataframe 'data'
correlation_matrix <- cor(data)
# Print the correlation matrix
print(correlation_matrix)

# Compute the correlation matrix for the dataframe 'data' and round it to four decimal digits
corr <- round(cor(data), 6)
# Print the correlation matrix
print(corr)

# Assuming 'cor_pmat()' calculates the p-values for correlation coefficients
p.mat <- cor_pmat(data)
# Plot the correlation matrix using ggcorrplot
ggcorrplot(corr, hc.order = FALSE, type = "lower", lab = TRUE, lab_size = 2, digits = 3) +
  theme(axis.text.x = element_text(size = 8))  # Adjust the size of x-axis labels as needed

### Plot 3
st(data)


# Step 3: Split the data into training and testing datasets
set.seed(789) # for reproducibility
train_indices <- createDataPartition(data$fraud, p = 0.7, list = FALSE)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]



# Step 4: Fit logistic regression model using Fisher Scoring on training data
# Define formula
formula <- as.formula("fraud ~ distance_from_home 
                       + distance_from_last_transaction 
                       + ratio_to_median_purchase_price 
                       + repeat_retailer 
                       + used_chip 
                       + used_pin_number 
                       + online_order")

# Fit logistic regression model using Fisher Scoring on training data
model <- glm(formula, data = train_data, family = binomial(link = "logit"), 
             control=list(trace=TRUE))

# Display the path performed by the Fisher Scoring algorithm
FisherScoring(model)





formula <- as.formula("fraud ~ distance_from_home + distance_from_last_transaction + ratio_to_median_purchase_price + repeat_retailer + used_chip + used_pin_number + online_order")

# Fit logistic regression model using Fisher Scoring on training data
model <- glm(formula, data = train_data, family = binomial(link = "logit"), 
             control=list(trace=TRUE))

# Summary Results from Logistic Regression using Fisher Scoring
summary(model)

# Display the path performed by the Fisher Scoring algorithm
FisherScoring(model)




# Step 5: Use the trained model to predict "fraud" on the testing dataset
# Predict on testing data
predictions <- predict(model, newdata = test_data, type = "response")

# Convert binary_predictions to factor
binary_predictions <- factor(predictions > 0.5, levels = c(FALSE, TRUE))

# Convert test_data$fraud to factor with the same levels
test_data$fraud <- factor(test_data$fraud, levels = c(0, 1))

# Step 6: Print out the performance score
# Compare predictions to actual values in the testing dataset
# Calculate confusion matrix
confusion_matrix <- table(binary_predictions, test_data$fraud)
print(confusion_matrix)


# Extract true positives, true negatives, false positives, and false negatives
tp <- confusion_matrix[2, 2]  # True Positives
tn <- confusion_matrix[1, 1]  # True Negatives
fp <- confusion_matrix[1, 2]  # False Positives
fn <- confusion_matrix[2, 1]  # False Negatives

# Calculate accuracy
accuracy <- (tp + tn) / sum(confusion_matrix)

# Print accuracy in percentage and decimal form
cat("Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Accuracy:", round(accuracy, 4), "\n")

# formula <- as.formula("fraud ~ s_distance_from_home + s_distance_from_last_transaction + s_ratio_to_median_purchase_price + repeat_retailer + used_chip + used_pin_number + online_order")
