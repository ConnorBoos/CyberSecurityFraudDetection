
rm(list = ls())

# Load required libraries
library(glmtoolbox) # For Fisher Scoring algorithm
library(caret)      # For cross-validation

# Step 1: Read in the data, put data in working directory folder
data <- read.csv("CreditCardFraud_2.csv")
data <- na.omit(data)

data$s_distance_from_home = scale(data$distance_from_home)
data$s_distance_from_last_transaction = scale(data$distance_from_last_transaction)
data$s_ratio_to_median_purchase_price = scale(data$ratio_to_median_purchase_price)


# Step 2: Split the data into training and testing datasets
set.seed(789) # for reproducibility
train_indices <- createDataPartition(data$fraud, p = 0.7, list = FALSE)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Step 3: Fit logistic regression model using Fisher Scoring on the training dataset
# Define formula
# formula <- as.formula("fraud ~ s_distance_from_home + s_distance_from_last_transaction + s_ratio_to_median_purchase_price + repeat_retailer + used_chip + used_pin_number + online_order")
formula <- as.formula("fraud ~ distance_from_home + distance_from_last_transaction + ratio_to_median_purchase_price + repeat_retailer + used_chip + used_pin_number + online_order")


# Fit logistic regression model using Fisher Scoring on training data
model <- glm(formula, data = train_data, family = binomial(link = "logit"), control=list(trace=TRUE))

FisherScoring(model)

# Step 4: Use the trained model to predict "fraud" on the testing dataset
# Predict on testing data
predictions <- predict(model, newdata = test_data, type = "response")

# Convert binary_predictions to factor
binary_predictions <- factor(binary_predictions, levels = c(0, 1))

# Convert test_data$fraud to factor with the same levels
test_data$fraud <- factor(test_data$fraud, levels = c(0, 1))

# Step 5: Print out the performance score
# Compare predictions to actual values in the testing dataset
# Calculate confusion matrix
confusion_matrix <- confusionMatrix(data = binary_predictions, reference = test_data$fraud)
print(confusion_matrix)


