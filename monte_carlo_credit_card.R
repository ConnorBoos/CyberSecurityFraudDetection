# Read the CSV file
data <- read.csv("CreditCardFraud_2.csv")

# Perform Monte Carlo simulation for fraud detection
monte_carlo_f_d <- function(data, num_iterations, threshold) {
  num_frauds <- sum(data$fraud == 1)
  num_transactions <- nrow(data)
  fraud_prob <- num_frauds / num_transactions
  
  detected_frauds <- numeric(num_iterations)
  
  for (i in 1:num_iterations) {
    simulated_data <- sample(c(0, 1), size = num_transactions, replace = TRUE, prob = c(1 - fraud_prob, fraud_prob))
    detected_frauds[i] <- sum(simulated_data == 1) > threshold * num_transactions
  }
  
  return(detected_frauds)
}

# Perform Monte Carlo simulation
num_iterations <- 1000 
threshold <- 0.02 

fraud_detection_results <- monte_carlo_f_d(data, num_iterations, threshold)

# Calculate detection rate
detection_rate <- sum(fraud_detection_results) / num_iterations
cat("Detection rate:", detection_rate, "\n")
