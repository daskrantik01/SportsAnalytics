
library(tidyverse)
Barcelona_data <- read_csv("/Users/daskrantik01/Downloads/Barcelona_20_21 FINAL.csv")
library(markovchain)
Barcelona_data
# Create a sequence of states (replace with your own data)
# Define the possible states
match_results <- c(
  "WDD", "DDD", "DDW", "DWD", "WDW", "DWD", "WDD", "DDW", "DWW", "WWW",
  "WWD", "WDW", "DWD", "WDW", "DWW", "WWD", "WDD", "DDW", "DWD", "WDL",
  "DLL", "LLL", "LLW", "LWW", "WWW", "WWW", "WWW", "WWL", "WLW", "LWW",
  "WWW", "WWL", "WLW", "LWD", "WDW", "DWD", "WDD", "DDD", "DDL", "DLW",
  "LWW", "WWW", "WWD", "WDD", "DDD", "DDD", "DDW", "DWL", "WLW", "LWW",
  "WWW", "WWW", "WWL", "WLD", "LDL", "DLW", "LWL", "WLW", "LWD", "WDD",
  "DDW", "DWD", "WDD", "DDW", "DWW", "WWW", "WWW", "WWW", "WWW", "WWW",
  "WWD", "WDW", "DWL", "WLW"
) 

# Define the possible state spaces
state_spaces <- unique(match_results)

# Create an empty transition probability matrix
n_states <- length(state_spaces)
transition_matrix <- matrix(0, nrow = n_states, ncol = n_states, dimnames = list(state_spaces, state_spaces))

# Function to update the transition matrix
update_transition_matrix <- function(matrix, sequence) {
  from_state <- sequence[1]
  to_state <- sequence[2]
  matrix[from_state, to_state] <- matrix[from_state, to_state] + 1
  return(matrix)
}

# Loop through the match results and update the transition matrix
for (i in 1:(length(match_results) - 1)) {
  transition_matrix <- update_transition_matrix(transition_matrix, c(match_results[i], match_results[i + 1]))
}

# Normalize the transition matrix to obtain probabilities
row_sums <- rowSums(transition_matrix)
transition_matrix_prob <- transition_matrix / row_sums

# Print the transition probability matrix
print(transition_matrix_prob)

View(transition_matrix_prob)

# Load the required package for color palettes
library(RColorBrewer)


# Define the custom color palette
custom_palette <- c("beige", brewer.pal(8, "YlGnBu"))

# Create a heatmap with the custom color palette
heatmap(transition_matrix_prob, 
        col = custom_palette,
        xlab = "State Spaces",
        ylab = "State Spaces",
        labRow = NA,  # Remove row labels
        labCol = NA)  # Remove column labels


# Define the number of iterations for Gibbs sampling
num_iterations <- 1000


# Number of burn-in iterations
burn_in_iterations <- 100

# Ensure state_sequence contains unique state sequences
match_results <- unique(match_results)

# Create an index map to match state sequences to matrix indices
state_indices <- match(match_results, rownames(transition_matrix_prob))

# Normalize the transition probability matrix
transition_matrix_prob_normalized <- transition_matrix_prob / rowSums(transition_matrix_prob)

# Initialize an initial state
current_state_index <- sample(1:length(match_results), 1)

# Initialize empty vectors to store prior and posterior samples
prior_samples <- character(0)
posterior_samples <- character(0)

# Gibbs sampling loop
for (iteration in 1:num_iterations) {
  # Sample the prior distribution
  prior_sample_index <- sample(1:length(match_results), 1, prob = transition_matrix_prob_normalized[current_state_index, ])
  prior_samples <- c(prior_samples, match_results[prior_sample_index])
  
  # Update the current state
  current_state_index <- prior_sample_index
  
  # Sample the posterior distribution
  posterior_sample_index <- sample(1:length(match_results), 1, prob = transition_matrix_prob_normalized[current_state_index, ])
  posterior_samples <- c(posterior_samples, match_results[posterior_sample_index])
}


# Optionally, you can convert state indices back to state sequences for interpretation
posterior_samples_sequences <- match_results[match(posterior_samples, match_results)]

# Calculate the prior and posterior distributions
prior_distribution <- table(prior_samples) / length(prior_samples)
posterior_distribution <- table(posterior_samples_sequences) / length(posterior_samples_sequences)

# Print or visualize the prior and posterior distributions
print("Prior Distribution:")
print(prior_distribution)

print("Posterior Distribution:")
print(posterior_distribution)

# Create a bar plot of the posterior distribution
barplot(posterior_distribution, names.arg = names(posterior_distribution), 
        xlab = "State", ylab = "Probability", main = "Posterior Distribution")


#CONVERGENCE
# Initialize the vector for trace plot
trace_plot <- numeric(num_iterations)

y_axis_limits <- c(0.0, 0.01)
# Create a trace plot for the posterior distribution
plot(1:num_iterations, trace_plot, type = "l", col = "blue", xlab = "Iteration", ylab = "State Index", main = "Trace Plot for Posterior Distribution",ylim = y_axis_limits)



#STATIONARY DISTRIBUTION


# Compute the eigenvalues and eigenvectors
eigen_result <- eigen(t(transition_matrix_prob))  # Transpose the matrix for column-wise eigenvectors

# Find the index of the eigenvalue closest to 1 (stationary distribution)
index <- which(abs(eigen_result$values - 1) < 1e-8)

# Extract the corresponding eigenvector as the stationary distribution
stationary_distribution <- eigen_result$vectors[, index]

# Normalize the stationary distribution to sum to 1 (if needed)
stationary_distribution <- stationary_distribution / sum(stationary_distribution)

# Print the stationary distribution
print(stationary_distribution)

#MATRIX MULTIPLICATION OVER A NUMBER OF TIME PERIODS

#Matrx multiplication
#t=2
result2 = transition_matrix_prob %*% transition_matrix_prob
view(result2)


#t=7
result7 = transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob 
view(result7)

#t=8
result8 = transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob %*% transition_matrix_prob
view(result8)



# Define the transition matrix (adjust this based on your specific matrix)
transition_matrix <- transition_matrix_prob

# Define a tolerance level for convergence
tolerance <- 1e-7

# Initialize an iteration counter
iterations <- 0

# Create a copy of the transition matrix for comparison
previous_matrix <- transition_matrix

# Perform matrix multiplication iteratively until convergence
while (TRUE) {
  iterations <- iterations + 1
  
  # Calculate the next iteration of the transition matrix
  transition_matrix <- transition_matrix %*% previous_matrix
  
  # Check for convergence by comparing the difference between iterations
  if (max(abs(transition_matrix - previous_matrix)) < tolerance) {
    break
  }
  
  # Update the previous matrix for the next iteration
  previous_matrix <- transition_matrix
}

# Print the number of iterations required for convergence
cat("Number of iterations for convergence:", iterations, "\n")

# Print the stationary matrix
cat("Stationary Matrix:\n")
View(transition_matrix)


# Load the Matrix package
library(Matrix)

# Create a sparse matrix (replace this with your actual sparse matrix)
your_sparse_matrix <- transition_matrix_prob

# Raise the sparse matrix to the power of 42
result1 <- (your_sparse_matrix)^42
result1_rounded <- round(result1, digits = 10)



View(result1_rounded)
