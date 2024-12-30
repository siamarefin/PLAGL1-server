# Load required libraries
library(jsonlite)  # For generating JSON output

# Get the user_id as an argument
args <- commandArgs(trailingOnly = TRUE)
user_id <- args[1]

# Define the path to the RDS file
rds_path <- file.path("code", user_id, "rds", "count_data.rds")

tryCatch({
  # Check if the file exists
  if (!file.exists(rds_path)) {
    result <- list(success = FALSE, message = paste("File not found:", rds_path))
  } else {
    # Read the RDS file
    count_data <- readRDS(rds_path)
    
    # Extract column names
    column_names <- colnames(count_data)
    result <- list(success = TRUE, columns = column_names)
  }
  
  # Output the result as JSON
  cat(toJSON(result, auto_unbox = TRUE))
}, error = function(e) {
  # Handle errors and output as JSON
  error_result <- list(success = FALSE, message = e$message)
  cat(toJSON(error_result, auto_unbox = TRUE))
})
