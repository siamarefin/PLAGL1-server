
# Check if the gplots package is installed, and install if it's not
if (!requireNamespace("gplots", quietly = TRUE)) {
  install.packages("gplots")
}

# Load the gplots library
library(gplots)

# Define the function
create_venn_dataframe <- function(gene_lists) {
  
  # Create Venn diagram data without plotting
  venn_data <- venn(gene_lists, show.plot = FALSE)
  
  # Access the intersection data
  intersections <- attr(venn_data, "intersections")
  
  # Initialize a list to store the results
  result <- list()
  
  # Iterate over the intersections to categorize genes
  for (set_name in names(intersections)) {
    genes <- intersections[[set_name]]
    if (length(genes) > 0) {
      result[[set_name]] <- genes
    }
  }
  
  # Convert the result list to a data frame
  max_length <- max(sapply(result, length))
  wide_df <- data.frame(matrix(ncol = length(result), nrow = max_length + 1))
  
  colnames(wide_df) <- names(result)
  
  # Fill the first row with counts of each intersection
  wide_df[1, ] <- sapply(result, length)
  
  # Fill the data frame with gene names
  for (i in seq_along(result)) {
    wide_df[2:(length(result[[i]]) + 1), i] <- result[[i]]
  }
  
  # Replace NA values with empty strings
  wide_df[is.na(wide_df)] <- ""
  
  # Display the modified data frame
  print(wide_df)
  
  return(wide_df)  # Optionally return the data frame
}