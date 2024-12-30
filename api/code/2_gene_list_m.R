#################################################################################


# Helper function to get gene list from a CSV file with manual inputs
get_gene_list_from_csv <- function(file_path, gene_list_name) {
  # Check if the file exists

  print("DeBUG")
  print(file_path)
  if (!file.exists(file_path)) {
    message("File does not exist. Please check the path and try again.")
    return(NULL)
  }

  # Read the gene list from the CSV file
  data_i <- tryCatch(
    read.csv(file_path),
    error = function(e) {
      message("Error reading file. Please check the path and try again.")
      return(NULL)
    }
  )

  # Skip if the file couldn't be read
  if (is.null(data_i)) {
    return(NULL)
  }

  # Check if the required column exists
  col_name <- "external_gene_name" # Default column name
  if (!col_name %in% colnames(data_i)) {
    message("Error: The column '", col_name, "' is not found in the file. Please ensure it exists and try again.")
    return(NULL)
  }

  # Extract the specified column, remove duplicates and NA values
  gene_list_i <- na.omit(unique(data_i[[col_name]]))

  return(list(gene_list_name = gene_list_name, gene_list = gene_list_i))
}

# Main function to take user inputs for gene lists from external input_data
take_user_inputs <- function(input_data) {
  # Initialize a list to store gene lists
  gene_lists <- list()

  for (input in input_data) {
    # Call the helper function to get the gene list
    result <- get_gene_list_from_csv(input$file_path, input$gene_list_name)

    # Skip if there was an error
    if (is.null(result)) {
      next
    }

    # Store the gene list with the unique name
    gene_lists[[result$gene_list_name]] <- result$gene_list
  }

  return(gene_lists)
}
