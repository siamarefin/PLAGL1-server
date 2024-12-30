

################################################################################
# Define the function to extract log2FoldChange from multiple CSVs with user-specified column names
extract_lfc_multiple <- function(gene_list, csv_files, column_names, gene_col = "external_gene_name", lfc_col = "log2FoldChange") {
  # Initialize the data frame with gene names as the first column
  gene_frame <- data.frame(Gene = gene_list)
  
  # Loop over the provided CSV files and column names
  for (i in seq_along(csv_files)) {
    csv_file <- csv_files[i]
    column_name <- column_names[i]
    
    # Check if the file exists
    if (!file.exists(csv_file)) {
      cat("File", csv_file, "does not exist. Skipping...\n")
      next
    }
    
    # Read the CSV file
    data <- read.csv(csv_file)
    
    # Check if the necessary columns exist in the data
    if (!gene_col %in% colnames(data) || !lfc_col %in% colnames(data)) {
      cat("Columns", gene_col, "or", lfc_col, "not found in", csv_file, ". Skipping...\n")
      next
    }
    
    # Extract the log2FoldChange values for the genes in gene_list
    lfc_values <- data[data[[gene_col]] %in% gene_list, c(gene_col, lfc_col)]
    
    # Match the log2FoldChange values with the gene list order
    matched_lfc <- sapply(gene_list, function(gene) {
      ifelse(gene %in% lfc_values[[gene_col]], 
             lfc_values[[lfc_col]][lfc_values[[gene_col]] == gene], NA)
    })
    
    # Add the log2FoldChange values as a new column in gene_frame
    gene_frame[[column_name]] <- matched_lfc
  }
  
  # Return the combined data frame
  return(gene_frame)
}