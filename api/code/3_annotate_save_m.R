# Define a function to perform annotation, replace gene IDs with gene names, and save the results
annotate_and_replace_ids <- function(input_file_path, output_file_path, ensembl_mart, id_type) {
  # Set default attributes for annotation (include both Ensembl and Entrez IDs)
  default_attributes <- c(
    "ensembl_gene_id",       # Ensembl Gene ID
    "entrezgene_id",         # Entrez Gene ID
    "external_gene_name",    # Gene Symbol (Gene Name)
    "description"            # Gene Description
  )
  
  # Read the input CSV file and set the column name of the first column to "X"
  ids_1 <- read.csv(input_file_path)
  colnames(ids_1)[1] <- "Given_id"
  
  # Extract the gene IDs
  ids <- ids_1$Given_id
  
  # Perform annotation based on the ID type
  if (id_type == "ensembl") {
    # Annotation using Ensembl IDs
    annotations <- getBM(
      attributes = default_attributes,
      filters = "ensembl_gene_id",
      values = ids,
      mart = ensembl_mart
    )
    cat("Annotation performed using Ensembl IDs.\n")
  } else if (id_type == "entrez") {
    # Annotation using Entrez IDs
    annotations <- getBM(
      attributes = default_attributes,
      filters = "entrezgene_id",
      values = ids,
      mart = ensembl_mart
    )
    cat("Annotation performed using Entrez IDs.\n")
  }
  
  # Merge the annotation results with the original data
  annotated_data <- merge(data.frame(Gene_ID = ids, ids_1, row.names = NULL),
                          annotations, by.x = "Gene_ID", by.y = ifelse(id_type == "ensembl", "ensembl_gene_id", "entrezgene_id"), all.x = TRUE)
  
  # Remove rows where no gene name was found (i.e., NA in external_gene_name or empty values)
  annotated_data <- annotated_data[!is.na(annotated_data$external_gene_name) & 
                                     annotated_data$external_gene_name != "", ]
  
  # Dynamically select columns based on available data
  columns_to_include <- c()  # Initialize the column order
  
  # Add both Ensembl and Entrez IDs if they exist
  if ("ensembl_gene_id" %in% colnames(annotated_data)) {
    columns_to_include <- c(columns_to_include, "ensembl_gene_id")
  }
  if ("entrezgene_id" %in% colnames(annotated_data)) {
    columns_to_include <- c(columns_to_include, "entrezgene_id")
  }
  
  # Add Gene Name (external_gene_name) and Description (if available)
  columns_to_include <- c(columns_to_include, "external_gene_name")
  if ("description" %in% colnames(annotated_data)) {
    columns_to_include <- c(columns_to_include, "description")
  }
  
  # Add the original data columns (excluding the first one, since it's used for ID)
  columns_to_include <- c(columns_to_include, colnames(ids_1))
  
  # Reorder the columns
  annotated_data <- annotated_data[, columns_to_include, drop = FALSE]  # Use drop = FALSE to avoid dropping data
  
  # Save the annotated data with gene IDs, gene names, and descriptions to a new CSV file
  write.csv(annotated_data, file = output_file_path, row.names = FALSE)
  
  cat("Annotated data with both Ensembl and Entrez IDs, gene names, and descriptions saved to", output_file_path, "\n")
}

