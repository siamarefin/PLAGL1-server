source("string/1_setup_string_env.R")
source("string/2_map_n_plot_input_genes.R")
source("string/3_single_gene_analysis.R")
source("string/4_enrichment_analysis_n_plot_alt.R")
source("string/5_query_genes_analysis.R")

setup_stringdb_environment()

# Load necessary libraries
library(jsonlite)
library(STRINGdb)

# Read arguments from the command line
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Error: Missing input arguments. Expected species name, gene symbols JSON, and output directory.")
}

species_name <- args[1]  # Read species name
gene_symbols_json <- args[2]  # Read gene symbols JSON
output_dir <- args[3]  # Output directory

# Convert JSON to R list
gene_symbols <- fromJSON(gene_symbols_json)

cat("Received Species Name:", species_name, "\n")
cat("Received Gene Symbols:", gene_symbols, "\n")
cat("Output Directory:", output_dir, "\n")

# Ensure output directory exists
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load STRING database
organism_List <- read.csv("string/Organism_List.csv")
taxon_id <- get_taxon_id(organism_List, species_name)
string_db <- initialize_stringdb(taxon_id)

# Map genes to STRING IDs
mapped_genes <- map_genes_to_string_ids(string_db, gene_symbols)
write.csv(mapped_genes, file.path(output_dir, "string_ids_according_to_initial_input_genes.csv"), row.names = FALSE)

mapped_gene_str_IDs <- mapped_genes$STRING_id

# Find and map neighbor genes
input_genes_neighbors <- find_unique_neighbors(string_db, mapped_gene_str_IDs)
neighbor_genes_symbols <- map_neighbors_to_symbols(string_db, input_genes_neighbors)
write.csv(neighbor_genes_symbols, file.path(output_dir, "all_neighbor_genes_of_initial_input_genes.csv"), row.names = FALSE)

# Plot network
pdf(file.path(output_dir, "ppi_of_input_genes.pdf"))
plot_network(string_db, mapped_gene_str_IDs, "Input Genes")
dev.off()

# Enrichment Analysis
input_genes_enrichments <- enrichment_finder(string_db, mapped_gene_str_IDs)

if (!is.null(input_genes_enrichments) && nrow(input_genes_enrichments) > 0) {
  enrichment_file <- file.path(output_dir, "all_enrichments_of_initial_input_genes.csv")
  write.csv(input_genes_enrichments, enrichment_file, row.names = FALSE)
  
  # Extract unique enrichment categories
  available_categories_ig <- extract_all_enrichments_categories(input_genes_enrichments)
  
  for (cat in available_categories_ig) {
    filtered_data <- filter_enrichments_by_category(input_genes_enrichments, cat)
    
    if (nrow(filtered_data) > 0) {
      pdf_filename <- file.path(output_dir, paste0("enrichment_plot_for_initial_input_genes_", gsub(" ", "_", cat), ".pdf"))
      
      pdf(pdf_filename)
      print(plot_top_enrichments(filtered_data, cat, "Input"))
      dev.off()
    }
  }
} else {
  cat("No enrichment available for the initial input genes.\n")
}

cat("Workflow completed successfully! Outputs saved in:", output_dir, "\n")
