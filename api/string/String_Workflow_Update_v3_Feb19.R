# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript script.R species_name gene_symbols_json clustering_method1 clustering_method2 output_dir")
}

species_name      <- args[1]
gene_symbols_json <- args[2]
clustering_method1 <- args[3]  # For input gene clustering
clustering_method2 <- args[4]  # For matched gene clustering
output_dir        <- args[5]

# Load necessary package for JSON parsing
library(jsonlite)
gene_symbols <- fromJSON(gene_symbols_json)

# Ensure the output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Source all required R scripts
source("string/1_setup_string_env.R")
source("string/2_map_n_plot_input_genes.R")
source("string/3_single_gene_analysis.R")
source("string/4_enrichment_analysis_n_plot_alt.R")
source("string/5_query_genes_analysis.R")
source("string/6_cluster_finder.R")
source("string/7_single_gene_analysis_compiled.R")
source("string/8_query_gene_network_analysis.R")
source("string/9_query_gene_enrichment_analysis.R")
source("string/11_input_genes_enrichment_analysis.R")

################# Workflow #####################################
setup_stringdb_environment()

# Load organism list and initialize STRING database
organism_List <- read.csv("string/Organism_List.csv") 
print(organism_List)

taxon_id  <- get_taxon_id(organism_List, species_name)
string_db <- initialize_stringdb(taxon_id)

# Input gene list provided by the user
# species_name and gene_symbols come from the command-line arguments
# Example: species_name <- 'Homo sapiens'
#          gene_symbols <- c("ZNF212", "ZNF451", "PLAGL1", "NFAT5", "ICAM5", "RRAD")

# Map genes to STRING IDs
mapped_genes <- map_genes_to_string_ids(string_db, gene_symbols)
write.csv(mapped_genes, file.path(output_dir, "string_ids_according_to_initial_input_genes.csv"))

mapped_gene_str_IDs <- mapped_genes$STRING_id

# Find neighbors of the input genes and map to gene symbols
input_genes_neighbors <- find_unique_neighbors(string_db, mapped_gene_str_IDs)
neighbor_genes_symbols <- map_neighbors_to_symbols(string_db, input_genes_neighbors)
write.csv(neighbor_genes_symbols, file.path(output_dir, "all_neighbor_genes_of_initial_input_genes.csv"))

# Plot the PPI network of the input genes and save as PDF
gene_identity1 <- "Input Genes"
pdf(file.path(output_dir, "ppi_network_of_input_genes.pdf"))
plot_network(string_db, mapped_gene_str_IDs, gene_identity1)
dev.off()

####################### Cluster Finder #######################
clustering_methods <- c("fastgreedy", "walktrap", "spinglass", "edge.betweenness")
print(clustering_methods)

# INPUT: Use clustering_method1 for clustering the input gene list
find_all_clusters(gene_symbols, clustering_method1,output_dir)

####################### Enrichment Analysis for Input Genes #########
input_genes_enrichment_analysis(mapped_gene_str_IDs, output_dir)

############################# Single Gene Analysis ##########################
single_gene_lists <- mapped_gene_str_IDs
single_gene_analysis(single_gene_lists, output_dir)

##############################################################################
############################# Query Gene Analysis ############################

load_query_gene <- "string/Fibro_UP_genes.csv"
matched_gene_ids <- query_gene_network_analysis(load_query_gene, mapped_gene_str_IDs, neighbor_genes_symbols,output_dir)

##################### Cluster Finder for Matched Genes #######################
print(clustering_methods)
# INPUT: Use clustering_method2 for clustering the matched genes
find_all_clusters(matched_gene_ids, clustering_method2)

############################# Enrichment Analysis for Query Genes ############################
query_genes_enrichment_analysis(matched_gene_ids, output_dir)
