source("string/1_setup_string_env.R")
source("string/2_map_n_plot_input_genes.R")
source("string/3_single_gene_analysis.R")
source("string/4_enrichment_analysis_n_plot_alt.R")
source("string/5_query_genes_analysis.R")
################# Workflow #####################################

setup_stringdb_environment()

# Load organism list and initialize STRING database
organism_List <- read.csv("string/Organism_List.csv")
print(organism_List) #Front end

species_name <- 'Homo sapiens' # User Input
taxon_id <- get_taxon_id(organism_List, species_name)
string_db <- initialize_stringdb(taxon_id)

# Input gene list
gene_symbols <- c("ZNF212", "ZNF451", "PLAGL1", "NFAT5", "ICAM5", "RRAD") #User Input

######################### Mapping and Plotting Input Genes ###################

# Map genes to STRING IDs
mapped_genes <- map_genes_to_string_ids(string_db, gene_symbols)
write.csv(mapped_genes, "string_ids_according_to_initial_input_genes.csv")

mapped_gene_str_IDs <- mapped_genes$STRING_id

input_genes_neighbors <- find_unique_neighbors(string_db, mapped_gene_str_IDs)
neighbor_genes_symbols <- map_neighbors_to_symbols(string_db, input_genes_neighbors)
write.csv(neighbor_genes_symbols, "all_neighbor_genes_of_initial_input_genes.csv")

# Plot network and perform enrichment analysis for input genes
gene_identity1 <- "Input Genes" #For title in the figure
pdf("ppi_of_input_genes.pdf")
# Ensure the plot is printed
plot_network(string_db, mapped_gene_str_IDs, gene_identity1)
# Close PDF device
dev.off()

# Enrichment Analysis
input_genes_enrichments <- enrichment_finder(string_db, mapped_gene_str_IDs)

if (!is.null(input_genes_enrichments) && nrow(input_genes_enrichments) > 0) {
  # Generate file name for enrichments
  input_genes_enrichment_file_name <- paste0("all_enrichments_of_initial_input_genes.csv")

  # Write enrichments to CSV
  write.csv(input_genes_enrichments, input_genes_enrichment_file_name, row.names = FALSE)
  
  ################
  # Extract unique enrichment categories
  available_categories_ig <- extract_all_enrichments_categories(input_genes_enrichments)
  
  for (cat in available_categories_ig) {
    filtered_data <- filter_enrichments_by_category(input_genes_enrichments, cat)
    
    if (nrow(filtered_data) > 0) {
      print(paste("Plotting for category:", cat, "for initial input genes."))
      
      # Corrected gsub usage for file names
      pdf_filename <- paste0("enrichment_plot_for_initial_input_genes_", gsub(" ", "_", cat), ".pdf")
      
      # Open PDF device
      pdf(pdf_filename)
      
      gene_identity2 = 'Input'
      # Ensure the plot is printed
      print(plot_top_enrichments(filtered_data, cat, gene_identity2))
      
      # Close PDF device
      dev.off()
      
    } else {
      print(paste("No enrichments found for category:", cat, "for gene:", single_string_id))
    }
  }
}else{
  print("No enrichment available for the initial input genes.")
}

############# Single Gene Analysis ##########################

# Iterate over all genes in mapped_gene_str_IDs
for (single_string_id in mapped_gene_str_IDs) {
  map_reverese_ssi <- map_reverse_to_symbols(string_db, single_string_id)
  ssi_symbol <- map_reverese_ssi$alias
  # Finding Neighbor Genes of the Single Gene
  interaction_ssi <- get_single_gene_interactions(string_db, single_string_id)
  
  # Generate file name for interactions
  ssi_interactions_file_name <- paste0("all_neighbor_genes_of_", ssi_symbol, ".csv")
  
  # Write the interactions data to the CSV file
  write.csv(interaction_ssi$interactions, ssi_interactions_file_name, row.names = FALSE)
  
  # Plot for top 21 neighbor genes
  # Define the PDF file name
  interactions_pdf_filename <- paste0("PPI_of_top_20_genes_and_", ssi_symbol, ".pdf")
  
  # Open PDF device
  pdf(interactions_pdf_filename)
  
  # Ensure the plot is printed
  print(plot_gene_interaction_network(string_db, interaction_ssi$combined_targets, ssi_symbol))
  
  # Close PDF device
  dev.off()
  
  # Finding Enrichments of the Single Gene
  ssi_enrichment_results <- enrichment_finder(string_db, interaction_ssi$all_targets)
  
  # Check if enrichments are found
  if (!is.null(ssi_enrichment_results) && nrow(ssi_enrichment_results) > 0) {
    
    # Generate file name for enrichments
    ssi_enrichment_file_name <- paste0("all_enrichments_of_", ssi_symbol, "_gene_and_its_neighbor_genes.csv")
    
    # Write enrichments to CSV
    write.csv(ssi_enrichment_results, ssi_enrichment_file_name, row.names = FALSE)
    
    # Extract unique enrichment categories
    available_categories <- extract_all_enrichments_categories(ssi_enrichment_results)
    
    for (cat in available_categories) {
      filtered_data <- filter_enrichments_by_category(ssi_enrichment_results, cat)
      
      if (nrow(filtered_data) > 0) {
        print(paste("Plotting for category:", cat, "for gene:", ssi_symbol))
        
        # Corrected gsub usage for file names
        pdf_filename <- paste0("enrichment_plot_", ssi_symbol, "_", gsub(" ", "_", cat), ".pdf")
        
        # Open PDF device
        pdf(pdf_filename)
        
        # Ensure the plot is printed
        print(plot_top_enrichments(filtered_data, cat, ssi_symbol ))
        
        # Close PDF device
        dev.off()
        
      } else {
        print(paste("No enrichments found for category:", cat, "for gene:", single_string_id))
      }
    }
  } else {
    print(paste("No Enrichment is Evident for gene", single_string_id))
  }
}

############################# Query Gene Analysis ############################
# Load query genes
load_query_gene <- "Fibro_UP_genes.csv"
query_gene_ids <- find_unique_neighbors(string_db, mapped_gene_str_IDs)

# Match query genes with neighbors
matched_genes <- match_query_with_neighbors(load_query_gene, neighbor_genes_symbols)
matched_gene_ids <- matched_genes[, 1]
write.csv(matched_genes, "matched_genes_between_all_neighbor_genes_and_your_query_genes.csv")

# Plot network and perform enrichment analysis for input genes
gene_identity2 <- "Matched Genes" #For title in the figure
pdf("ppi_of_matched_genes.pdf")
# Ensure the plot is printed
plot_network(string_db, matched_gene_ids, gene_identity2)
# Close PDF device
dev.off()

matched_genes_enrichment <- enrichment_finder(string_db, matched_gene_ids)

if (!is.null(matched_genes_enrichment) && nrow(matched_genes_enrichment) > 0) {
  # Generate file name for enrichments
  matched_genes_enrichment_file_name <- paste0("all_enrichments_of_matched_genes.csv")
  
  # Write enrichments to CSV
  write.csv(matched_genes_enrichment, matched_genes_enrichment_file_name, row.names = FALSE)
  
  ################
  # Extract unique enrichment categories
  available_categories_mg <- extract_all_enrichments_categories(matched_genes_enrichment)
  gene_identity4 <- 'Matched Genes'
  for (cat in available_categories_mg) {
    filtered_data <- filter_enrichments_by_category(matched_genes_enrichment, cat)
    
    if (nrow(filtered_data) > 0) {
      print(paste("Plotting for category:", cat, "for matched genes."))
      
      # Corrected gsub usage for file names
      pdf_filename <- paste0("enrichment_plot_for_matched_genes_", gsub(" ", "_", cat), ".pdf")
      
      # Open PDF device
      pdf(pdf_filename)
  
      # Ensure the plot is printed
      plot_top_enrichments(filtered_data, cat, gene_identity4)
      
      # Close PDF device
      dev.off()
      
    } else {
      print(paste("No enrichments found for category:", cat, "for matched gene"))
    }
  }
}else{
  print("No enrichment available for the initial input genes.")
}

