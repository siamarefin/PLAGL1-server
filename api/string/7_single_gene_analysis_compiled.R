single_gene_analysis <- function(mapped_gene_str_IDs, output_dir){
  # Iterate over all genes in mapped_gene_str_IDs
  for (single_string_id in mapped_gene_str_IDs) {
    map_reverese_ssi <- map_reverse_to_symbols(string_db, single_string_id)
    ssi_symbol <- map_reverese_ssi$alias
    
    # Finding Neighbor Genes of the Single Gene
    interaction_ssi <- get_single_gene_interactions(string_db, single_string_id)
    
    # Generate file name for interactions using file.path()
    ssi_interactions_file_name <- file.path(output_dir, paste0("all_neighbor_genes_of_", ssi_symbol, ".csv"))
    
    # OUTPUT: Write neighbor interactions to CSV
    write.csv(interaction_ssi$interactions, ssi_interactions_file_name, row.names = FALSE)
    
    # Define the PDF file name for top neighbor genes plot
    interactions_pdf_filename <- file.path(output_dir, paste0("PPI_of_top_20_genes_and_", ssi_symbol, ".pdf"))
    
    # Open PDF device
    pdf(interactions_pdf_filename)
    
    # Output: Plot gene interaction network
    print(plot_gene_interaction_network(string_db, interaction_ssi$combined_targets))
    
    # Close PDF device
    dev.off()
    
    # Finding Enrichments of the Single Gene
    ssi_enrichment_results <- enrichment_finder(string_db, interaction_ssi$all_targets)
    
    # Check if enrichments are found
    if (!is.null(ssi_enrichment_results) && nrow(ssi_enrichment_results) > 0) {
      
      # Generate file name for enrichment results
      ssi_enrichment_file_name <- file.path(output_dir, paste0("all_enrichments_of_", ssi_symbol, "_gene_and_its_neighbor_genes.csv"))
      
      # OUTPUT: Write enrichment results to CSV
      write.csv(ssi_enrichment_results, ssi_enrichment_file_name, row.names = FALSE)
      
      # Extract unique enrichment categories
      available_categories <- extract_all_enrichments_categories(ssi_enrichment_results)
      
      for (cat in available_categories) {
        filtered_data <- filter_enrichments_by_category(ssi_enrichment_results, cat)
        
        if (nrow(filtered_data) > 0) {
          print(paste("Plotting for category:", cat, "for gene:", ssi_symbol))
          
          # Generate PDF file name for enrichment plot using file.path()
          pdf_filename <- file.path(output_dir, paste0("enrichment_plot_", ssi_symbol, "_", gsub(" ", "_", cat), ".pdf"))
          
          # Open PDF device
          pdf(pdf_filename)
          
          # Output: Plot the top enrichments
          print(plot_top_enrichments(filtered_data, cat, ssi_symbol))
          
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
}
