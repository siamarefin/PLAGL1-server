query_genes_enrichment_analysis <- function(matched_gene_ids, output_dir) {
  matched_genes_enrichment <- enrichment_finder(string_db, matched_gene_ids)
  
  if (!is.null(matched_genes_enrichment) && nrow(matched_genes_enrichment) > 0) {
    # Generate file name for enrichments using file.path()
    matched_genes_enrichment_file_name <- file.path(output_dir, "all_enrichments_of_matched_genes_between_all_neighbor_genes_and_your_query_genes.csv")
    
    # Output CSV
    write.csv(matched_genes_enrichment, matched_genes_enrichment_file_name, row.names = FALSE)
    
    ################
    # Extract unique enrichment categories
    available_categories_mg <- extract_all_enrichments_categories(matched_genes_enrichment)
    gene_identityA <- "Matched Genes"
    for (cat in available_categories_mg) {
      filtered_data <- filter_enrichments_by_category(matched_genes_enrichment, cat)
      
      if (nrow(filtered_data) > 0) {
        print(paste("Plotting for category:", cat, "for", gene_identityA))
        
        # Generate PDF file name using file.path()
        pdf_filename <- file.path(output_dir, paste0("enrichment_plot_for_matched_genes_between_all_neighbor_genes_and_your_query_genes_", gsub(" ", "_", cat), ".pdf"))
        
        # Open PDF device
        pdf(pdf_filename)
        
        # Output the plot
        print(plot_top_enrichments(filtered_data, cat, gene_identityA))
        
        # Close PDF device
        dev.off()
        
      } else {
        print(paste("No enrichments found for category:", cat, "for matched gene"))
      }
    }
  } else {
    print("No enrichment available for the matched genes.")
  }
}
