input_genes_enrichment_analysis <- function(mapped_gene_str_IDs, output_dir){
  input_genes_enrichments <- enrichment_finder(string_db, mapped_gene_str_IDs)
  if (!is.null(input_genes_enrichments) && nrow(input_genes_enrichments) > 0) {
    # Generate file name for enrichments using output_dir
    input_genes_enrichment_file_name <- file.path(output_dir, "all_enrichments_of_initial_input_genes.csv")
    
    # OUTPUT: Save the enrichment results
    write.csv(input_genes_enrichments, input_genes_enrichment_file_name, row.names = FALSE)
    
    # Extract unique enrichment categories
    available_categories_ig <- extract_all_enrichments_categories(input_genes_enrichments)
    
    for (cat in available_categories_ig) {
      filtered_data <- filter_enrichments_by_category(input_genes_enrichments, cat)
      
      if (nrow(filtered_data) > 0) {
        print(paste("Plotting for category:", cat, "for initial input genes."))
        
        # Generate PDF file name within output_dir
        pdf_filename <- file.path(output_dir, paste0("enrichment_plot_for_initial_input_genes_", gsub(" ", "_", cat), ".pdf"))
        
        # Open PDF device
        pdf(pdf_filename)
        
        # Plot and print the top enrichments
        print(plot_top_enrichments(filtered_data, cat, gene_identity1))
        
        # Close PDF device
        dev.off()
        
      } else {
        print(paste("No enrichments found for category:", cat, "for input genes"))
      }
    }
  } else {
    print("No enrichments available for the input genes.")
  }
}
