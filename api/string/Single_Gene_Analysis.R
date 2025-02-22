source("3_single_gene_analysis.R")

############# Single Gene Analysis ##########################

for (single_string_id in mapped_gene_str_IDs) {
  map_reverese_ssi <- map_reverse_to_symbols(string_db, single_string_id)
  ssi_symbol <- map_reverese_ssi$alias

  # Finding Neighbor Genes of the Single Gene
  interaction_ssi <- get_single_gene_interactions(string_db, single_string_id)

  # Save interactions to CSV
  ssi_interactions_file_name <- paste0("files/all_neighbor_genes_of_", ssi_symbol, ".csv")
  write.csv(interaction_ssi$interactions, ssi_interactions_file_name, row.names = FALSE)

  # Plot interactions
  interactions_pdf_filename <- paste0("files/PPI_of_top_20_genes_and_", ssi_symbol, ".pdf")
  pdf(interactions_pdf_filename)
  print(plot_gene_interaction_network(string_db, interaction_ssi$combined_targets, ssi_symbol))
  dev.off()

  # Enrichment Analysis for Single Gene
  ssi_enrichment_results <- enrichment_finder(string_db, interaction_ssi$all_targets)

  if (!is.null(ssi_enrichment_results) && nrow(ssi_enrichment_results) > 0) {
    ssi_enrichment_file_name <- paste0("files/all_enrichments_of_", ssi_symbol, "_gene_and_its_neighbor_genes.csv")
    write.csv(ssi_enrichment_results, ssi_enrichment_file_name, row.names = FALSE)

    available_categories <- extract_all_enrichments_categories(ssi_enrichment_results)

    for (cat in available_categories) {
      filtered_data <- filter_enrichments_by_category(ssi_enrichment_results, cat)

      if (nrow(filtered_data) > 0) {
        print(paste("Plotting for category:", cat, "for gene:", ssi_symbol))

        pdf_filename <- paste0("files/enrichment_plot_", ssi_symbol, "_", gsub(" ", "_", cat), ".pdf")
        pdf(pdf_filename)
        print(plot_top_enrichments(filtered_data, cat, ssi_symbol))
        dev.off()
      }
    }
  } else {
    print(paste("No Enrichment is Evident for gene", single_string_id))
  }
}
