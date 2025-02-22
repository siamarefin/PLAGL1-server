source("4_enrichment_analysis_n_plot_alt.R")
source("5_query_genes_analysis.R")

############################# Query Gene Analysis ############################

# Load query genes
load_query_gene <- "Fibro_UP_genes.csv"
query_gene_ids <- find_unique_neighbors(string_db, mapped_gene_str_IDs)

# Match query genes with neighbors
matched_genes <- match_query_with_neighbors(load_query_gene, neighbor_genes_symbols)
write.csv(matched_genes, "files/matched_genes_between_all_neighbor_genes_and_your_query_genes.csv")

# Plot network for matched genes
gene_identity2 <- "Matched Genes"
pdf("files/ppi_of_matched_genes.pdf")
plot_network(string_db, matched_genes[, 1], gene_identity2)
dev.off()

# Enrichment Analysis for Matched Genes
matched_genes_enrichment <- enrichment_finder(string_db, matched_genes[, 1])

if (!is.null(matched_genes_enrichment) && nrow(matched_genes_enrichment) > 0) {
  write.csv(matched_genes_enrichment, "files/all_enrichments_of_matched_genes.csv", row.names = FALSE)

  available_categories_mg <- extract_all_enrichments_categories(matched_genes_enrichment)
  gene_identity4 <- 'Matched Genes'

  for (cat in available_categories_mg) {
    filtered_data <- filter_enrichments_by_category(matched_genes_enrichment, cat)

    if (nrow(filtered_data) > 0) {
      print(paste("Plotting for category:", cat, "for matched genes."))

      pdf_filename <- paste0("files/enrichment_plot_for_matched_genes_", gsub(" ", "_", cat), ".pdf")
      pdf(pdf_filename)
      plot_top_enrichments(filtered_data, cat, gene_identity4)
      dev.off()
    }
  }
} else {
  print("No enrichment available for the matched genes.")
}
