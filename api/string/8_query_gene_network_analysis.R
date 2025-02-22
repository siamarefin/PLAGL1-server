query_gene_network_analysis <- function(load_query_gene, mapped_gene_str_IDs, neighbor_genes_symbols, output_dir) {
  query_gene_ids <- find_unique_neighbors(string_db, mapped_gene_str_IDs)
  
  # Match query genes with neighbors
  matched_genes <- match_query_with_neighbors(load_query_gene, neighbor_genes_symbols)
  matched_gene_ids <- matched_genes[, 1]
  
  # OUTPUT: Save the matched genes CSV to the output directory
  write.csv(matched_genes, file.path(output_dir, "matched_genes_between_all_neighbor_genes_of_input_genes_and_your_query_genes.csv"))
  
  # Plot network and perform enrichment analysis for input genes
  pdf(file.path(output_dir, "ppi_of_matched_genes_between_neighbor_genes_of_input_genes_and_your_query_genes.pdf"))
  gene_identityN <- "Query Genes"
  
  # Output the plot to the PDF
  print(plot_network(string_db, matched_gene_ids, gene_identityN))
  
  # Close PDF device
  dev.off()
  
  return(matched_gene_ids)
}
