find_all_clusters <- function(gene_IDs_for_clustering, clustering_method, output_dir){
  gene_IDs_fc_mapped <- map_genes_to_string_ids(string_db, gene_IDs_for_clustering)
  clustering_IDs <- gene_IDs_fc_mapped$STRING_id
  # get clusters
  clustersList <- string_db$get_clusters(clustering_IDs, clustering_method)
  # Check if clustersList is null
  if (is.null(clustersList)) {
    message("No clusters found for the given input.")
    return()
  }
  # Plot each cluster
  for (i in seq_along(clustersList)){
    pdf_filename <- file.path(output_dir, paste0("cluster_", i, "_", clustering_method, ".pdf"))
    pdf(pdf_filename)
    string_db$plot_network(clustersList[[i]])
    
    # Create a proper title string and add it to the plot
    title_text <- paste("cluster-", i)
    mtext(title_text, side = 3, adj = 0, line = 1.2, col = "black", font = 2, cex = 0.8)
    
    dev.off()
    
    # Map cluster genes back to symbols and save as CSV
    cg_symbols <- map_reverse_to_symbols(string_db, clustersList[[i]])
    csv_filename <- file.path(output_dir, paste0("clusterP_", i, "_", clustering_method, ".csv"))
    write.csv(cg_symbols, csv_filename, row.names = FALSE)
  }
}
