# Map gene symbols to STRING IDs
map_genes_to_string_ids <- function(string_db, gene_symbols) {
  string_db$map(data.frame(gene = gene_symbols), "gene", removeUnmappedRows = FALSE)
}

# Plot network for a list of STRING IDs with a left-aligned title
plot_network <- function(string_db, string_ids, gene_identity) {
  string_db$plot_network(string_ids)  # Plot the network
  
  # Correcting title function by properly concatenating the text
  title_text <- paste("PPI of", gene_identity)  # Create a proper title string
  
  # Left-align title with a smaller font size
  mtext(title_text, side = 3, adj = 0, line = 1.2, col = "black", font = 2, cex = 0.8)  # Decreased font size
}
