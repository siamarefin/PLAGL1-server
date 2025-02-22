# Load required libraries
library(ggplot2)
library(dplyr)

# Function to perform enrichment analysis
perform_enrichment <- function(string_db, string_ids) {
  string_db$get_enrichment(string_ids)
}

# Function to find enrichment results
enrichment_finder <- function(string_db, ids) {
  enrichment <- perform_enrichment(string_db, ids)
  return(enrichment)
}

# Function to extract all unique enrichment categories
extract_all_enrichments_categories <- function(enrichment) {
  unique_categories <- unique(enrichment$category)
  print("The available categories of enrichments are:")
  print(unique_categories)
  return(unique_categories)
}

# Function to filter enrichment results by a specific category
filter_enrichments_by_category <- function(enrichment, cat) {
  filtered_enrichment <- enrichment %>%
    filter(category == cat) %>%
    arrange(fdr)
  return(filtered_enrichment)
}

# Function to plot the top 10 enriched pathways
plot_top_enrichments <- function(filtered_enrichment, cat, gene_identity) {
  top_enrichments <- filtered_enrichment %>%
    arrange(p_value) %>%
    head(10)
  
  ggplot(top_enrichments, aes(x = p_value, y = reorder(description, p_value))) +
    geom_point(aes(size = number_of_genes, color = -log10(fdr)), alpha = 0.7) +
    scale_color_gradient(low = "lightblue", high = "darkblue", name = "FDR (-log10)") +
    scale_size_continuous(name = "Gene count", range = c(3, 10)) +
    scale_x_log10() +
    labs(
      title = paste("Top 10 Enrichments -", cat, "for", gene_identity),
      x = "P-value (log scale)",
      y = "Description"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "right",
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10)
    )
}
