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

plot_top_enrichments <- function(filtered_enrichment, cat, gene_identity_Q) {
  # Select top 10 enrichments
  top_enrichments <- filtered_enrichment %>%
    arrange(p_value) %>%
    head(10)
  
  # Wrap long descriptions into multiple lines
  top_enrichments <- top_enrichments %>%
    mutate(description = str_wrap(description, width = 40)) # Adjust width as needed
  
  # Plot
  p <- ggplot(top_enrichments, aes(x = p_value, y = reorder(description, p_value))) +
    geom_point(aes(size = number_of_genes, color = -log10(fdr)), alpha = 0.7) +
    scale_color_gradient(low = "#83d0cb", high = "#145277", name = "FDR (-log10)") +
    scale_size_continuous(name = "Gene count", range = c(3, 10)) +
    scale_x_log10() +
    labs(
      title = paste(gene_identity_Q, "- Top 10 Enrichments -", cat),
      x = "P-value",
      y = "Description"
    ) +
    theme_minimal(base_size = 8) + # Reduced base font size
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12), # Slightly larger title
      legend.position = "right",
      axis.text.y = element_text(size = 10), # Adjust y-axis text size
      axis.text.x = element_text(size = 10), # Adjust x-axis text size
      axis.title = element_text(size = 12) # Adjust axis title size
    )
  
  # Dynamically adjust plot height based on the number of rows
  plot_height <- 4 + 0.5 * nrow(top_enrichments) # Adjust scaling factor as needed
  
  # # Save the plot to a PDF with dynamic dimensions
  # pdf_filename <- paste0("enrichment_plot_for_matched_genes_between_all_neighbor_genes_and_your_query_genes_", gsub(" ", "_", cat), ".pdf")
  # ggsave(pdf_filename, p, width = 10, height = plot_height, limitsize = FALSE) # Adjust width and height as needed
  
  # Return the plot for display in RStudio or notebook
  return(p)
}