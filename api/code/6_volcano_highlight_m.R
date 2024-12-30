create_volcano_plot <- function(data, gene_ids, log_fold_change_col = "log2FoldChange", padj_col = "padj", gene_id_col = "external_gene_name", output_file_path, id_type = "ensembl") {
  # Ensure the gene column exists
  if (!(gene_id_col %in% colnames(data))) {
    stop(paste("Column", gene_id_col, "not found in the data."))
  }

  data$gene <- data[[gene_id_col]]

  # Set nudge_y based on id_type
  nudge_y_value <- ifelse(id_type == "ensembl", 3.5, 0.04)

  # Create the plot
  plot <- ggplot(data, aes(x = .data[[log_fold_change_col]], y = -log10(.data[[padj_col]]))) +

    # Scatter plot points with color-coded regulation
    geom_point(
      aes(color = ifelse(.data[[log_fold_change_col]] > 1.0 & -log10(.data[[padj_col]]) > 1.3,
        "Upregulated",
        ifelse(.data[[log_fold_change_col]] < -1.0 & -log10(.data[[padj_col]]) > 1.3,
          "Downregulated", "Not Significant"
        )
      )),
      size = 2.5, alpha = 0.5
    ) +

    # Add horizontal and vertical dashed lines
    geom_hline(yintercept = 1.3, linetype = "dashed", color = "black") +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +

    # Customize plot labels and add the title
    labs(
      title = "Volcano Plot",
      x = "Log2 Fold Change",
      y = "-log10(padj)",
      color = "Regulation"
    ) +

    # Customize color palette for regulation categories
    scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +

    # Minimal theme
    theme_minimal() +

    # Add borders around specific genes and label them
    geom_point(
      data = subset(data, gene %in% gene_ids),
      aes(x = .data[[log_fold_change_col]], y = -log10(.data[[padj_col]])),
      color = "black", size = 2, shape = 0, stroke = 1
    ) + # Circle with black border and no fill
    geom_text(
      data = subset(data, gene %in% gene_ids),
      aes(x = .data[[log_fold_change_col]], y = -log10(.data[[padj_col]]), label = gene),
      vjust = -1, color = "black", size = 3, fontface = "bold"
    ) # Adjusted text parameters

  # Print the plot
  print(plot)

  # remove the files/ from output file path
  output_file_path <- gsub("files/", "", output_file_path)
  # remove the .csv from output file path
  # output_file_path <- gsub(".csv", "", output_file_path)

  # Save the plot in different formats using the output_file_path
  ggsave(paste0("figures/Volcano_", output_file_path, ".tif"), plot, width = 10, height = 8, dpi = 300)
  ggsave(paste0("figures/Volcano_", output_file_path, ".pdf"), plot, width = 10, height = 8)
  ggsave(paste0("figures/Volcano_", output_file_path, ".png"), plot, width = 10, height = 8, dpi = 300)
  ggsave(paste0("figures/Volcano_", output_file_path, ".jpg"), plot, width = 10, height = 8, dpi = 300)
}
