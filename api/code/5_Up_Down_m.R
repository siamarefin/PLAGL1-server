# Install ggplot2 if not already installed
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}

# Load the ggplot2 library
library(ggplot2)


# Combined function to get user input and save regulated genes as CSV files
save_regulated_genes <- function(resLFC, X) {
  # Get user input for column names
  log_fold_change_col <- "log2FoldChange"

  padj_col <- "padj"

  # Upregulated genes
  Upregulated <- resLFC[resLFC[[log_fold_change_col]] > 1 &
    resLFC[[padj_col]] < 0.05, ]
  Upregulated_padj <- Upregulated[order(Upregulated[[padj_col]]), ]
  write.csv(Upregulated_padj, file = paste0("files/annotated_Upregulated_padj_", X), row.names = FALSE)

  # Downregulated genes
  Downregulated <- resLFC[resLFC[[log_fold_change_col]] < -1 &
    resLFC[[padj_col]] < 0.05, ]
  Downregulated_padj <- Downregulated[order(Downregulated[[padj_col]]), ]
  write.csv(Downregulated_padj, file = paste0("files/annotated_Downregulated_padj_", X), row.names = FALSE)
}
