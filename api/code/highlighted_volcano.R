args <- commandArgs(trailingOnly = TRUE)
id <- args[1]

library(here)
library(DESeq2)
library(ggplot2)

print(id)

# Ensure the number of samples match


setwd(here("api", "code", id))


# User will select specific gene names to label and highlight with borders
genes_to_highlight <- readRDS("rds/genes_to_highlight.rds")
print(genes_to_highlight)
resLFC <- readRDS("rds/resLFC.rds")


# Adding gene names as a column
resLFC$gene <- row.names(resLFC)

plot <- ggplot(resLFC, aes(x = log2FoldChange, y = -log10(padj))) +

    # Scatter plot points with color-coded regulation
    geom_point(
        aes(color = ifelse(log2FoldChange > 1.0 & -log10(padj) > 1.3, "Upregulated",
            ifelse(log2FoldChange < -1.0 & -log10(padj) > 1.3, "Downregulated", "Not Significant")
        )),
        size = 2.5, alpha = 0.5
    ) +

    # Add horizontal dashed line
    geom_hline(yintercept = 1.3, linetype = "dashed", color = "black") +

    # Add vertical dashed lines
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +

    # Customize plot labels and add the header
    labs(
        title = "Volcano Plot",
        x = "Log2 Fold Change",
        y = "-log10(padj)",
        color = "Regulation"
    ) +

    # Customize color palette for regulation categories
    scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +

    # Use a minimal theme for the plot
    theme_minimal() +

    # Add borders around specific genes and label them
    geom_point(
        data = subset(resLFC, gene %in% genes_to_highlight),
        aes(x = log2FoldChange, y = -log10(padj)),
        shape = 21, size = 4, stroke = 1, color = "black", fill = NA
    ) + # shape = 21 for a circle with a border
    geom_text(
        data = subset(resLFC, gene %in% genes_to_highlight),
        aes(label = gene),
        vjust = -0.5, hjust = 0.5, size = 3, color = "black"
    )


ggsave(plot, filename = "figures/highlighted_volcano.png")
ggsave(plot, filename = "figures/highlighted_volcano.pdf")
