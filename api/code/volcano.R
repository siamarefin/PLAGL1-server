args <- commandArgs(trailingOnly = TRUE)
id <- args[1]

library(here)
library(DESeq2)
library(ggplot2)

print(id)

# Ensure the number of samples match


setwd(here("api", "code", id))


dds <- readRDS("rds/dds.rds")
X <- readRDS("rds/X.rds")
ref_level <- readRDS("rds/ref_level.rds")




resLFC <- lfcShrink(dds, coef = X, type = "apeglm")

# change resLFC to a dataframe
resLFC <- as.data.frame(resLFC)

# Researchers are often interested in minimizing the number of false discoveries.
# Only Keep the significant genes padj (FDR) values is less than 0.05
# resLFC_p_cut <- resLFC[resLFC$padj < 0.05,]

##************** remove the NA rows ************##

# create histogram plot of p-values
# save the plot as a png file



png("figures/histogram_pvalues.png")
hist(resLFC$padj,
    breaks = seq(0, 1, length = 21), col = "grey", border = "white",
    xlab = "", ylab = "", main = "Frequencies of padj-values"
)
dev.off()

pdf("figures/histogram_pvalues.pdf")
hist(resLFC$padj,
    breaks = seq(0, 1, length = 21), col = "grey", border = "white",
    xlab = "", ylab = "", main = "Frequencies of padj-values"
)
dev.off()

summary(resLFC)


############################################ UP and Downregulated Genes | NODE 5.1 ####################################################

# resLFC
write.csv(resLFC, file = paste0("files/resLFC_", X, ".csv"), row.names = TRUE)

# Upregulated genes
Upregulated <- resLFC[resLFC$log2FoldChange > 1 & resLFC$padj < 0.05, ]
Upregulated_padj <- Upregulated[order(Upregulated$padj), ]
write.csv(Upregulated_padj, file = paste0("files/Upregulated_padj_", X, ".csv"), row.names = TRUE)

# Downregulated genes
Downregulated <- resLFC[resLFC$log2FoldChange < -1 & resLFC$padj < 0.05, ]
Downregulated_padj <- Downregulated[order(Downregulated$padj), ]
write.csv(Downregulated_padj, file = paste0("files/Downregulated_padj_", X, ".csv"), row.names = TRUE)



################################################### Volcano Plot | NODE 6 ###########################################################

# Create a Volcano Plot
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
        title = "Volcano Plot", # Add the header here
        x = "Log2 Fold Change",
        y = "-log10(padj)",
        color = "Regulation"
    ) +

    # Customize color palette for regulation categories
    scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +

    # Use a minimal theme for the plot
    theme_minimal()

# Create a Volcano Plot

ggsave(plot, filename = "figures/volcano_plot.png")
ggsave(plot, filename = "figures/volcano_plot.pdf")

saveRDS(resLFC, file = "rds/resLFC.rds")

# library(ggplot2)

# # User will select specific gene names to label and highlight with borders
# genes_to_highlight <- c("ENSG00000231500", "ENSG00000204388", "ENSG00000081277")

# # Adding gene names as a column
# resLFC$gene <- row.names(resLFC)

# ggplot(resLFC, aes(x = log2FoldChange, y = -log10(padj))) +

#     # Scatter plot points with color-coded regulation
#     geom_point(
#         aes(color = ifelse(log2FoldChange > 1.0 & -log10(padj) > 1.3, "Upregulated",
#             ifelse(log2FoldChange < -1.0 & -log10(padj) > 1.3, "Downregulated", "Not Significant")
#         )),
#         size = 2.5, alpha = 0.5
#     ) +

#     # Add horizontal dashed line
#     geom_hline(yintercept = 1.3, linetype = "dashed", color = "black") +

#     # Add vertical dashed lines
#     geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +

#     # Customize plot labels and add the header
#     labs(
#         title = "Volcano Plot",
#         x = "Log2 Fold Change",
#         y = "-log10(padj)",
#         color = "Regulation"
#     ) +

#     # Customize color palette for regulation categories
#     scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +

#     # Use a minimal theme for the plot
#     theme_minimal() +

#     # Add borders around specific genes and label them
#     geom_point(
#         data = subset(resLFC, gene %in% genes_to_highlight),
#         aes(x = log2FoldChange, y = -log10(padj)),
#         shape = 21, size = 4, stroke = 1, color = "black", fill = NA
#     ) + # shape = 21 for a circle with a border
#     geom_text(
#         data = subset(resLFC, gene %in% genes_to_highlight),
#         aes(label = gene),
#         vjust = -0.5, hjust = 0.5, size = 3, color = "black"
#     )
