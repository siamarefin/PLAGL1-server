# Function to load and install required libraries
load_and_install_libraries <- function() {
  required_libraries <- c("readr", "limma", "umap", "ggplot2", "Rtsne", "ape")
  
  # Function to check if a package is installed, and install it if not
  install_if_missing <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
    }
    library(pkg, character.only = TRUE)
  }
  # Iterate over the list of libraries and install if missing
  invisible(lapply(required_libraries, install_if_missing))
}


# Function to load and preprocess data
load_and_preprocess_data <- function(count_file, metadata_file) {
  count_data <- read.csv(count_file, row.names = 1, header = TRUE)
  count_data <- as.data.frame(lapply(count_data, as.numeric), row.names = rownames(count_data))
  
  sample_info <- read.csv(metadata_file, row.names = 1, header = TRUE)
  sample_list <- row.names(sample_info)
  count_data_subset <- count_data[, sample_list]
  
  return(list(count_data_subset = count_data_subset, sample_info = sample_info, sample_list = sample_list))
}

# Function to complete the cases
complete_cases_fx <- function(count_data_subset) {
  count_data_subset_cc <- count_data_subset[complete.cases(count_data_subset), ]
  return(count_data_subset_cc)
}


# Function to perform Log2 Transformation and Normalization
normalize_data <- function(count_data_subset) {
  
  qx <- as.numeric(quantile(count_data_subset, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm = TRUE))
  
  if ((qx[5] > 100) || (qx[6] - qx[1] > 50 && qx[2] > 0)) {
    count_data_subset[count_data_subset <= 0] <- NaN
    count_data_subset_log2 <- log2(count_data_subset)
    count_data_normalized <- normalizeBetweenArrays(count_data_subset_log2, method = "quantile")
  }
  else{
    count_data_normalized <- normalizeBetweenArrays(count_data_subset, method = "quantile")
  }
  
  # write.csv(count_data_normalized, "normalized_data_8vs8_direct.csv")
  return(count_data_normalized)
}


# Function to generate UMAP Plot
plot_umap <- function(count_data_subset, sample_info, title = "UMAP Plot") {
  set.seed(123)
  num_samples <- ncol(count_data_subset)
  umap_result <- umap(t(count_data_subset), n_neighbors = min((num_samples / 1.5), num_samples - 1), min_dist = 0.5)
  
  umap_df <- data.frame(X1 = umap_result$layout[, 1], X2 = umap_result$layout[, 2], sample_info)
  
  umap_plot <- ggplot(umap_df, aes(x = X1, y = X2, color = group, label = row.names(sample_info))) +
    geom_point(size = 3) + 
    geom_text(aes(y = X2 - 0.02), size = 3, hjust = 0.5, vjust = 1) +  # Adjusted to place text below the points
    labs(title = title, x = "UMAP 1", y = "UMAP 2") +
    theme_minimal() +
    theme(legend.position = "right")
  
  print(umap_plot)
}

# Function to generate t-SNE Plot
plot_tsne <- function(count_data_subset, sample_info, title = "t-SNE Plot") {
  set.seed(123)
  tsne_result <- Rtsne(t(count_data_subset), dims = 2, perplexity = 1)
  tsne_data <- data.frame(X = tsne_result$Y[, 1], Y = tsne_result$Y[, 2], sample_info)
  
  tsne_plot <- ggplot(tsne_data, aes(x = X, y = Y, color = group, label = row.names(sample_info))) +
    geom_point(size = 2) +
    geom_text(aes(y = Y - 0.02), size = 3, hjust = 0.5, vjust = 1) +  # Adjusted to place text below the points
    theme_minimal() +
    labs(title = title, x = "t-SNE 1", y = "t-SNE 2") +
    theme(legend.position = "right")
  
  print(tsne_plot)
}

# Function to generate PCA Plot
plot_pca <- function(count_data_subset, sample_info, title = "PCA Plot") {
  set.seed(123)
  pca <- prcomp(t(count_data_subset), scale. = FALSE)
  
  pca_data <- as.data.frame(pca$x)
  pca.var <- pca$sdev^2
  pca.var.percent <- round(pca.var / sum(pca.var) * 100, digits = 2)
  pca_data <- cbind(pca_data, sample_info)
  
  pca_plot <- ggplot(pca_data, aes(PC1, PC2, color = group)) +
    geom_point(size = 2) +
    geom_text(aes(label = row.names(sample_info), y = PC2 - 0.02), size = 3, hjust = 0.5, vjust = 1) +  # Adjust text below points
    labs(x = paste0('PC1: ', pca.var.percent[1], ' %'),
         y = paste0('PC2: ', pca.var.percent[2], ' %'),
         title = title) +
    theme_minimal() +
    theme(legend.position = "right")
  
  print(pca_plot)
}


# Function to plot Phylogenetic Tree
plot_phylo_tree <- function(count_data_subset, title = "Phylogenetic Tree") {
  set.seed(123)
  dist_matrix <- dist(t(count_data_subset))
  hc <- hclust(dist_matrix, method = "average")
  phylo_tree <- as.phylo(hc)
  
  plot.phylo(phylo_tree, type = "phylogram", tip.color = "blue", cex = 0.8, main = title)
}

# Function to plot Phylogenetic Tree with color-coded sample names based on group
plot_phylo_tree <- function(count_data_subset, sample_info, title = "Phylogenetic Tree") {
  set.seed(123)
  dist_matrix <- dist(t(count_data_subset))
  hc <- hclust(dist_matrix, method = "average")
  phylo_tree <- as.phylo(hc)
  
  # Get group information
  group_colors <- as.factor(sample_info$group)
  tip_colors <- as.numeric(group_colors)
  
  # Plot the phylogenetic tree with color-coded tip labels
  plot.phylo(phylo_tree, type = "phylogram", 
             tip.color = tip_colors, 
             cex = 0.8, 
             main = title)
  
  # Add a legend for the group colors
  legend("topleft", 
         legend = levels(group_colors), 
         col = 1:length(levels(group_colors)), 
         pch = 19, 
         cex = 0.8, 
         bty = "n",   # No border
         bg = "transparent")  # Transparent background
}



# Function to plot K-Means Clustering
plot_kmeans <- function(count_data_subset, sample_info, num_clusters = 3, title = "K-Means Clustering Plot") {
  
  # Perform K-means clustering
  kmeans_result <- kmeans(t(count_data_subset), centers = num_clusters)
  
  # Create a data frame with the clustering results
  kmeans_df <- data.frame(PC1 = prcomp(t(count_data_subset))$x[,1],
                          PC2 = prcomp(t(count_data_subset))$x[,2],
                          cluster = as.factor(kmeans_result$cluster),
                          sample_info)
  
  # Plot the K-means clusters
  kmeans_plot <- ggplot(kmeans_df, aes(x = PC1, y = PC2, color = cluster, label = row.names(sample_info))) +
    geom_point(size = 3) +
    geom_text(hjust = 0, vjust = 1) +
    labs(title = title, x = "PC1", y = "PC2") +
    theme_minimal()
  
  print(kmeans_plot)
}


# Function to generate boxplot
plot_boxplot <- function(count_data_subset, sample_info, title = "Boxplot") {
  group_colors <- as.factor(sample_info$group)
  palette(c("#1B9E77", "#D95F02", "#7570B3"))
  
  boxplot(count_data_subset, outline = FALSE, las = 2, 
          main = title,
          xlab = "Samples", ylab = "Expression levels",
          col = palette()[group_colors],  
          names = row.names(sample_info))
}

# Function to plot the P-value distribution
plot_pvalue_distribution <- function(topTable1, contrast_name) {
  hist(topTable1$adj.P.Val, 
       col = "#B32424", 
       border = "white", 
       xlab = "Adjusted P-value",
       ylab = "Number of Genes", 
       main = paste("Adjusted P-value Distribution: ", contrast_name))
}

# Function to generate Volcano Plot
plot_volcano <- function(topTable1, title = "Volcano Plot") {
  # Mark the regulated genes
  UP_Genes <- topTable1[topTable1$logFC > 1 & topTable1$adj.P.Val < 0.05, ]
  Down_Genes <- topTable1[topTable1$logFC < -1 & topTable1$adj.P.Val < 0.05, ]
  
  combined_data_adj <- rbind(
    transform(UP_Genes, Regulation = "Upregulated"),
    transform(Down_Genes, Regulation = "Downregulated"),
    transform(topTable1[!(rownames(topTable1) %in% rownames(UP_Genes) | rownames(topTable1) %in% rownames(Down_Genes)),], Regulation = "Not Significant")
  )
  
  # Create volcano plot
  volcano_plot <- ggplot(combined_data_adj, aes(x = logFC, y = -log10(adj.P.Val))) +
    geom_point(aes(color = Regulation), size = 2, alpha = 0.7) +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "red") +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "red") +
    labs(
      title = title,
      x = "Log2 Fold Change",
      y = "-log10(adj.P-Value)",
      color = "Regulation"
    ) +
    scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "grey")) +
    theme_minimal()
  
  print(volcano_plot)
}

# Function to generate Volcano Plot with Highlighted Genes
plot_volcano_with_highlight <- function(topTable1, highlight_data, gene_ids, title = "Volcano Plot with Highlights") {
  # Mark the regulated genes
  UP_Genes <- topTable1[topTable1$logFC > 1 & topTable1$adj.P.Val < 0.05, ]
  Down_Genes <- topTable1[topTable1$logFC < -1 & topTable1$adj.P.Val < 0.05, ]
  
  combined_data_adj <- rbind(
    transform(UP_Genes, Regulation = "Upregulated"),
    transform(Down_Genes, Regulation = "Downregulated"),
    transform(topTable1[!(rownames(topTable1) %in% rownames(UP_Genes) | rownames(topTable1) %in% rownames(Down_Genes)),], Regulation = "Not Significant")
  )
  
  # Create volcano plot
  volcano_plot <- ggplot(combined_data_adj, aes(x = logFC, y = -log10(adj.P.Val))) +
    geom_point(aes(color = Regulation), size = 2, alpha = 0.7) +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "red") +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "red") +
    labs(
      title = title,
      x = "Log2 Fold Change",
      y = "-log10(adj.P-Value)",
      color = "Regulation"
    ) +
    scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "grey")) +
    theme_minimal()
  
  # Highlight specific genes by circling them
  if (nrow(highlight_data) > 0) {
    # Draw the circle around the highlighted points
    volcano_plot <- volcano_plot +
      geom_point(data = highlight_data, aes(x = logFC, y = -log10(adj.P.Val)), color = "black", size = 4, shape = 1, stroke = 1.5) + # Circle
      geom_text(data = highlight_data, aes(x = logFC, y = -log10(adj.P.Val), label = gene_ids), vjust = -1, color = "black")
  }
  
  print(volcano_plot)
}



# Function to detect and remove outliers
remove_outliers <- function(count_data_subset, sample_info) {
  # Display the available sample names
  cat("Available sample IDs:\n")
  cat(row.names(sample_info), sep = ", ")
  
  # Prompt the user to enter outliers
  outliers_input <- readline(prompt = "\nEnter the sample IDs to remove, separated by commas: ")
  outliers <- unlist(strsplit(outliers_input, ","))
  outliers <- trimws(outliers)  # Trim any leading/trailing spaces
  
  # Check if the entered sample IDs are valid
  invalid_outliers <- outliers[!(outliers %in% row.names(sample_info))]
  if (length(invalid_outliers) > 0) {
    stop(paste("Invalid sample IDs entered:", paste(invalid_outliers, collapse = ", ")))
  }
  
  if (length(outliers) > 0) {
    count_data_clean <- count_data_subset[, !colnames(count_data_subset) %in% outliers]
    # count_data_normalized_clean <- count_data_normalized[, !colnames(count_data_normalized) %in% outliers]
    count_data_clean <- complete_cases_fx(count_data_clean)
    count_data_normalized_clean <- normalize_data(count_data_clean)
    sample_info_clean <- sample_info[!row.names(sample_info) %in% outliers, , drop = FALSE]
    return(list(count_data_clean = count_data_clean, 
                count_data_normalized_clean = count_data_normalized_clean, 
                sample_info_clean = sample_info_clean))
  }
  
  return(NULL)
}


# Differential Expression Function
perform_differential_expression <- function(count_data_subset, sample_info, group_col = "group") {
  group <- sample_info[, 1]
  groups <- factor(group)
  design <- model.matrix(~0 + groups)
  colnames(design) <- levels(groups)
  
  fit <- lmFit(count_data_subset, design)
  condition <- unique(group)
  
  # Display the available conditions (reference options)
  cat("Available conditions:\n")
  cat(condition, sep = ", ")
  
  # Ask the user to input the reference condition
  Reference <- readline(prompt = "\nEnter the reference condition: ")
  
  # Check if the entered reference condition is valid
  if (!(Reference %in% condition)) {
    stop("Invalid reference condition. Please enter one of the listed conditions.")
  }
  
  # Select the treatment condition(s)
  Treatment <- condition[condition != Reference]
  
  for (treat in Treatment) {
    cts <- paste(treat, Reference, sep = "-")
    message("Contrast: ", cts)
    
    cont.matrix <- makeContrasts(contrasts = cts, levels = design)
    fit2 <- contrasts.fit(fit, cont.matrix)
    fit2 <- eBayes(fit2)
    
    #resLFC
    topTable1 <- topTable(fit2, adjust = "fdr", number = Inf)
    write.csv(topTable1, "LFC.csv")
    
    # Call function to plot adjusted P-value distribution
    plot_pvalue_distribution(topTable1, contrast_name = cts)
    
    # Mark the regulated genes
    UP_Genes <- topTable1[topTable1$logFC > 1 & topTable1$adj.P.Val < 0.05, ]
    Down_Genes <- topTable1[topTable1$logFC < -1 & topTable1$adj.P.Val < 0.05, ]
    
    message("Number of Upregulated Genes (logFC > 1, adj.P.Val < 0.05): ", nrow(UP_Genes))
    message("Number of Downregulated Genes (logFC < -1, adj.P.Val < 0.05): ", nrow(Down_Genes))
    
    up_file <- paste0("Upregulated_Genes_", treat, "_vs_", Reference, ".csv")
    down_file <- paste0("Downregulated_Genes_", treat, "_vs_", Reference, ".csv")
    
    write.csv(UP_Genes, up_file)
    write.csv(Down_Genes, down_file)
    
    
    # Plot volcano plot for each contrast
    plot_volcano(topTable1, title = paste("Volcano Plot: ", treat, " vs ", Reference))
    
    # Ask the user if they want to highlight any specific gene(s)
    highlight_genes <- readline(prompt = "Highlight any specific gene(s)? Enter the gene IDs separated by commas, or press Enter to skip: ")
    
    if (nchar(highlight_genes) > 0) {
      gene_ids <- trimws(unlist(strsplit(highlight_genes, ",")))
      
      # Highlight the specified genes in the volcano plot
      highlight_data <- topTable1[rownames(topTable1) %in% gene_ids, ]
      if (nrow(highlight_data) > 0) {
        plot_volcano_with_highlight(topTable1, highlight_data, gene_ids, title = paste("Highlighted Volcano Plot: ", treat, " vs ", Reference))
      } else {
        message("None of the entered gene IDs were found in the data.")
      }
    }
  }
}


##################################### Main Workflow ##############################

# load and install libraries
load_and_install_libraries()


#Load data
data_files <- load_and_preprocess_data("Full_Expression_11234.csv", "Metadata_11234_F.csv")
count_data_subset <- data_files$count_data_subset
sample_info <- data_files$sample_info

count_data_subset_cc <- complete_cases_fx(count_data_subset)

# Apply normalization
count_data_normalized <- normalize_data(count_data_subset_cc)
#write.csv(count_data_normalized, "4vs8_clean_normalized.csv")
# Generate figures before normalization
#Umap
plot_umap(count_data_subset_cc, sample_info, title = "UMAP Plot (Before Normalization)")
plot_umap(count_data_normalized, sample_info, title = "UMAP Plot (After Normalization)")

#t-SNE
plot_tsne(count_data_subset_cc, sample_info, title = "t-SNE Plot (Before Normalization)")
plot_tsne(count_data_normalized, sample_info, title = "t-SNE Plot (After Normalization)")

#PCA
plot_pca(count_data_subset_cc, sample_info, title = "PCA Plot (Before Normalization)")
plot_pca(count_data_normalized, sample_info, title = "PCA Plot (After Normalization)")

#Phylogenetic tree
plot_phylo_tree(count_data_subset_cc, sample_info, title = "Phylogenetic Tree (Before Normalization)")
plot_phylo_tree(count_data_normalized, sample_info, title = "Phylogenetic Tree (After Normalization)")

#K-Means Clustering
plot_kmeans(count_data_subset_cc, sample_info, num_clusters = 2, title = "K-Means Clustering (Before Normalization)")
plot_kmeans(count_data_normalized, sample_info, num_clusters = 2, title = "K-Means Clustering (After Normalization)")


#Boxplot
plot_boxplot(count_data_subset_cc, sample_info, title = "Boxplot (Before Normalization)")
plot_boxplot(count_data_normalized, sample_info, title = "Boxplot (After Normalization)")




# Remove outliers
outlier_removal <- remove_outliers(count_data_subset, sample_info)

if (!is.null(outlier_removal)) {
  count_data_subset_clean <- outlier_removal$count_data_clean
  # write.csv(count_data_subset_clean, "4vs8_clean.csv")
  sample_info_clean <- outlier_removal$sample_info_clean
  
  count_data_subset_clean_normalized <- outlier_removal$count_data_normalized_clean
  
  #Umap
  plot_umap(count_data_subset_clean, sample_info_clean, title = "UMAP Plot (After Outlier Removal)")
  plot_umap(count_data_subset_clean_normalized, sample_info_clean, title = "UMAP Plot (After Outlier Removal(Normalized))")
  
  #t-SNE
  plot_tsne(count_data_subset_clean, sample_info_clean, title = "t-SNE Plot (After Outlier Removal)")
  plot_tsne(count_data_subset_clean_normalized, sample_info_clean, title = "t-SNE Plot (After Outlier Removal(Normalized))")
  
  #PCA
  plot_pca(count_data_subset_clean, sample_info_clean, title = "PCA Plot (After Outlier Removal)")
  plot_pca(count_data_subset_clean_normalized, sample_info_clean, title = "PCA Plot (After Outlier Removal(Normalized))")
  
  #Phylogenetic Tree
  plot_phylo_tree(count_data_subset_clean,sample_info_clean, title = "Phylogenetic Tree (After Outlier Removal)")
  plot_phylo_tree(count_data_subset_clean_normalized,sample_info_clean, title = "Phylogenetic Tree (After Outlier Removal(Normalized))")
  
  # Boxplot
  plot_boxplot(count_data_subset_clean, sample_info_clean, title = "Boxplot (After Outlier Removal)")
  plot_boxplot(count_data_subset_clean_normalized, sample_info_clean, title = "Boxplot (After Outlier Removal(Normalized))")
  
  
  perform_differential_expression(count_data_subset_clean_normalized, sample_info_clean)
} else {
  perform_differential_expression(count_data_normalized, sample_info)
}

#write.csv(count_data_subset_clean_normalized,"4vs8_Cleaned.csv")

# 
# write.csv(UP_Genes, up_file)
# write.csv(Down_Genes, down_file)
# write.csv(topTable1, "LFC_3.csv")







