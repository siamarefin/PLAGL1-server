# Generate figures before normalization
# Umap
args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)
# test()
setwd(here("api", "code", id, "micro"))


source("../../micro_functions.R")

load_and_install_libraries()

count_data_subset_cc <- readRDS("rds/count_data_subset_cc.rds")
sample_info <- readRDS("rds/sample_info.rds")
count_data_normalized <- readRDS("rds/count_data_normalized.rds")

plot_umap(count_data_subset_cc, sample_info, title = "UMAP Plot (Before Normalization)")
plot_umap(count_data_normalized, sample_info, title = "UMAP Plot (After Normalization)")

# t-SNE
plot_tsne(count_data_subset_cc, sample_info, title = "t-SNE Plot (Before Normalization)")
plot_tsne(count_data_normalized, sample_info, title = "t-SNE Plot (After Normalization)")

# PCA
plot_pca(count_data_subset_cc, sample_info, title = "PCA Plot (Before Normalization)")
plot_pca(count_data_normalized, sample_info, title = "PCA Plot (After Normalization)")

# Phylogenetic tree
plot_phylo_tree(count_data_subset_cc, sample_info, title = "Phylogenetic Tree (Before Normalization)")
plot_phylo_tree(count_data_normalized, sample_info, title = "Phylogenetic Tree (After Normalization)")

# K-Means Clustering
plot_kmeans(count_data_subset_cc, sample_info, num_clusters = 2, title = "K-Means Clustering (Before Normalization)")
plot_kmeans(count_data_normalized, sample_info, num_clusters = 2, title = "K-Means Clustering (After Normalization)")


# Boxplot
plot_boxplot(count_data_subset_cc, sample_info, title = "Boxplot (Before Normalization)")
plot_boxplot(count_data_normalized, sample_info, title = "Boxplot (After Normalization)")


print("analysis complete")
