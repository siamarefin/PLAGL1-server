args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)
# test()

# print("ami jekhane")
setwd(here("api", "code", id, "micro"))
source("../../micro_functions.R")
load_and_install_libraries()

count_data_subset_cc <- readRDS("rds/count_data_subset_cc.rds")
count_data_subset <- readRDS("rds/count_data_subset.rds")
sample_info <- readRDS("rds/sample_info.rds")
count_data_normalized <- readRDS("rds/count_data_normalized.rds")
outliers <- readRDS("rds/outliers.rds")


# print(outliers[1])

outlier_removal <- remove_outliers(count_data_subset, sample_info, outliers)

print(outlier_removal$sample_info_clean)

if (!is.null(outlier_removal)) {
    count_data_subset_clean <- outlier_removal$count_data_clean
    # write.csv(count_data_subset_clean, "4vs8_clean.csv")
    sample_info_clean <- outlier_removal$sample_info_clean

    count_data_subset_clean_normalized <- outlier_removal$count_data_normalized_clean

    # Umap
    plot_umap(count_data_subset_clean, sample_info_clean, title = "UMAP Plot (Before Normalization)")
    plot_umap(count_data_subset_clean_normalized, sample_info_clean, title = "UMAP Plot (After Normalization)")

    # t-SNE
    plot_tsne(count_data_subset_clean, sample_info_clean, title = "t-SNE Plot (Before Normalization)")
    plot_tsne(count_data_subset_clean_normalized, sample_info_clean, title = "t-SNE Plot (After Normalization)")

    # PCA
    plot_pca(count_data_subset_clean, sample_info_clean, title = "PCA Plot (Before Normalization)")
    plot_pca(count_data_subset_clean_normalized, sample_info_clean, title = "PCA Plot (After Normalization)")

    # Phylogenetic Tree
    plot_phylo_tree(count_data_subset_clean, sample_info_clean, title = "Phylogenetic Tree (Before Normalization)")
    plot_phylo_tree(count_data_subset_clean_normalized, sample_info_clean, title = "Phylogenetic Tree (After Normalization)")

    # Boxplot
    plot_boxplot(count_data_subset_clean, sample_info_clean, title = "Boxplot (Before Normalization)")
    plot_boxplot(count_data_subset_clean_normalized, sample_info_clean, title = "Boxplot (After Normalization)")


    saveRDS(count_data_subset_clean, "rds/count_data_subset_clean.rds")
    saveRDS(sample_info_clean, "rds/sample_info_clean.rds")
    saveRDS(count_data_subset_clean_normalized, "rds/count_data_subset_clean_normalized.rds")
} else {
    saveRDS(count_data_subset, "rds/count_data_subset_clean.rds")
    saveRDS(sample_info, "rds/sample_info_clean.rds")
    saveRDS(count_data_normalized, "rds/count_data_subset_clean_normalized.rds")
    print("No outliers removed!")
}


# perform_differential_expression(count_data_subset_clean_normalized, sample_info_clean)

# perform_differential_expression(count_data_normalized, sample_info)
