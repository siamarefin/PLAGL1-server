args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)
# test()

# print("ami jekhane")
setwd(here("api", "code", id, "micro"))
source("../../micro_functions.R")
load_and_install_libraries()


highlight_genes <- readRDS("rds/highlight_genes.rds")
topTable1 <- readRDS("rds/topTable1.rds")
Reference <- readRDS("rds/Reference.rds")


print(Reference)

if (nchar(highlight_genes) > 0) {
    gene_ids <- trimws(unlist(strsplit(highlight_genes, ",")))

    # Highlight the specified genes in the volcano plot
    highlight_data <- topTable1[rownames(topTable1) %in% gene_ids, ]
    if (nrow(highlight_data) > 0) {
        plot_volcano_with_highlight(topTable1, highlight_data, gene_ids, title = paste("Highlighted Volcano Plot"))
    } else {
        message("None of the entered gene IDs were found in the data.")
    }
}
