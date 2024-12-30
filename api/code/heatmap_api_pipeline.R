# Load necessary source files

args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)

setwd(here("api", "code", id, "heatmap"))

source("../../1_Extract_LFC.R")
source("../../2_plot_heatmap.R")


gene_list <- readRDS("rds/gene_list.rds")
# take all annotated_resLFC files
csv_files <- readRDS("rds/csv_files.rds")
column_names <- readRDS("rds/column_names.rds")

# Call the function
result <- extract_lfc_multiple(gene_list, csv_files, column_names)


# Save the result to a CSV file with a timestamp
timestamp <- format(Sys.time(), "%Y.%m.%d_%H.%M.%S")
file_name <- paste0("files/Heatmap_data.csv")
write.csv(result, file = file_name, row.names = FALSE)

cat("Data extraction complete. Saved to", file_name, "\n")




###############################################################################

# either this
# Convert to matrix and remove the 'Gene' column
heatmap_data <- as.matrix(result[, -1])
rownames(heatmap_data) <- result$Gene

if (file.exists("files/Heatmap_annotation.csv")) {
  ann_df <- read.csv("files/Heatmap_annotation.csv", row.names = 1)
} else {
  print("heatmap annotation file not found")
}


# then run
# Check if 'ann_df' exists and has content before calling the function
if (exists("ann_df") && !is.null(ann_df) && nrow(ann_df) > 0) {
  create_heatmap(heatmap_data, ann_df, save_path = "figures/heatmap_output")
} else {
  create_heatmap(heatmap_data, save_path = "figures/heatmap_output")
}
