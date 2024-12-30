# Load necessary source files

args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)

setwd(here("api", "code", id, "heatmap"))

source("../../1_Extract_LFC.R")
source("../../2_plot_heatmap.R")

###############################################################################

heatmap_data <- read.csv("files/Heatmap_data.csv", row.names = 1)

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
