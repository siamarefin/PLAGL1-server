# Load necessary source files

args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)

setwd(here("api", "code", id, "venn"))

source("../../1_init_venn_m.R")
source("../../2_gene_list_m.R")
source("../../3_plot_venn_m.R")
source("../../4_wide_frame_venn_m.R")


reg <- readRDS("rds/reg.rds")
input_data <- readRDS("rds/input_data.rds")


input_data

gene_lists <- take_user_inputs(input_data)

plot_and_save_venn(gene_lists, reg) # This will save files with "down" in their names


# Call the function to create the Venn data frame
venn_result <- create_venn_dataframe(gene_lists)

# Concatenate to create the output file name
output_file <- paste0("files/", reg, "_venn_result.csv")
write.csv(venn_result, file = output_file, row.names = FALSE)
###############################################################################
