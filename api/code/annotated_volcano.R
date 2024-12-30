################################################################################
# Source external R scripts containing necessary functions


args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)

setwd(here("api", "code", id, "annotation"))
# source("../../1_Biomart_init_m.R")
# source("../../2_Select_Organism_m.R")
# source("../../3_annotate_save_m.R")

source("../../5_Up_Down_m.R")
source("../../6_volcano_highlight_m.R")


gene_ids <- readRDS("rds/gene_ids.rds")
output_file_paths <- readRDS("rds/output_file_paths.rds")
csv_file_paths <- readRDS("rds/csv_file_paths.rds")

# Create a list of specific gene IDs (can be one or multiple)
# gene_ids <- c(
# "CCND1", "HSD17B10", "LPAR3", "SNRPC", "HSD17B4", "TBC1D4", "CTSD", "ZNF451", "GATA6",
# "PLAGL1", "ICAM5", "RRAD", "ZNF212", "DHX38", "GDF9", "BCL10", "MGRN1", "SLC9A8",
# "PHTF1", "NUP58", "PIK3CD", "NFAT5", "RFPL3S", "CARS1", "UBXN7", "DIO2", "PKNOX1",
# "MSI1", "ZNF266", "SZT2", "DMPK", "NEDD4L", "FOSL1", "SIN3B"
# ) # Take the input from the user as needed

# Loop through each output file path along with the corresponding original CSV file path
for (i in seq_along(output_file_paths)) {
    # Check if the output file exists before trying to read it
    if (file.exists(output_file_paths[i])) {
        # Load the annotated data
        data <- read.csv(output_file_paths[i])

        # Save regulated genes from the data using the original csv_file_paths
        save_regulated_genes(data, csv_file_paths[i]) # Use the original CSV file path

        # Create the volcano plot using the loaded data and specified gene IDs
        create_volcano_plot(data, gene_ids, output_file_path = output_file_paths[i])
    } else {
        warning(paste("File does not exist:", output_file_paths[i]))
    }
}

saveRDS(output_file_paths, "rds/output_file_paths.rds")
