args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)

setwd(here("api", "code", id, "annotation"))
source("../../1_Biomart_init_m.R")
source("../../2_Select_Organism_m.R")
source("../../3_annotate_save_m.R")

organism_name <- readRDS("rds/organism_name.rds")
id_type <- readRDS("rds/id_type.rds")
csv_file_paths <- readRDS("rds/csv_file_paths.rds")


ensembl_mart <- select_orgnsm("files/Organisms_name.csv", organism_name, id_type)

output_file_paths <- character(0)

for (csv_file_path in csv_file_paths) {
    # Define the output file path
    output_file_path <- paste0("annotated_", csv_file_path)

    csv_file_path <- paste0("files/", csv_file_path)
    output_file_path <- paste0("files/", output_file_path)

    # Annotate genes from the input CSV using the selected organism's Ensembl BioMart and save the result
    annotate_and_replace_ids(csv_file_path, output_file_path, ensembl_mart$mart, ensembl_mart$id_type)

    # Add the output file path to the list
    output_file_paths <- c(output_file_paths, output_file_path)
}


saveRDS(output_file_paths, "rds/output_file_paths.rds")
saveRDS(csv_file_paths, "rds/csv_file_paths.rds")
