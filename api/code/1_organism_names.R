args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)

setwd(here("api", "code", id, "annotation"))
source("../../1_Biomart_init_m.R")

available_organisms <- read.csv("files/Organisms_name.csv")

# Extract scientific names and dataset names
Organisms <- available_organisms$Scientific_Name
dataset_name <- available_organisms$Dataset

# Show available organisms to the user
cat("Available organisms:\n")
# print(Organisms)


saveRDS(Organisms, "rds/Organisms.rds")
