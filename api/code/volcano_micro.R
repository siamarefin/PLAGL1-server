args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)
# test()

# print("ami jekhane")
setwd(here("api", "code", id, "micro"))
source("../../micro_functions.R")
load_and_install_libraries()


count_data_subset_clean_normalized <- readRDS("rds/count_data_subset_clean_normalized.rds")
sample_info_clean <- readRDS("rds/sample_info_clean.rds")



perform_differential_expression(count_data_subset_clean_normalized, sample_info_clean)
