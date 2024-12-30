args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)


print(id)

# Ensure the number of samples match

# print(here())

setwd(here("api", "code", id))


user_data <- readRDS("rds/count_data.rds")
user_sample_info <- readRDS("rds/sample_info.rds")
genes_str <- readRDS("rds/genes.rds")

user_data <- user_data[, !colnames(user_data) %in% {
    genes_str
}]
user_sample_info <- user_sample_info[!rownames(user_sample_info) %in% {
    genes_str
}, , drop = FALSE]


saveRDS(user_data, "rds/count_data.rds")
saveRDS(user_sample_info, "rds/sample_info.rds")
