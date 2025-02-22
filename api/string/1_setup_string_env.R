install_and_load <- function(packages) {
  for (package in packages) {
    if (!require(package, character.only = TRUE)) {
      install.packages(package, dependencies = TRUE)
      library(package, character.only = TRUE)
    }
  }
}

setup_stringdb_environment <- function() {
  # Define required libraries
  required_packages <- c("STRINGdb", "dplyr", "readr", "ggplot2")
  
  # Install and load required packages
  install_and_load(required_packages)
  
  # Set timeout to 6000 seconds (100 minutes)
  options(timeout = 6000)
  
  # Inform the user that the setup is complete
  message("Libraries loaded and environment set up successfully.")
}

# Initialize STRINGdb object
initialize_stringdb <- function(species_unique_id, score_threshold = 150) {
  STRINGdb$new(version = "12.0", species = species_unique_id, score_threshold = score_threshold)
}

# Define the function to extract taxon_ID by Species Name
get_taxon_id <- function(data, species_name) {
  # Check if the expected columns exist
  if (!("Species_Name" %in% colnames(data)) || !("taxon_ID" %in% colnames(data))) {
    stop("Error: Expected columns 'Species_Name' and 'taxon_ID' not found in the dataset.")
  }

  # Filter the dataset for the given species name
  result <- data[data$Species_Name == species_name, "taxon_ID"]

  # Check if a match was found
  if (length(result) == 0) {
    stop(paste("Error: Species", species_name, "not found in the dataset."))
  } else {
    return(as.integer(result[1]))  # Convert to integer and return first match
  }
}