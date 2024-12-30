# Function to select an organism and initialize Ensembl biomart, and also select ID type
select_orgnsm <- function(csv_file, organism_name, id_type) {
  # Read the CSV file to get the available organisms
  available_organisms <- read.csv(csv_file)
  
  # Extract scientific names and dataset names
  Organisms <- available_organisms$Scientific_Name
  dataset_name <- available_organisms$Dataset
  
  # Show available organisms to the user
  cat("Available organisms:\n")
  print(Organisms)
  
  # Validate organism name
  organism_index <- match(organism_name, Organisms)
  
  if (is.na(organism_index)) {
    stop("Invalid organism name. Please enter a valid organism name from the list.")
  }
  
  # Validate ID type
  if (!(id_type %in% c("entrez", "ensembl"))) {
    stop("Invalid ID type. Please enter 'entrez' or 'ensembl'.")
  }
  
  # Select the corresponding dataset
  selected_organism <- dataset_name[organism_index]
  cat("Selected dataset:", selected_organism, "\n")
  
  # Initialize Ensembl biomart based on the organism index range
  if (organism_index >= 215 && organism_index <= 367) {
    ensembl_mart <- useMart(biomart = "plants_mart", dataset = selected_organism, host = "https://plants.ensembl.org")
    cat("Using Plants Mart with dataset:", selected_organism, "\n")
  } else {
    ensembl_mart <- useMart("ensembl", dataset = selected_organism)
    cat("Using Gene Mart with dataset:", selected_organism, "\n")
  }
  
  # Return the selected mart and ID type
  return(list(mart = ensembl_mart, id_type = id_type))
}


