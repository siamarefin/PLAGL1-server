# Function to check and install annotation libraries
install_annotation_libs <- function(libs) {
  for (lib in libs) {
    if (!requireNamespace(lib, quietly = TRUE)) {
      message(paste("Installing", lib, "..."))
      BiocManager::install(lib)
    } else {
      message(paste(lib, "is already installed."))
    }
  }
}

# Example usage with biomaRt
install_annotation_libs(c("biomaRt"))

# Load the biomaRt package
library(biomaRt)