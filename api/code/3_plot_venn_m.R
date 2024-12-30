# Function to install and load required packages
install_and_load <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package)
  }
  library(package, character.only = TRUE)
}

# Function to create Venn diagram, compute intersections, and save results
plot_and_save_venn <- function(gene_lists, reg) {
  # Check if at least two lists are provided
  if (length(gene_lists) < 2) {
    message("You need at least two gene lists to generate a Venn diagram and compute intersections.")
    return(NULL)
  }

  # Install and load required packages
  install_and_load("ggplot2")
  install_and_load("ggVennDiagram")

  # Generate the Venn diagram
  venn_plot <- ggVennDiagram(gene_lists, label = "count")

  # Save the plot in different formats with reg in the filename
  ggsave(paste0("figures/venn_diagram_", reg, ".png"), plot = venn_plot, width = 8, height = 6)
  ggsave(paste0("figures/venn_diagram_", reg, ".pdf"), plot = venn_plot, width = 8, height = 6)
  ggsave(paste0("figures/venn_diagram_", reg, ".tiff"), plot = venn_plot, width = 8, height = 6)
  ggsave(paste0("figures/venn_diagram_", reg, ".jpg"), plot = venn_plot, width = 8, height = 6)
}
