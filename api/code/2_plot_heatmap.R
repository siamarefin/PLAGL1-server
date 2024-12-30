# Install necessary packages if not already installed
if (!requireNamespace("pheatmap", quietly = TRUE)) {
  install.packages("pheatmap")
}

if (!requireNamespace("RColorBrewer", quietly = TRUE)) {
  install.packages("RColorBrewer")
}

# Load necessary libraries
library(pheatmap)
library(RColorBrewer)

# Function to generate a random color palette based on unique groups
generate_color_palette <- function(groups) {
  num_groups <- length(unique(groups))
  palette <- grDevices::colors()[sample(1:length(grDevices::colors()), num_groups, replace = FALSE)]
  return(palette)
}

# Function to create a heatmap
create_heatmap <- function(heatmap_data, ann_df = NULL, 
                           cell_width = 24, cell_height = 10, 
                           fontsize_row = 8, fontsize_col = 8, 
                           save_path = NULL) {
  
  # Check if 'Gene' column exists and extract rownames accordingly
  if ("Gene" %in% colnames(heatmap_data)) {
    rownames(heatmap_data) <- heatmap_data$Gene
    heatmap_data <- heatmap_data[, -which(colnames(heatmap_data) == "Gene")]
  }
  
  # Convert to matrix
  heatmap_data <- as.matrix(heatmap_data)
  
  # Define color palette (from blue to red)
  color_palette <- colorRampPalette(c("#1a237e", "#f6f5fa", "#b71c1c"))(1000)
  
  # Calculate the maximum absolute value in the data for setting breaks
  max_abs_value <- max(abs(heatmap_data), na.rm = TRUE)
  
  # Define breaks for coloring
  breaks <- seq(-max_abs_value, max_abs_value, length.out = 1000)
  
  # Check if annotation data exists
  if (!is.null(ann_df) && nrow(ann_df) > 0) {
    unique_groups <- unique(ann_df$group)
    ann_colors <- generate_color_palette(unique_groups)
    annotation_colors <- list(group = setNames(ann_colors, unique_groups))
    
    # Create heatmap with annotations
    heatmap_plot <- pheatmap(heatmap_data,
                             col = color_palette,
                             border_color = "#696880",
                             breaks = breaks,
                             clustering_distance_rows = "euclidean",
                             clustering_distance_cols = "euclidean",
                             clustering_method = "complete",
                             cellwidth = cell_width,
                             cellheight = cell_height,
                             annotation_col = ann_df,
                             annotation_colors = annotation_colors,
                             angle_col = 45,
                             fontsize_row = fontsize_row,
                             fontsize_col = fontsize_col)
  } else {
    # Create heatmap without annotations
    heatmap_plot <- pheatmap(heatmap_data,
                             col = color_palette,
                             border_color = "#696880",
                             breaks = breaks,
                             clustering_distance_rows = "euclidean",
                             clustering_distance_cols = "euclidean",
                             clustering_method = "complete",
                             cellwidth = cell_width,
                             cellheight = cell_height,
                             angle_col = 45,
                             fontsize_row = fontsize_row,
                             fontsize_col = fontsize_col)
  }
  
  # Save the heatmap in different formats if a save path is provided
  if (!is.null(save_path)) {
    # Save as PDF
    pdf(file = paste0(save_path, ".pdf"), width = 10, height = 8)
    print(heatmap_plot)
    dev.off()
    
    # Save as PNG
    png(file = paste0(save_path, ".png"), width = 800, height = 600)
    print(heatmap_plot)
    dev.off()
    
    # Save as TIFF
    tiff(file = paste0(save_path, ".tiff"), width = 800, height = 600, res = 300)
    print(heatmap_plot)
    dev.off()
  }
}
