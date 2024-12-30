batch_effect_correction <- function(input_file, output_dir,user_id) {
  library(jsonlite)
  library(sva)  # For batch effect correction
  
  tryCatch({
    # Read and preprocess data
    merged_df_data <- read.csv(input_file, header = TRUE, row.names = 1)
    merged_df_data <- na.omit(merged_df_data)
    condition_info <- merged_df_data$condition
    data_t <- t(merged_df_data[, !(colnames(merged_df_data) %in% c('condition', 'batch'))])
    sample_names <- colnames(data_t)  # Save sample names for labeling
    
    # Batch effect correction with ComBat
    batch_info <- merged_df_data$batch
    data_combat <- ComBat(dat = as.matrix(data_t), batch = batch_info, par.prior = TRUE, prior.plots = FALSE)
    
    # Save corrected data
    output_file <- file.path(output_dir, paste0("batch_", basename(input_file)))
    data_corrected <- t(data_combat)
    data_corrected_with_condition <- cbind(condition = condition_info, data_corrected)
    write.csv(data_corrected_with_condition, output_file, row.names = TRUE)
    
    # Save boxplots in multiple formats
    plot_formats <- c("png", "jpg", "tif", "pdf")
    for (fmt in plot_formats) {
      file_name <- file.path(output_dir, paste0("batch_correction_boxplots.", fmt))
      if (fmt == "png") {
        png(file_name, width = 1200, height = 600)
      } else if (fmt == "jpg") {
        jpeg(file_name, width = 1200, height = 600)
      } else if (fmt == "tif") {
        tiff(file_name, width = 1200, height = 600)
      } else if (fmt == "pdf") {
        pdf(file_name, width = 12, height = 6)
      }
      par(mfrow = c(1, 2), mar = c(10, 5, 4, 2))
      boxplot(data_t, main = "Normalized Data", las = 2, col = "lightblue", outline = FALSE,
              ylab = "Expression Levels", cex.axis = 0.7, names = sample_names)
      boxplot(data_combat, main = "Batch Corrected Data", las = 2, col = "lightgreen",
              outline = FALSE, ylab = "Expression Levels", cex.axis = 0.7, names = sample_names)
      dev.off()
    }

    # Output completion message
    cat("Batch effect correction completed. Corrected data saved to:", output_file, "\n")
    cat("Boxplots saved in PNG, JPG, TIF, and PDF formats.\n")

  }, error = function(e) {
    # Handle errors gracefully
    cat("Error: ", e$message, "\n")
  })
}

# Example command-line arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_dir <- args[2]
user_id    <- args[3]
batch_effect_correction(input_file, output_dir,user_id)
