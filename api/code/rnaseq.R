########################################## Install Packages #######################################

# # Install BiocManager if not already installed
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# # Install Bioconductor packages
# BiocManager::install(c("WGCNA", "DESeq2"))
# 
# # Install CRAN packages
# install.packages(c("tidyverse", "Rtsne", "umap", "ggplot2"))


########################################## RNA-Seq Data ############################################

# Set Working Directory

########################################### Read Files | NODE 1 #############################################

# Load Data

# Expression Data
# load the count data
count_data <- read.csv("count_data.csv", header = TRUE, row.names = 1)

# Metadata
# load the sample info
sample_info <- read.csv("meta_data.csv", header = TRUE, row.names = 1)

# Ensure the number of samples match
if (ncol(count_data) != nrow(sample_info)) {
  stop("Number of samples in count_data and sample_info do not match!")
}

# Convert the Col name of user to "Treatment"
colnames(sample_info)[colnames(sample_info) == colnames(sample_info)] <- "Treatment"

# Convert Condition to factor
sample_info$Treatment <- factor(sample_info$Treatment)

##############################################################################################################

########################################### Screen Data Quality | NODE 2 ##############################################

# Dimension Reduction
# Data Quality
# Remove Outlier Samples

##############################################################################################################

# Quality Control: detect outlier genes
library(WGCNA)
gsg <- goodSamplesGenes(t(count_data)) # check the data format
summary(gsg)

# Remove outlier genes
data <- count_data[gsg$goodGenes == TRUE,]

################################################# PCA ########################################################

# Load Expression Data
# data <- read.csv("count_data.csv")

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

#***** Make A Loop *****#

# Check for NA or Infinite values
summary(data)
is.na(data)
# Replace NA and Infinite values with zero

data[is.na(data)] <- 0

# Replacing Infinite Values
data_list <- as.list(data)

for (name in names(data_list)) {
  data_list[[name]][is.infinite(data_list[[name]])] <- 0
}

data <- as.data.frame(data_list)

# Verify no NA or Infinite values remain
summary(data)

#***** Make A Loop *****#

# Remove non-numeric columns for PCA
# Remove the gene ID column
data_numeric <- data[, sapply(data, is.numeric)]
# data_numeric <- data[,1:12]

# Perform PCA
pca <- prcomp(t(data_numeric))

# View the PCA results
summary(pca)

# Prepare PCA data for plotting
pca.dat <- as.data.frame(pca$x)
pca.var <- pca$sdev^2
pca.var.percent <- round(pca.var / sum(pca.var) * 100, digits = 2)

# Merge PCA data with metadata
pca.dat <- cbind(pca.dat, sample_info)

library(ggplot2)
# Plot PCA with metadata groups
ggplot(pca.dat, aes(PC1, PC2, color = Treatment)) +
  geom_point() +
  geom_text(aes(label = rownames(pca.dat)), hjust = 0, vjust = 1) +
  labs(x = paste0('PC1: ', pca.var.percent[1], ' %'),
       y = paste0('PC2: ', pca.var.percent[2], ' %')) +
  theme_minimal() +
  theme(legend.title = element_blank())

################################################# UMAP #####################################################

# Load libraries
library(umap)

# Load Data
# data <- read.csv("count_data.csv")
# Remove non-numeric columns for Umap
# Remove the gene ID column
# data_numeric <- data[, sapply(data, is.numeric)]

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

# Set random seed for reproducibility
set.seed(123)

# Perform UMAP dimensionality reduction
umap_result <- umap(t(data_numeric), n_neighbors = 5, min_dist = 0.5)

# Extract UMAP coordinates and combine with metadata
umap_df <- data.frame(
  X1 = umap_result$layout[, 1],  # UMAP component 1
  X2 = umap_result$layout[, 2],  # UMAP component 2
  sample_info
)

library(ggplot2)
# Plot using ggplot2
ggplot(umap_df, aes(x = X1, y = X2, color = Treatment)) +
  geom_text(aes(label = rownames(sample_info)), hjust = 0, vjust = 1) +
  geom_point(size = 3) +
  labs(title = "UMAP",
       x = "UMAP 1", y = "UMAP 2") +
  theme_minimal()

################################################# t-SNE ###############################################

library(Rtsne)

# Load Data
# data <- read.csv("count_data.csv")
# Remove non-numeric columns for t-SNE
# Remove the gene ID column
# data_numeric <- data[, sapply(data, is.numeric)]

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

set.seed(123)
# Perform t-SNE
tsne_result <- Rtsne(t(data_numeric), dims=2, perplexity=3, verbose=TRUE, max_iter=500)

# Create a data frame for plotting
tsne_data <- data.frame(
  X = tsne_result$Y[,1],
  Y = tsne_result$Y[,2],
  sample_info
)

# Plot the t-SNE results using ggplot2
ggplot(tsne_data, aes(x = X, y = Y, color = Treatment, label = rownames(sample_info))) +
  geom_point(size = 3) +
  geom_text(aes(label = rownames(sample_info)), hjust = 0, vjust = 1) +
  theme_minimal() +
  ggtitle("t-SNE Plot") +
  xlab("t-SNE 1") +
  ylab("t-SNE 2") +
  theme_minimal()

########################################### Phylogenetic Tree ###############################################

# Load Data
# data <- read.csv("count_data.csv")
# Remove non-numeric columns for htree
# Remove the gene ID column
# data_numeric <- data[, sapply(data, is.numeric)]

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

# Detect outlier samples using hierarchical clustering
htree <- hclust(dist(t(data_numeric)), method = "average")

# Assuming 'htree' and 'metadata' are correctly defined

# Convert labels to factors or unique numeric indices for colors
label_colors <- as.numeric(factor(sample_info$Treatment))

# Plot the dendrogram with colored labels
plot(htree, labels = rownames(sample_info), main = "Hierarchical Clustering Dendrogram", col = label_colors)

##**********************#

# Update this # add color

##**********************##

################################################## Data Manipulation | NODE 2.1 | NODE 2.2 | NODE 2.3 (Yet to be included) #######################

# if there is any outlier samples exclude them from the both expression 
# data and metadata

# samples_to_exclude <- c(1, 2, 5)
# 
# count_data <- count_data[, -samples_to_exclude]
# sample_info <- sample_info [-samples_to_exclude,]

############################################################################################################

########################################### DGE Analysis ###################################################

########################################## Normalization | NODE 3 ###################################################

# Expression Data
# load the count data
# count_data <- read.csv("count_data.csv", header=TRUE,row.names = 1)

# Metadata
# load the sample info
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

# Ensure the number of samples match
# if (ncol(count_data) != nrow(sample_info)) {
#   stop("Number of samples in count_data and sample_info do not match!")
# }

# Convert Condition to factor
# sample_info$Treatment <- factor(sample_info$Treatment)
# 
# Quality Control: detect outlier genes
# library(WGCNA)
# 
# gsg <- goodSamplesGenes(t(count_data))
# summary(gsg)
# 
# Remove outlier genes
# data <- count_data[gsg$goodGenes == TRUE,]

# Convert non-integer values to integers in count data
data <- round(data)
head(data)

# Create a new count data object
new_data <- as.matrix(data)
head(new_data)

# Display dimensions for verification
cat("Dimensions of data:", dim(data), "\n")
cat("Dimensions of new_data:", dim(new_data), "\n")
cat("Dimensions of sample_info:", dim(sample_info), "\n")

# Start Normalization

library(DESeq2)


# Generate the DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = new_data, colData = sample_info, design = ~ Treatment)

# Set the factor levels for the Treatment column based on unique values
condition <- unique(sample_info$Treatment)
dds$Treatment <- factor(dds$Treatment, levels = condition)

# Input all factor from metadata # TO do

# Filter genes with low counts (less than 75% of sample number)
threshold <- round(dim(sample_info)[1] * 0.70)
keep <- rowSums(counts(dds)) >= threshold
dds <- dds[keep,]

# Perform DESeq2 analysis
dds <- DESeq(dds)

# save the normalized counts
normalize_counts <- counts(dds,normalized=TRUE)
head(normalize_counts)
dim(normalize_counts)
write.csv(normalize_counts,"Normalized_Count_Data.csv")


##################################################################################################################################################

################################################## BoxPlot | NODE 4 ######################################################################################

# Log2 transformation for count data
count_matrix <- counts(dds) + 1  # Adding 1 to avoid log(0)
log2_count_matrix <- log2(count_matrix)
boxplot(log2_count_matrix, outline = FALSE, main = "Boxplot of Log2-transformed Count Data",
        xlab = "Sample Name",
        ylab = "Log2-transformed Counts")

# Log2 transformation for normalized count data
normalized_counts <- counts(dds, normalized = TRUE)
log2_normalized_counts <- log2(normalized_counts + 1)  # Adding 1 to avoid log(0)
boxplot(log2_normalized_counts, outline = FALSE,
        main = "Boxplot of Log2-transformed Normalized Count Data",
        xlab = "Sample Name",
        ylab = "Log2-transformed Counts")

################################################################# NODE 4 / NODE 2 #################################################################################


# Now Again perform

# PCA
# UMAP
# tSNE
# Phylogenetic Tree
# Boxplot whisker plot

# With "Normalized_Count_Data.csv" This Normalized Count data
# instead of "count_data.csv" this Count data

############################################################################################################

################################################### DEGs, LFC, FDR | NODE 5 ###################################################

# Find out P values, Log Fold Change(LFC) Values, False Discovery Rate (FDR) Values 

condition <- as.data.frame(condition)

print(condition)

# condition
# 1                    mock
# 2 MPXV_clade_IIa_infected
# 3 MPXV_clade_IIb_infected
# 4   MPXV_clade_I_infected

# Ensure the reference level is a character string
ref_level <- as.character(condition[2,]) #User selection

# set the reference/ control for the treatment factor
dds$Treatment <- relevel(dds$Treatment, ref = ref_level)# User input ref = "....."

# Perform DESeq2 analysis
dds <- DESeq(dds)

# Identify available coefficient names
coeff_names <- as.data.frame(resultsNames(dds))

# Print the coefficient names
print(coeff_names)

#   resultsNames(dds)
# 1                                 Intercept
# 2 Treatment_MPXV_clade_IIa_infected_vs_mock
# 3 Treatment_MPXV_clade_IIb_infected_vs_mock
# 4 Treatment_MPXV_clade_I_infected_vs_mock

# For User: Select 1 or 2 or 3 or 4

X <- coeff_names[2,]

resLFC <- lfcShrink(dds, coef =X  , type = "apeglm")

#change resLFC to a dataframe
resLFC <- as.data.frame(resLFC)

# Researchers are often interested in minimizing the number of false discoveries. 
# Only Keep the significant genes padj (FDR) values is less than 0.05
# resLFC_p_cut <- resLFC[resLFC$padj < 0.05,]

##************** remove the NA rows ************##

# create histogram plot of p-values
hist(resLFC$padj, breaks=seq(0, 1, length = 21), col = "grey", border = "white", 
     xlab = "", ylab = "", main = "Frequencies of padj-values")

summary(resLFC)

############################################ UP and Downregulated Genes | NODE 5.1 ####################################################

# Upregulated genes
Upregulated <- resLFC[resLFC$log2FoldChange > 1 & resLFC$padj < 0.05, ]
Upregulated_padj <- Upregulated[order(Upregulated$padj ),]
write.csv(Upregulated_padj, file = paste0('Upregulated_padj_', X,'.csv'), row.names = TRUE)

# Downregulated genes
Downregulated <- resLFC[resLFC$log2FoldChange < -1 & resLFC$padj < 0.05, ]
Downregulated_padj <- Downregulated[order(Downregulated$padj),]
write.csv(Downregulated_padj, file = paste0('Downregulated_padj_', X,'.csv'), row.names = TRUE )

################################################### Volcano Plot | NODE 6 ###########################################################

# Create a Volcano Plot
ggplot(resLFC,aes(x = log2FoldChange, y = -log10(padj))) +
  
  # Scatter plot points with color-coded regulation
  geom_point(aes(color = ifelse(log2FoldChange > 1.0 & -log10(padj) > 1.3, "Upregulated",
                                ifelse(log2FoldChange < -1.0 & -log10(padj) > 1.3, "Downregulated", "Not Significant"))),
             size = 2.5, alpha = 0.5) +
  
  # Add horizontal dashed line
  geom_hline(yintercept = 1.3, linetype = "dashed", color = "black") +
  
  # Add vertical dashed lines
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +
  
  # Customize plot labels and add the header
  labs(
    title = "Volcano Plot", # Add the header here
    x = "Log2 Fold Change",
    y = "-log10(padj)",
    color = "Regulation"
  ) +
  
  # Customize color palette for regulation categories
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
  
  # Use a minimal theme for the plot
  theme_minimal()

# Create a Volcano Plot

library(ggplot2)

# User will select specific gene names to label and highlight with borders
genes_to_highlight <- c("ENSG00000231500", "ENSG00000204388", "ENSG00000081277")  

# Adding gene names as a column
resLFC$gene <- row.names(resLFC)

ggplot(resLFC, aes(x = log2FoldChange, y = -log10(padj))) +
  
  # Scatter plot points with color-coded regulation
  geom_point(aes(color = ifelse(log2FoldChange > 1.0 & -log10(padj) > 1.3, "Upregulated",
                                ifelse(log2FoldChange < -1.0 & -log10(padj) > 1.3, "Downregulated", "Not Significant"))),
             size = 2.5, alpha = 0.5) +
  
  # Add horizontal dashed line
  geom_hline(yintercept = 1.3, linetype = "dashed", color = "black") +
  
  # Add vertical dashed lines
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +
  
  # Customize plot labels and add the header
  labs(
    title = "Volcano Plot",
    x = "Log2 Fold Change",
    y = "-log10(padj)",
    color = "Regulation"
  ) +
  
  # Customize color palette for regulation categories
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
  
  # Use a minimal theme for the plot
  theme_minimal() +
  
  # Add borders around specific genes and label them
  geom_point(data = subset(resLFC, gene %in% genes_to_highlight),
             aes(x = log2FoldChange, y = -log10(padj)),
             shape = 21, size = 4, stroke = 1, color = "black", fill = NA) +  # shape = 21 for a circle with a border
  geom_text(data = subset(resLFC, gene %in% genes_to_highlight),
            aes(label = gene),
            vjust = -0.5, hjust = 0.5, size = 3, color = "black")


###################################################################################################################