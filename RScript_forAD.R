#Code to generate Heatmap and PCA from the proteomic dataset

library(tidyr)
library(dplyr)
library(stringr)
library(stats)

#AI table contains the LFQ value for all the samples and protein identified by MS-proteomics
#Conditions are set as CT, AD TAU+ and AD TAU-

AI_table <- read.delim("AI_table.txt")

# Transform the table
AI_table <- AI_table %>%
  pivot_longer(
    cols = -T..Genes,  # Keep gene names
    names_to = "Sample",
    values_to = "Value") %>%
  pivot_wider(
    names_from = T..Genes,
    values_from = Value)

# Convert columns 3 to the last column to numeric
AI_table[, 3:ncol(AI_table)] <- lapply(AI_table[, 3:ncol(AI_table)], function(x) as.numeric(as.character(x)))

# Define the control group
ct_group <- AI_table$Condition == "CT"  # Adjust the condition to match your CT group

# Function to normalize using the median and SD of the control group
normalize_column <- function(column, ct_group) {
  # Check if column is numeric
  if (is.numeric(column)) {
    # Get the median and SD for the CT group, excluding NaN values
    ct_median <- median(column[ct_group], na.rm = TRUE)
    ct_sd <- sd(column[ct_group], na.rm = TRUE)
    
    # Check if SD is zero to avoid division by zero
    if (ct_sd == 0) {
      return(rep(NA, length(column)))  # If SD is zero, return NA for all values in the column
    }
    
    # Normalize by (x - median) / SD, excluding NaN values
    return((column - ct_median) / ct_sd)
  } else {
    # If column is not numeric, return it unchanged
    return(column)
  }
}

# Apply the normalization to each column
normalized_data <- as.data.frame(lapply(AI_table, function(col) normalize_column(col, ct_group)))

write.csv(normalized_data, "normalized_sig_proteins.csv")


#Subset the database according to the proteins obtained from XGBoost

genes_of_interest = c("YWHAZ", "YWHAE", "GMFB",  "YWHAG",  "LCAT", "APLP1", 
                      "ADAM23", "MAN2A1", "SHISA7", "CBLN1", "DKK3", "ECM1",  
                      "SPOCK3", "PROCR")

normalized_data<-normalized_data[,names(normalized_data) %in% c(genes_of_interest, "Condition")]

# Load necessary package
library(pheatmap)

rownames(normalized_data) <- normalized_data$Sample 

# Extract numeric data (assuming columns 3 onward contain protein data)
protein_data <- normalized_data[, 3:ncol(normalized_data)]

# Convert to matrix (heatmap requires a numeric matrix)
protein_matrix <- as.matrix(protein_data)

# Handle NA values (optional: replace with 0 or impute)
protein_matrix[is.na(protein_matrix)] <- 0

annotation_row <- data.frame(Category = normalized_data$Condition)  # Adjust column name if needed
rownames(annotation_row) <- rownames(protein_matrix)  # Set row names to match protein_matrix

# Define colors for annotation (Make sure the categories exist in annotation_row)
unique_categories <- unique(annotation_row$Category)  # Get unique categories
annotation_colors <- list(
  Category = setNames(colorRampPalette(c("blue", "green"))(length(unique_categories)), unique_categories)
)

# Generate heatmap with row annotations
pheatmap(protein_matrix, 
         scale = "row",  # Normalizes rows (optional)
         clustering_distance_rows = "euclidean", 
         clustering_distance_cols = "euclidean", 
         clustering_method = "ward.D2",
         color = colorRampPalette(c("blue", "white", "red"))(50),
         show_rownames = FALSE, 
         show_colnames = FALSE,
         annotation_row = annotation_row,  # Add row annotations
         annotation_colors = annotation_colors)  # Apply colors to row annotations


# Ensure that the Condition column is a factor
normalized_data$Condition <- as.factor(normalized_data$Condition)

# Compute the median for each protein per condition
median_df <- aggregate(normalized_data[, 3:ncol(normalized_data)], 
                       by = list(Condition = normalized_data$Condition), 
                       FUN = median, na.rm = TRUE)

# Set Condition as row names to maintain structure
rownames(median_df) <- median_df$Condition
median_df$Condition <- NULL  # Remove redundant column

# Transpose the dataframe so that proteins are in columns
median_matrix <- as.matrix(t(median_df))

# Handle NA values (optional: replace with 0 or impute)
median_matrix[is.na(median_matrix)] <- 0

# Run pheatmap and store the object
pheatmap_object <- pheatmap(median_matrix, 
                            scale = "row",
                            clustering_distance_rows = "euclidean", 
                            clustering_distance_cols = "euclidean", 
                            clustering_method = "ward.D2",
                            color = colorRampPalette(c("blue", "white", "red"))(50),
                            show_rownames = FALSE, 
                            show_colnames = TRUE)

# Extract row (protein) clustering
protein_clusters <- cutree(pheatmap_object$tree_row, k = 5)  # Adjust 'k' based on visual clusters

# Convert to dataframe for visualization
protein_cluster_df <- data.frame(Protein = rownames(median_matrix), 
                                 Cluster = protein_clusters)

# Save to CSV for further analysis
write.csv(protein_cluster_df, "protein_clusters.csv", row.names = FALSE)

# Extract protein (row) clusters
num_clusters <- 5  # Adjust based on visual clusters
protein_clusters <- cutree(pheatmap_object$tree_row, k = num_clusters)

# Create annotation dataframe for rows (proteins)
annotation_row <- data.frame(Cluster = factor(protein_clusters))
rownames(annotation_row) <- rownames(median_matrix)

# Define colors for clusters
cluster_colors <- colorRampPalette(c("purple", "orange", "green", "cyan", "pink"))(num_clusters)
annotation_colors <- list(Cluster = setNames(cluster_colors, unique(annotation_row$Cluster)))

# Re-run pheatmap with annotations
pheatmap(median_matrix, 
         scale = "row",
         clustering_distance_rows = "euclidean", 
         clustering_distance_cols = "euclidean", 
         clustering_method = "ward.D2",
         color = colorRampPalette(c("blue", "white", "red"))(50),
         show_rownames = FALSE, 
         show_colnames = TRUE,
         annotation_row = annotation_row,   # Add row cluster colors
         annotation_colors = annotation_colors)  # Apply cluster colors


# Perform PCA
pca_result <- prcomp(umap_input, scale. = TRUE)

# Plot Scree plot to show explained variance
scree_plot <- data.frame(PC = 1:length(pca_result$sdev), 
                         Variance = (pca_result$sdev)^2 / sum((pca_result$sdev)^2))

ggplot(scree_plot, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  labs(title = "Scree Plot", y = "Proportion of Variance", x = "Principal Component")

# Extract PCA result (scores) and create a dataframe
pca_scores <- data.frame(pca_result$x)
pca_scores$Condition <- normalized_data$Condition  # Add condition (row names) to the dataframe

# Plot PCA biplot (PC1 vs PC2)
ggplot(pca_scores, aes(x = PC1, y = PC2, color = Condition)) +
  geom_point(size = 3) +
  theme_minimal() +
  labs(title = "PCA Biplot (PC1 vs PC2)", x = "PC1", y = "PC2")