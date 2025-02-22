# Find unique neighbors of queried genes
find_unique_neighbors <- function(string_db, string_ids) {
  neighbor_genes <- as.data.frame(string_db$get_neighbors(string_ids))
  unique_neighbors <- unique(neighbor_genes)
  unique_neighbors[, 1]  # Return STRING IDs of unique neighbors
}

# Map neighbor STRING IDs to gene symbols
map_neighbors_to_symbols <- function(string_db, neighbor_ids) {
  aliases <- string_db$get_aliases()
  neighbor_genes_symbols_df <- aliases[aliases$STRING_id %in% neighbor_ids, ]
  neighbor_genes_symbols_df %>% distinct(STRING_id, .keep_all = TRUE)
}

# Match query genes with neighbor genes
match_query_with_neighbors <- function(query_genes_file, neighbor_genes_symbols) {
  query_genes_df <- read.csv(query_genes_file, header = TRUE)
  query_genes_symbol <- query_genes_df[[1]]  # Assuming the first column contains the genes of interest
  neighbor_genes_symbols %>% filter(alias %in% query_genes_symbol)
}

map_reverse_to_symbols <- function(string_db, str_IDs) {
  aliases <- string_db$get_aliases()
  map_rev_to_symbols_df <- aliases[aliases$STRING_id %in% str_IDs, ]
  map_rev_to_symbols_df %>% distinct(STRING_id, .keep_all = TRUE)
}