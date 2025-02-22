# Function to retrieve interactions for a given STRING ID
get_single_gene_interactions <- function(string_db, single_string_id) {
  # Retrieve neighbors
  neighbors <- string_db$get_neighbors(single_string_id)
  
  # Combine single_string_id with neighbors
  ssi_and_neighbors <- c(single_string_id, neighbors)
  
  # Retrieve interactions and process
  interactions <- string_db$get_interactions(ssi_and_neighbors) %>%
    filter(from == single_string_id | to == single_string_id) %>%
    mutate(target = ifelse(from == single_string_id, to, from)) %>%
    distinct(target, .keep_all = TRUE) %>%
    arrange(desc(combined_score))
  
  # Get all targets
  all_targets <- interactions %>% select(target)
  all_targets_and_ssi <- rbind(all_targets, single_string_id)
  
  # Select top 20 interactions
  top_interactions <- interactions %>% head(20)
  top_targets <- top_interactions %>% select(target)
  
  # Combine top targets with single_string_id
  combined_targets <- rbind(top_targets, single_string_id)
  
  # Return results as a list
  return(list(interactions = interactions,
              combined_targets = combined_targets,
              all_targets = all_targets_and_ssi))
}

# Function to plot the network based on the retrieved interactions
plot_gene_interaction_network <- function(string_db, combined_targets) {
  string_db$plot_network(combined_targets)
}

