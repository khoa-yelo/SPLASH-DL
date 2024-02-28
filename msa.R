BiocManager::install("msa")

library(msa)
library(Biostrings)
library(dplyr)
library(purrr)


calMSAprob <- function(seqs){
  # Convert sequences to DNAStringSet
  dna_string_set <- DNAStringSet(seqs)
  # Run MSA
  msa_result <- msa(dna_string_set, type = "dna", order = "input")
  # Convert MSA to a matrix of substitution frequencies
  substitution_matrix <- consensusMatrix(msa_result)
  # Calculate probabilities by normalizing the substitution frequencies
  prob_matrix <- substitution_matrix / colSums(substitution_matrix, na.rm = TRUE)
  # keep only rows with names in ["A", "C", "G", "T", "-"]
  prob_matrix <- prob_matrix[c("A", "C", "G", "T", "-"),]
  msa_and_prob <- list(DNAStringSet(msa_result), as.character(msa_result), prob_matrix)
  msa_and_prob
}

path = "/oak/stanford/groups/horence/khoa/data/splash_classification/subset_TSP_SS2_tissue-TSP7-Blood-classified_compactors.tsv"
aligned_path = "/oak/stanford/groups/horence/khoa/data/splash_classification/aligned_subset_TSP_SS2_tissue-TSP7-Blood-classified_compactors.tsv"

data <- read.csv(path, header = TRUE)
df <- data.frame(data)
result <- df %>%
  group_by(anchor_index) %>%
  summarize(compactors = list(data.frame(compactor_valid, compactor_index)))
# apply the above function for all column
result$compactors_aligned <- map(result$compactors, ~ calMSAprob(.x$compactor_valid)[[2]])
# loop through all compactors and append compactor_index to a list
compactor_indexes <- unlist(lapply(result$compactors, function(x) x$compactor_index))
aligned_compactors <- unlist(result$compactors_aligned)
df$aligned_compactor <- aligned_compactors[match(df$compactor_index, compactor_indexes)]

#set_of_msa <- map(result$compactors, ~ calMSAprob(.x$compactor_valid)[[1]])
#set_of_msa[1]
# write to csv
write.csv(df, "C:\\Users\\khoah\\PhD_Documents\\SPLASH\\compactor_classified_small_aligned.csv", row.names = FALSE)

# get index of each anchor_event class in df
anchor_event_index <- which(df$anchor_event == 1)
df$anchor_event
calMSAprob(result$compactors[[1000]]$compactor_valid)[[1]]
