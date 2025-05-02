library(salso)
setwd("C:/Code/sot-gmm/")
# Define loss functions
loss_functions <- list(
  "Binder" = binder(),
  "VI" = VI(),
  "omARI" = omARI()
)


# Initialize structures to store evaluation scores and unique cluster counts
all_evaluations <- list()
unique_counts <- list()

for (est_loss in names(loss_functions)) {
  all_evaluations[[est_loss]] <- list()
  unique_counts[[est_loss]] <- c()
  for (eval_loss in names(loss_functions)) {
    all_evaluations[[est_loss]][[eval_loss]] <- c()
    all_evaluations[[est_loss]][[paste0(eval_loss, "_True")]] <- c()
  }
}


# Load data
draws_path <- paste0("faithful/saved/Zs.txt")
draws <- as.matrix(read.csv(draws_path, header = FALSE))

for (est_loss in names(loss_functions)) {
  loss_function <- loss_functions[[est_loss]]
  estimated_partition <- salso(draws, loss = loss_function, nRuns = 1, nCores = 1)
  partition_path <- paste0("faithful/saved/", est_loss, ".csv")
  write.table(estimated_partition, file = partition_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
  # Track the number of unique clusters
  unique_counts[[est_loss]] <- c(unique_counts[[est_loss]], length(unique(estimated_partition)))

  for (eval_loss in names(loss_functions)) {
    eval_fn <- loss_functions[[eval_loss]]
    score_posterior <- partition.loss(truth = draws, estimate = estimated_partition, loss = eval_fn)

    all_evaluations[[est_loss]][[eval_loss]] <- c(all_evaluations[[est_loss]][[eval_loss]], score_posterior)
  }
}


# Compute and print statistics

for (est_loss in names(all_evaluations)) {
  cat("\nEstimated using loss:", est_loss, "\n")

  # Print evaluation scores
  for (eval_loss in names(all_evaluations[[est_loss]])) {
    scores <- all_evaluations[[est_loss]][[eval_loss]]
    mean_score <- mean(scores)
    sd_score <- sd(scores)
    cat(sprintf("  %s: %.4f ± %.4f\n", eval_loss, mean_score, sd_score))
  }

  # Print unique cluster count stats
  mean_unique <- mean(unique_counts[[est_loss]])
  sd_unique <- sd(unique_counts[[est_loss]])
  cat(sprintf("  Num. Unique Clusters: %.2f ± %.2f\n", mean_unique, sd_unique))
}

