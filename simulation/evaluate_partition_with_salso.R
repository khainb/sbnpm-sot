library(salso)
install.packages(salso)
setwd("C:/Code/sot-gmm/")
methods <- c("MixSW", "SMixW", "SW")
loss_functions <- list(
  "Binder" = binder(),
  "VI" = VI(),
  "omARI" = omARI()
)
num_repeats <- 25

for (method in methods) {
  cat("\n=== Evaluating Method:", method, "===\n")

  # Storage for all scores and unique cluster counts
  all_evaluations <- list()
  unique_counts <- c()

  for (eval_loss in names(loss_functions)) {
    all_evaluations[[eval_loss]] <- list(
      posterior = c(),
      true = c()
    )
  }

  for (rep in 0:(num_repeats - 1)) {
    #cat("Processing repetition:", rep, "\n")

    # Load posterior draws and ground truth
    draws_path <- paste0("simulation/saved/Zs_n200_K100_repeat", rep, ".txt")
    label_path <- paste0("simulation/saved/label_n200_repeat", rep, ".txt")
    method_path <- paste0("simulation/saved/Zs_", method, "_n200_L100_K100_repeat", rep, ".txt")

    draws <- as.matrix(read.csv(draws_path, header = FALSE))
    label <- as.vector(as.matrix(read.csv(label_path, header = FALSE)))
    Z <- as.vector(as.matrix(read.csv(method_path, header = FALSE)))

    unique_counts <- c(unique_counts, length(unique(Z)))

    for (eval_loss in names(loss_functions)) {
      eval_fn <- loss_functions[[eval_loss]]

      score_posterior <- partition.loss(truth = draws, estimate = Z, loss = eval_fn)
      score_true <- partition.loss(truth = label, estimate = Z, loss = eval_fn)

      all_evaluations[[eval_loss]]$posterior <- c(all_evaluations[[eval_loss]]$posterior, score_posterior)
      all_evaluations[[eval_loss]]$true <- c(all_evaluations[[eval_loss]]$true, score_true)
    }
  }

  # Summary
  cat("\n---", method, "Evaluation Summary (Mean ± SD over", num_repeats, "repetitions) ---\n")
  for (eval_loss in names(all_evaluations)) {
    posterior_scores <- all_evaluations[[eval_loss]]$posterior
    true_scores <- all_evaluations[[eval_loss]]$true

    cat("\nLoss:", eval_loss, "\n")
    cat(sprintf("  Posterior: %.4f ± %.4f\n", mean(posterior_scores), sd(posterior_scores)))
    cat(sprintf("  True     : %.4f ± %.4f\n", mean(true_scores), sd(true_scores)))
  }

  cat(sprintf("\nUnique Clusters in Z: %.2f ± %.2f\n", mean(unique_counts), sd(unique_counts)))
}

