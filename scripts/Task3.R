library(tidyverse)
library(keras3)
library(tensorflow)
source('scripts/preprocessing.R')

seed <- 110122
set.seed(seed)
tf$random$set_seed(seed)

dir.create('results', showWarnings = FALSE)

## 1. Load the exact same cleaned data & common IDs as Tasks 1 & 2
load('data/claims-raw.RData')
load('data/claims-clean-p.RData')
clean_p <- claims_clean
load('data/claims-clean-ph.RData')
clean_ph <- claims_clean

# Match Task 1's exact page universe (ids_common)
streams_p  <- token_stream_fn(clean_p)
streams_ph <- token_stream_fn(clean_ph)
ids_common <- sort(intersect(unique(streams_p$.id), unique(streams_ph$.id)))

# Build normalized text strings per page from the lemmatized/stopword-filtered stream
text_df <- clean_ph %>%
  mutate(.id = as.character(.id)) %>%
  filter(.id %in% ids_common) %>%
  select(.id, text = text_clean)

labels <- claims_raw %>%
  distinct(.id, bclass, mclass) %>%
  mutate(.id = as.character(.id))

df <- tibble(.id = ids_common) %>%
  left_join(text_df, by = ".id") %>%
  left_join(labels, by = ".id") %>%
  mutate(bclass = factor(bclass))

## 2. Load the SHARED 70/10/20 split (data/split-ids.csv)
split_ids <- read_csv('data/split-ids.csv', col_types = cols(.default = 'c'))
df$split <- split_ids$split[match(df$.id, split_ids$.id)]
stopifnot(!anyNA(df$split))

train_df <- filter(df, split == 'train')
val_df   <- filter(df, split == 'val')
test_df  <- filter(df, split == 'test')

x_train <- train_df$text
y_train <- as.integer(train_df$bclass == levels(df$bclass)[2])

x_val   <- val_df$text
y_val   <- as.integer(val_df$bclass == levels(df$bclass)[2])

x_test  <- test_df$text
y_test  <- as.integer(test_df$bclass == levels(df$bclass)[2])

## 3. Metric helper matching Task 1 & 2's metrics_fn()
auc_fn <- function(y1, p){
  r <- rank(p)
  n1 <- sum(y1); n0 <- sum(1 - y1)
  (sum(r[y1 == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

eval_metrics <- function(y1, p){
  pc <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  tibble(
    accuracy = mean((p > 0.5) == y1),
    auc      = auc_fn(y1, p),
    logloss  = -mean(y1 * log(pc) + (1 - y1) * log(1 - pc))
  )
}

## 4. Experiment Grid: Architectures, N-grams, and Regularization
configs <- tribble(
  ~model_name,              ~ngrams, ~max_tokens, ~units1, ~units2, ~dropout, ~l2,    ~lr,
  "1_starter_linear_25",    0L,      0L,          25L,     0L,      0.2,      0,      0.001,
  "2_relu_32_reg",          0L,      5000L,       32L,     0L,      0.3,      0.001,  0.001,
  "3_relu_64_reg",          0L,      5000L,       64L,     0L,      0.4,      0.002,  0.001,
  "4_deep_64_16_reg",       0L,      5000L,       64L,     16L,     0.4,      0.002,  0.001,
  "5_bigrams_64_reg",       2L,      8000L,       64L,     0L,      0.4,      0.002,  0.001
)

results_list <- list()
best_logloss <- Inf
best_model   <- NULL
best_name    <- ""

for (i in seq_len(nrow(configs))) {
  cfg <- configs[i, ]
  cat("\n--- Training:", cfg$model_name, "---\n")
  set.seed(seed + i)
  tf$random$set_seed(seed + i)
  
  # Adapt TF-IDF layer ONLY on training text
  vec_args <- list(
    standardize = NULL,
    split = "whitespace",
    output_mode = "tf_idf"
  )
  if (cfg$ngrams > 0L)     vec_args$ngrams     <- as.integer(cfg$ngrams)
  if (cfg$max_tokens > 0L) vec_args$max_tokens <- as.integer(cfg$max_tokens)
  
  vec_layer <- do.call(layer_text_vectorization, vec_args)
  vec_layer %>% adapt(x_train)
  
  # Build model
  nn <- keras_model_sequential() %>%
    vec_layer() %>%
    layer_dropout(cfg$dropout) %>%
    layer_dense(
      units = cfg$units1,
      activation = if (cfg$model_name == "1_starter_linear_25") "linear" else "relu",
      kernel_regularizer = if (cfg$l2 > 0) regularizer_l2(cfg$l2) else NULL
    ) %>%
    layer_dropout(cfg$dropout)
  
  if (cfg$units2 > 0) {
    nn <- nn %>%
      layer_dense(
        units = cfg$units2,
        activation = "relu",
        kernel_regularizer = if (cfg$l2 > 0) regularizer_l2(cfg$l2) else NULL
      ) %>%
      layer_dropout(cfg$dropout * 0.75)
  }
  
  nn <- nn %>% layer_dense(units = 1, activation = "sigmoid")
  
  nn %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = cfg$lr),
    metrics = c("binary_accuracy")
  )
  
  history <- nn %>% fit(
    x = x_train,
    y = y_train,
    validation_data = list(x_val, y_val),
    epochs = 40,
    batch_size = 32,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = 5,
        restore_best_weights = TRUE
      )
    ),
    verbose = 0
  )
  
  p_val  <- as.numeric(nn %>% predict(x_val, verbose = 0))
  p_test <- as.numeric(nn %>% predict(x_test, verbose = 0))
  
  val_m  <- eval_metrics(y_val, p_val) %>% rename_with(~ paste0("val_", .x))
  test_m <- eval_metrics(y_test, p_test)
  
  results_list[[i]] <- bind_cols(
    tibble(
      model_name = cfg$model_name,
      epochs_run = length(history$metrics$val_loss)
    ),
    val_m,
    test_m
  )
  
  # Select best model using validation log-loss (no test-set peeking!)
  if (val_m$val_logloss < best_logloss) {
    best_logloss <- val_m$val_logloss
    best_model   <- nn
    best_name    <- cfg$model_name
  }
}

task3_results <- bind_rows(results_list)
cat("\n== TASK 3: Neural Network Experiments (Shared Split) ==\n")
print(task3_results, width = Inf)
write_csv(task3_results, "results/prelim-task3-shared-split.csv")

## 5. Direct Comparison Against Task 1 Headline Benchmark
if (file.exists("results/prelim-task1-shared-split.csv")) {
  task1_bench <- read_csv("results/prelim-task1-shared-split.csv", show_col_types = FALSE) %>%
    filter(version == "ph") %>%
    select(accuracy, auc, logloss) %>%
    mutate(model = "Task 1: Logistic PCR (ph)", .before = 1)
  
  task3_best <- task3_results %>%
    filter(model_name == best_name) %>%
    select(accuracy, auc, logloss) %>%
    mutate(model = paste0("Task 3: NN (", best_name, ")"), .before = 1)
  
  cat("\n== HEAD-TO-HEAD: Task 1 Benchmark vs. Best Task 3 NN ==\n")
  print(bind_rows(task1_bench, task3_best))
}

save_model(best_model, "results/prelim-task3-best-nn.keras")