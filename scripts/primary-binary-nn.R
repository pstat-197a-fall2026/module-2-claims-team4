## PRIMARY TASK (a): binary classification with a neural network
## (also serves as preliminary task 3: an NN that beats logistic PCR)
##
## Run from the repo root with R 4.4.3:
##   & "C:\Program Files\R\R-4.4.3\bin\Rscript.exe" scripts/primary-binary-nn.R
##
## Steps
##   1. parse every page's HTML (Jocelyn's parser; cached to data/)
##   2. build model input text: stemmed title/meta tokens + header/paragraph tokens
##   3. use the shared split (data/split-ids.csv): train 70 / val 10 / test 20
##   4. tf-idf of words + bigrams, vocabulary learned from TRAIN pages only
##   5. try a few small architectures; early-stop each on VAL; pick best by VAL accuracy
##   6. score the chosen model ONCE on TEST (accuracy, sensitivity, specificity, AUC)
##   7. save a deployable model (raw token text in -> probability out) and
##      binary predictions for the 929 unlabeled pages
##
## Outputs
##   results/primary-binary-nn-val-comparison.csv   validation scores per architecture
##   results/primary-binary-nn-test-metrics.csv      final test metrics (for writeups)
##   results/primary-binary-nn-test-preds.csv        test-split predictions (per page)
##   results/model-binary-nn.keras                   deployable model
##   results/preds-binary-nn.csv                     predictions on claims-test (.id, bclass.pred)

suppressPackageStartupMessages({
  library(tidyverse)
  library(keras3)
  library(yardstick)
})
source('scripts/preprocessing-primary-jocelyn.R')
source('scripts/primary-nn-helpers.R')

seed        <- 110122
max_tokens  <- 20000   # vocabulary size (most frequent words + bigrams in train)
max_epochs  <- 30
batch_size  <- 32
patience    <- 3       # early stopping: epochs without val-loss improvement
parsed_file <- 'data/claims-parsed-primary.RData'

# candidate architectures (hidden layer sizes, dropout rate, L2 penalty)
configs <- tribble(
  ~name,          ~units,          ~dropout, ~l2,
  'small',        list(32),        0.3,      0,
  'medium',       list(64),        0.5,      0,
  'medium-l2',    list(64),        0.5,      1e-4,
  'two-layer',    list(128, 32),   0.5,      0
)

## 1. PARSE HTML (cached: delete parsed_file to re-parse) ---------------------
load('data/claims-raw.RData')
load('data/claims-test.RData')
if(file.exists(parsed_file)){
  load(parsed_file)
} else {
  message('parsing labeled pages...');   parsed_raw  <- parse_pages(claims_raw)
  message('parsing unlabeled pages...'); parsed_test <- parse_pages(claims_test)
  save(parsed_raw, parsed_test, file = parsed_file)
}

## 2-3. MODEL INPUT TEXT + SHARED SPLIT ---------------------------------------
split_ids <- read_csv('data/split-ids.csv', col_types = cols(.default = 'c'))
class_levels <- levels(claims_raw$bclass)   # 2nd level = 'Relevant claim content'

dat <- parsed_raw %>%
  mutate(text = nn_text_fn(parsed_raw)) %>%
  select(.id, text) %>%
  inner_join(split_ids, by = '.id') %>%
  mutate(bclass = factor(bclass, levels = class_levels),
         y = as.integer(bclass) - 1L)
stopifnot(nrow(dat) == nrow(claims_raw))
cat('pages per split:\n'); print(table(dat$split))
cat('pages with empty text:', sum(dat$text == ''), '\n')

tr <- filter(dat, split == 'train')
va <- filter(dat, split == 'val')
te <- filter(dat, split == 'test')

## 4. TF-IDF VECTORIZER (fit on train only) ------------------------------------
vectorizer <- layer_text_vectorization(
  standardize = NULL, split = 'whitespace', ngrams = 2,
  max_tokens = max_tokens, output_mode = 'tf_idf'
)
adapt(vectorizer, tr$text)
vectorize <- function(x) as.array(vectorizer(x))
X_tr <- vectorize(tr$text); X_va <- vectorize(va$text); X_te <- vectorize(te$text)
cat('feature matrix:', dim(X_tr), '\n')

## 5. TRAIN CANDIDATES, SELECT ON VALIDATION ----------------------------------
build_dense <- function(units, dropout, l2){
  reg <- if(l2 > 0) regularizer_l2(l2) else NULL
  inp <- keras_input(shape = ncol(X_tr))
  h <- inp
  for(u in units){
    h <- h |>
      layer_dropout(dropout) |>
      layer_dense(u, activation = 'relu', kernel_regularizer = reg)
  }
  out <- h |> layer_dropout(dropout) |> layer_dense(1, activation = 'sigmoid')
  model <- keras_model(inp, out)
  model |> compile(loss = 'binary_crossentropy', optimizer = 'adam',
                   metrics = 'binary_accuracy')
  model
}

fit_config <- function(cfg){
  set_random_seed(seed)
  model <- build_dense(unlist(cfg$units), cfg$dropout, cfg$l2)
  hist <- model |> fit(
    X_tr, tr$y,
    validation_data = list(X_va, va$y),
    epochs = max_epochs, batch_size = batch_size, verbose = 0,
    callbacks = list(callback_early_stopping(monitor = 'val_loss', patience = patience,
                                             restore_best_weights = TRUE))
  )
  p_va <- as.numeric(predict(model, X_va, verbose = 0))
  vl <- hist$metrics$val_loss
  list(model = model,
       row = tibble(name = cfg$name,
                    epochs_run = length(vl),
                    best_epoch = which.min(vl),
                    val_loss = min(vl),
                    val_accuracy = mean((p_va > 0.5) == va$y)))
}

fits <- map(seq_len(nrow(configs)), function(i){
  cfg <- configs[i, ]
  message('training ', cfg$name, '...')
  fit_config(cfg)
})
val_comparison <- map_dfr(fits, 'row')
cat('\n== validation comparison ==\n'); print(val_comparison)
write_csv(val_comparison, 'results/primary-binary-nn-val-comparison.csv')

# best by val accuracy; ties broken by val loss
best <- order(-val_comparison$val_accuracy, val_comparison$val_loss)[1]
best_name <- val_comparison$name[best]
dense_model <- fits[[best]]$model
cat('\nchosen architecture:', best_name, '\n')

## 6. TEST-SET EVALUATION (once) -----------------------------------------------
p_te <- as.numeric(predict(dense_model, X_te, verbose = 0))
test_preds <- te %>%
  transmute(.id, truth = bclass, prob_relevant = p_te,
            estimate = factor(class_levels[(p_te > 0.5) + 1], levels = class_levels))
write_csv(test_preds, 'results/primary-binary-nn-test-preds.csv')

bin_metrics <- metric_set(accuracy, sensitivity, specificity)
test_metrics <- bind_rows(
  bin_metrics(test_preds, truth = truth, estimate = estimate, event_level = 'second'),
  roc_auc(test_preds, truth = truth, prob_relevant, event_level = 'second')
) %>%
  mutate(model = paste0('NN (', best_name, ')'), n_test = nrow(test_preds), .before = 1)
cat('\n== TEST metrics (event = "Relevant claim content") ==\n'); print(test_metrics)
write_csv(test_metrics, 'results/primary-binary-nn-test-metrics.csv')

## 7. DEPLOYABLE MODEL + PREDICTIONS ON UNLABELED PAGES ------------------------
# one model: token string in -> tf-idf (train vocabulary) -> trained network -> probability
inp <- keras_input(shape = 1, dtype = 'string')
# (the vectorizer returns shape (n, 1, vocab) for a (n, 1) string input; flatten to (n, vocab))
deploy_model <- keras_model(inp, inp |> vectorizer() |> layer_flatten() |> dense_model())
save_model(deploy_model, 'results/model-binary-nn.keras', overwrite = TRUE)

# check the reloaded model reproduces the test predictions
# (R character vectors must be passed as a string tensor, one page per row)
as_string_tensor <- function(x) tensorflow::tf$constant(matrix(x), dtype = 'string')
reloaded <- load_model('results/model-binary-nn.keras')
p_check <- as.numeric(predict(reloaded, as_string_tensor(te$text), verbose = 0))
cat('\nmax |reloaded - original| on test pages:', max(abs(p_check - p_te)), '\n')

new_text <- nn_text_fn(parsed_test)
p_new <- as.numeric(predict(reloaded, as_string_tensor(new_text), verbose = 0))
preds_binary <- tibble(.id = parsed_test$.id,
                       bclass.pred = factor(class_levels[(p_new > 0.5) + 1], levels = class_levels))
write_csv(preds_binary, 'results/preds-binary-nn.csv')
cat('predictions for', nrow(preds_binary), 'unlabeled pages; class counts:\n')
print(table(preds_binary$bclass.pred))
