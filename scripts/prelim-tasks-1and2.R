library(tidyverse)
library(tidytext)
source('scripts/preprocessing.R')

## Setup
# For prelim tasks 1 + 2
seed            <- 110122
# Split 1 is the SHARED split in data/split-ids.csv (made by scripts/make-split.R),
# the same one the neural net and primary-task models use; it gives the headline
# numbers. Splits 2..n_splits are extra random 70/10/20 splits (same proportions,
# stratified by mclass) used only to show how much results vary across splits.
# In every split the logistic models train on the 70% train pages only: the 10%
# validation pages are held out (never trained on), so these models see exactly
# as much training data as the neural nets. The number of PCs is tuned by CV
# within the train pages, and accuracy is reported on the 20% test pages.
split_file      <- 'data/split-ids.csv'
split_props     <- c(train = 0.7, val = 0.1, test = 0.2)
n_splits        <- 10
n_folds         <- 5
k_max_word      <- 150     
k_grid_word     <- c(2, 5, 10, 20, 35, 50, 75, 100, 150)
# For task 2 only (bigrams)
k_max_bigram    <- 100     
m_grid_bigram   <- c(0, 1, 2, 5, 10, 20, 35, 50, 75, 100)  
min_docs_bigram <- 5       
text_version_task2 <- 'ph' 

dir.create('results', showWarnings = FALSE)

## Helper functions (to streamline the process)
# Helpers shared by both tasks, except oof_logodds(), that's exclusive to task 2

# PCA of a sparse tf-idf matrix via truncated SVD (irlba), centered not scaled.
# Centering is applied implicitly, so the sparse matrix is never densified.
fit_pca <- function(X, k){
  k <- min(k, ncol(X) - 1, nrow(X) - 1)
  ctr <- Matrix::colMeans(X)
  s <- irlba::irlba(X, nv = k, center = ctr)
  list(center = ctr, rotation = s$v, k = k)
}

# scores of (new) rows on the principal components of a fitted PCA
project_pca <- function(pca, X, prefix = 'PC'){
  Z <- as.matrix(X %*% pca$rotation)
  Z <- sweep(Z, 2, drop(crossprod(pca$center, pca$rotation)), '-')
  colnames(Z) <- paste0(prefix, seq_len(ncol(Z)))
  Z
}

# stratified random train/val/test split of row indices (same proportions as
# the shared split); used only for the extra robustness splits
split_idx <- function(strata, props){
  lab <- character(length(strata))
  for(l in unique(strata)){
    idx <- which(strata == l)
    n <- length(idx)
    n_tr <- round(props[['train']] * n); n_va <- round(props[['val']] * n)
    lab[idx] <- sample(rep(c('train', 'val', 'test'), c(n_tr, n_va, n - n_tr - n_va)))
  }
  list(train = which(lab == 'train'), val = which(lab == 'val'),
       test = which(lab == 'test'))
}

# stratified fold labels
make_folds <- function(y, v){
  f <- integer(length(y))
  for(l in levels(y)){
    idx <- which(y == l)
    f[idx] <- sample(rep_len(seq_len(v), length(idx)))
  }
  f
}

# logistic regression on the columns of Z. During CV grid searches the largest
# models can be near-separable and trigger glm convergence warnings; those are
# muffled (such models get a terrible CV log-loss and are never chosen). Fits
# that are actually used pass final = TRUE and warn if they did not converge.
glm_fit <- function(Z, y, final = FALSE){
  fit <- withCallingHandlers(
    glm(.y ~ ., data = data.frame(Z, .y = y), family = binomial),
    warning = function(w){
      if(grepl('numerically 0 or 1|did not converge', conditionMessage(w))) 
        invokeRestart('muffleWarning')
    })
  if(final && !fit$converged) 
    warning('a final logistic fit did not converge; consider a smaller PC grid')
  fit
}

pred_fn <- function(fit, Z, type = 'response'){
  as.numeric(predict(fit, newdata = data.frame(Z), type = type))
}

auc_fn <- function(y1, p){
  r <- rank(p)
  n1 <- sum(y1); n0 <- sum(1 - y1)
  (sum(r[y1 == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# accuracy, AUC and log-loss; p = predicted probability of the 2nd factor level
metrics_fn <- function(y, p){
  y1 <- as.integer(y == levels(y)[2])
  pc <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  tibble(accuracy = mean((p > 0.5) == y1),
         auc = auc_fn(y1, p),
         logloss = -mean(y1 * log(pc) + (1 - y1) * log(1 - pc)))
}

# cross-validated performance of logistic regression on the first g columns of Z,
# for each g in ncols
cv_grid <- function(Z, y, ncols, folds){
  map_dfr(ncols, function(g){
    p <- numeric(length(y))
    for(f in sort(unique(folds))){
      tr <- folds != f
      fit <- glm_fit(Z[tr, 1:g, drop = FALSE], y[tr])
      p[!tr] <- pred_fn(fit, Z[!tr, 1:g, drop = FALSE])
    }
    bind_cols(tibble(ncol = g), metrics_fn(y, p)) %>%
      rename_with(~ paste0('cv_', .x), -ncol)
  })
}

# [TASK 2 EXCLUSIVE] out-of-fold predicted log-odds (each row predicted by a model that never saw it)
oof_logodds <- function(Z, y, folds){
  out <- numeric(length(y))
  for(f in sort(unique(folds))){
    tr <- folds != f
    fit <- glm_fit(Z[tr, , drop = FALSE], y[tr])
    out[!tr] <- pred_fn(fit, Z[!tr, , drop = FALSE], type = 'link')
  }
  out
}

# logistic PCR: PCA on training rows, number of PCs chosen by CV log-loss,
# evaluated on test rows
fit_pcr <- function(X, y, train, test, k_max, k_grid, folds){
  pca <- fit_pca(X[train, , drop = FALSE], k_max)
  Ztr <- project_pca(pca, X[train, , drop = FALSE])
  Zte <- project_pca(pca, X[test, , drop = FALSE])
  cv <- cv_grid(Ztr, y[train], k_grid[k_grid <= ncol(Ztr)], folds)
  k <- cv$ncol[which.min(cv$cv_logloss)]
  fit <- glm_fit(Ztr[, 1:k, drop = FALSE], y[train], final = TRUE)
  list(k = k, cv = cv, fit = fit, Ztr = Ztr, Zte = Zte,
       p_test = pred_fn(fit, Zte[, 1:k, drop = FALSE]),
       lo_test = pred_fn(fit, Zte[, 1:k, drop = FALSE], type = 'link'))
}


## SHARED SETUP (used by Task 1 and Task 2)
## Parse the raw HTML, tokenize, and build labels.
load('data/claims-raw.RData')

# parsing takes a minute or two, so cache... DELETE these files to re-parse
get_clean <- function(.raw, .selector, .file){
  if(file.exists(.file)){
    load(.file)
  }else{
    claims_clean <- parse_data(.raw, .selector)
    save(claims_clean, file = .file)
  }
  claims_clean
}
# [TASK 1 EXCLUSIVE] the two scraping strategies being compared:
#   'p'  = paragraphs only           (the starter strategy)
#   'ph' = paragraphs + headers h1-h6 (the augmented strategy)
clean <- list(
  p  = get_clean(claims_raw, sel_paragraphs,         'data/claims-clean-p.RData'),
  ph = get_clean(claims_raw, sel_headers_paragraphs, 'data/claims-clean-ph.RData')
)

# ordered token streams (stopwords removed, lemmatized)
streams <- lapply(clean, token_stream_fn)

# [TASK 1 EXCLUSIVE] fair-comparison step: restrict to pages that have text under both
# strategies.
# Pages with no <p> text drop out of the paragraph-only data but may survive
# once headers are included. To compare the two fairly, both versions are
# evaluated on the same pages: those with text under both scraping strategies.
ids_common <- sort(intersect(unique(streams$p$.id), unique(streams$ph$.id)))
coverage <- tibble(
  version = c('p', 'ph'),
  pages_with_text = map_int(streams, ~ n_distinct(.x$.id)),
  pages_compared = length(ids_common))
print(coverage)

labels <- claims_raw %>%
  distinct(.id, bclass, mclass) %>%
  mutate(.id = as.character(.id))
y <- labels$bclass[match(ids_common, labels$.id)]
if(!is.factor(y)) y <- factor(y)
mclass_common <- labels$mclass[match(ids_common, labels$.id)]

# shared split assignment for the pages being compared
if(!file.exists(split_file)) stop(split_file, ' not found: run scripts/make-split.R first')
split_ids <- read_csv(split_file, col_types = cols(.default = 'c'))
split_common <- split_ids$split[match(ids_common, split_ids$.id)]
stopifnot(!anyNA(split_common))
cat('\nshared split, pages with text under both strategies:\n')
print(table(split_common))
cat('\nclass levels (2nd level is the one modeled, i.e. log-odds are for):', 
    levels(y), '\n')
print(table(y))

streams_common <- lapply(streams, function(s) filter(s, .id %in% ids_common))

##############################################################################
## TASK 1: DO HEADERS HELP LOGISTIC PCR?
##   Same word-level logistic PCR on 'p' vs 'ph' text, same train/test splits.
##   Outputs: results/prelim-task1-*.csv
##############################################################################

# Task 1, step 1: word-level tf-idf matrix for each text version
dtm_words <- lapply(streams_common, function(s){
  s %>% word_counts_fn() %>% tfidf_long_fn() %>% to_sparse_fn(ids_common)
})
cat('\nword tf-idf matrices (pages x terms):\n')
print(map(dtm_words, dim))

# Task 1, step 2: train/val/test splits (also reused in Task 2).
# Split 1 = shared split from data/split-ids.csv; the rest are random repeats.
# Only $train is used for fitting/tuning and only $test for scoring; $val is
# deliberately left unused here (it is reserved for the neural nets' early stopping).
shared_split <- list(train = which(split_common == 'train'),
                     val   = which(split_common == 'val'),
                     test  = which(split_common == 'test'))
set.seed(seed)
splits <- c(list(shared_split),
            replicate(n_splits - 1, split_idx(mclass_common, split_props), simplify = FALSE))
split_type <- c('shared', rep('repeat', n_splits - 1))

# Task 1, step 3: fit logistic PCR on each version using the same split and
# record held-out accuracy, AUC and log-loss
task1 <- map_dfr(seq_along(splits), function(r){
  set.seed(seed + r)
  sp <- splits[[r]]
  folds <- make_folds(y[sp$train], n_folds)
  map_dfr(names(dtm_words), function(v){
    fit <- fit_pcr(dtm_words[[v]], y, sp$train, sp$test, 
                   k_max_word, k_grid_word, folds)
    bind_cols(tibble(split = r, split_type = split_type[r], version = v,
                     n_train = length(sp$train), n_test = length(sp$test),
                     n_pcs = fit$k),
              metrics_fn(y[sp$test], fit$p_test))
  })
})
write_csv(task1, 'results/prelim-task1-by-split.csv')
# headline numbers: the shared split (same test pages as the neural net)
task1_shared <- filter(task1, split_type == 'shared')
write_csv(task1_shared, 'results/prelim-task1-shared-split.csv')
cat('\n== TASK 1, shared split (headline) ==\n')
print(task1_shared, width = Inf)

# Task 1, step 4: summarize across splits (mean/sd per version, then paired
# differences ph minus p)
task1_summary <- task1 %>%
  group_by(version) %>%
  summarize(across(c(n_pcs, accuracy, auc, logloss), 
                   list(mean = mean, sd = sd), .names = '{.col}_{.fn}'))

task1_paired <- task1 %>%
  pivot_wider(id_cols = split, names_from = version, 
              values_from = c(accuracy, auc, logloss)) %>%
  transmute(accuracy = accuracy_ph - accuracy_p,
            auc = auc_ph - auc_p,
            logloss = logloss_ph - logloss_p) %>%
  pivot_longer(everything(), names_to = 'metric') %>%
  group_by(metric) %>%
  summarize(mean_diff = mean(value),
            se_diff = sd(value) / sqrt(n()),
            n_splits_ph_better = sum(if(metric[1] == 'logloss') value < 0 else value > 0),
            n_splits = n())

cat('\n== TASK 1: test-set performance, paragraphs (p) vs paragraphs + headers (ph) ==\n')
print(task1_summary, width = Inf)
cat('\npaired difference (ph minus p); for logloss, negative favors ph.\n',
    'SEs treat splits as independent but they share pages, so they are optimistic.\n', sep = '')
print(task1_paired)
write_csv(task1_summary, 'results/prelim-task1-summary.csv')
write_csv(task1_paired, 'results/prelim-task1-paired-differences.csv')

##############################################################################
## TASK 2: DO BIGRAMS ADD INFORMATION?
##   Stage 1: word-level logistic PCR -> predicted log-odds.
##   Stage 2: logistic regression on [log-odds, first m bigram PCs].
##   Reuses `dtm_words` and `splits` from Task 1.
##   Outputs: results/prelim-task2-*.csv
##############################################################################

# Task 2, step 1: choose the text version, then build the bigram tf-idf matrix
# (the word matrix Xw comes from Task 1's dtm_words)
v2 <- text_version_task2
cat('\nTask 2 uses text version:', v2, '\n')

Xw <- dtm_words[[v2]]
Xb <- streams_common[[v2]] %>% 
  bigram_counts_fn() %>% 
  tfidf_long_fn(min_docs = min_docs_bigram) %>% 
  to_sparse_fn(ids_common)
cat('bigram tf-idf matrix (pages x bigrams):', dim(Xb), '\n')

# Task 2, step 2: define the two-stage model, evaluated on one split
# stage 1: word-level logistic PCR -> log-odds
# stage 2: logistic regression on [stage-1 log-odds, first m bigram PCs]
# Stage-1 log-odds for TRAINING pages are out-of-fold (a model that has seen a
# page fits it too well, which would make stage 2 over-trust the log-odds).
# Test pages get log-odds from the stage-1 model fit on all training pages.
run_stacking <- function(sp){
  ytr <- y[sp$train]; yte <- y[sp$test]
  folds1 <- make_folds(ytr, n_folds)
  
  s1 <- fit_pcr(Xw, y, sp$train, sp$test, k_max_word, k_grid_word, folds1)
  lo_tr <- oof_logodds(s1$Ztr[, 1:s1$k, drop = FALSE], ytr, folds1)
  
  pcab <- fit_pca(Xb[sp$train, , drop = FALSE], k_max_bigram)
  Btr <- project_pca(pcab, Xb[sp$train, , drop = FALSE], prefix = 'BPC')
  Bte <- project_pca(pcab, Xb[sp$test, , drop = FALSE], prefix = 'BPC')
  S_tr <- cbind(logodds = lo_tr, Btr)
  S_te <- cbind(logodds = s1$lo_test, Bte)
  
  m_grid <- m_grid_bigram[m_grid_bigram <= ncol(Btr)]
  folds2 <- make_folds(ytr, n_folds)
  
  # stacked model: choose m by CV log-loss (m = 0 means bigrams add nothing)
  cv2 <- cv_grid(S_tr, ytr, 1 + m_grid, folds2) %>% mutate(m = ncol - 1, .before = 1)
  m <- cv2$m[which.min(cv2$cv_logloss)]
  fit0 <- glm_fit(S_tr[, 1, drop = FALSE], ytr, final = TRUE)
  fit2 <- glm_fit(S_tr[, 1:(1 + m), drop = FALSE], ytr, final = TRUE)
  
  # likelihood-ratio test (training pages): do the first m bigram PCs improve on
  # the log-odds alone?
  lrt <- map_dfr(m_grid[m_grid > 0], function(mm){
    f <- glm_fit(S_tr[, 1:(1 + mm), drop = FALSE], ytr)
    chisq <- fit0$deviance - f$deviance
    # a negative statistic is impossible for nested models: it means the larger
    # fit did not converge (near-separable data), so report NA rather than a p-value
    ok <- f$converged && chisq >= 0
    tibble(m = mm, lrt_chisq = if_else(ok, chisq, NA_real_), 
           lrt_p = if_else(ok, pchisq(chisq, df = mm, lower.tail = FALSE), NA_real_))
  })
  cv2 <- left_join(cv2, lrt, by = 'm')
  
  # reference: bigram PCs alone
  cvb <- cv_grid(Btr, ytr, m_grid[m_grid > 0], folds2)
  mb <- cvb$ncol[which.min(cvb$cv_logloss)]
  fitb <- glm_fit(Btr[, 1:mb, drop = FALSE], ytr, final = TRUE)
  
  test <- bind_rows(
    tibble(model = 'words PCR', n_pcs = s1$k, 
           metrics_fn(yte, s1$p_test)),
    tibble(model = 'words PCR, recalibrated', n_pcs = s1$k, 
           metrics_fn(yte, pred_fn(fit0, S_te[, 1, drop = FALSE]))),
    tibble(model = 'bigram PCR only', n_pcs = mb, 
           metrics_fn(yte, pred_fn(fitb, Bte[, 1:mb, drop = FALSE]))),
    tibble(model = 'words log-odds + bigram PCs', n_pcs = s1$k, bigram_pcs = m,
           metrics_fn(yte, pred_fn(fit2, S_te[, 1:(1 + m), drop = FALSE])))
  )
  list(test = test, cv = cv2, stacked_fit = fit2, m = m)
}

# Task 2, step 3: run the two-stage model on every split
task2 <- vector('list', n_splits)
for(r in seq_len(n_splits)){
  set.seed(seed + 1000 + r)
  task2[[r]] <- run_stacking(splits[[r]])
  cat('split', r, 'of', n_splits, 'done; bigram PCs chosen:', task2[[r]]$m, '\n')
}

# Task 2, step 4: summarize across splits (mean/sd per model, paired
# differences of the stacked model vs the baselines, bigram PCs chosen)
task2_test <- map_dfr(seq_len(n_splits), ~ mutate(task2[[.x]]$test, split = .x,
                                                   split_type = split_type[.x], .before = 1))
write_csv(task2_test, 'results/prelim-task2-by-split.csv')
task2_shared <- filter(task2_test, split_type == 'shared')
write_csv(task2_shared, 'results/prelim-task2-shared-split.csv')
cat('\n== TASK 2, shared split (headline) ==\n')
print(task2_shared, width = Inf)

task2_summary <- task2_test %>%
  group_by(model) %>%
  summarize(across(c(accuracy, auc, logloss), 
                   list(mean = mean, sd = sd), .names = '{.col}_{.fn}')) 

# paired differences: stacked vs word-only, and stacked vs recalibrated words
# (the latter isolates what the bigram PCs add from simple recalibration)
paired_vs <- function(base){
  stk <- 'words log-odds + bigram PCs'
  task2_test %>%
    filter(model %in% c(stk, base)) %>%
    pivot_wider(id_cols = split, names_from = model, 
                values_from = c(accuracy, auc, logloss)) %>%
    transmute(accuracy = .data[[paste0('accuracy_', stk)]] - .data[[paste0('accuracy_', base)]],
              auc = .data[[paste0('auc_', stk)]] - .data[[paste0('auc_', base)]],
              logloss = .data[[paste0('logloss_', stk)]] - .data[[paste0('logloss_', base)]]) %>%
    pivot_longer(everything(), names_to = 'metric') %>%
    group_by(metric) %>%
    summarize(mean_diff = mean(value), se_diff = sd(value) / sqrt(n()),
              n_splits_stacked_better = sum(if(metric[1] == 'logloss') value < 0 else value > 0),
              n_splits = n()) %>%
    mutate(baseline = base, .before = 1)
}
task2_paired <- bind_rows(paired_vs('words PCR'), paired_vs('words PCR, recalibrated'))

task2_m <- tibble(split = seq_len(n_splits), 
                  bigram_pcs_chosen = map_int(task2, 'm'))

cat('\n== TASK 2: test-set performance ==\n')
print(task2_summary, width = Inf)
cat('\npaired difference (stacked minus baseline); for logloss, negative favors stacked.\n')
print(task2_paired, width = Inf)
cat('\nnumber of bigram PCs chosen by CV in each split (0 = bigrams not used):\n')
print(task2_m$bigram_pcs_chosen)

cat('\nFirst split, training pages: CV performance and likelihood-ratio test of\n',
    'adding m bigram PCs to the word-level log-odds\n', sep = '')
print(task2[[1]]$cv, n = Inf, width = Inf)
cat('\nFirst split, stacked model coefficients (first rows):\n')
print(head(summary(task2[[1]]$stacked_fit)$coefficients, 8))

write_csv(task2_summary, 'results/prelim-task2-summary.csv')
write_csv(task2_paired, 'results/prelim-task2-paired-differences.csv')
write_csv(task2_m, 'results/prelim-task2-bigram-pcs-chosen.csv')
write_csv(task2[[1]]$cv, 'results/prelim-task2-first-split-cv-lrt.csv')