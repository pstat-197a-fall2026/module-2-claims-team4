## PRIMARY TASK: train final binary + multiclass models
## Run from the repository root. Outputs:
##   results/primary-models.rds        deployable model bundle (vectorizer + 2 SVMs)
##   results/primary-cv-summary.csv    5-fold CV accuracy estimates (if run_cv = TRUE)
##
## Model summary
##   text      : page title + meta/og descriptions (kept as separate 'm_' features)
##               + headers h1-h6 + paragraphs; stopwords removed, stemmed
##   features  : unigrams + bigrams, sublinear tf-idf, L2-normalised rows,
##               terms in >= 3 training pages
##   binary    : linear SVM (e1071, cost = 1), Platt-scaled probabilities
##   multiclass: hierarchical -- the binary SVM decides relevant vs N/A; for pages
##               predicted relevant, a multiclass linear SVM picks among the four
##               relevant categories. Binary and multiclass predictions are
##               therefore always consistent.

source('scripts/preprocessing-primary-jocelyn.R')
suppressPackageStartupMessages(library(e1071))

run_cv <- TRUE      # set FALSE to skip the ~10 minute accuracy estimate
svm_cost <- 1
seed <- 197

## 1. parse training pages (cached; delete the file to re-parse)
if(file.exists('data/parsed-train.rds')){
  parsed_train <- readRDS('data/parsed-train.rds')
} else {
  load('data/claims-raw.RData')
  parsed_train <- parse_pages(claims_raw) %>%
    left_join(claims_raw %>% transmute(.id = as.character(.id), bclass, mclass), by = '.id')
  saveRDS(parsed_train, 'data/parsed-train.rds')
}

make_tokens <- function(parsed){
  Map(c, tokenize_docs(parsed$meta, prefix = 'm_'), tokenize_docs(parsed$hp))
}
tokens <- make_tokens(parsed_train)
yb <- parsed_train$bclass
ym <- parsed_train$mclass
na_level <- levels(yb)[1]                       # 'N/A: No relevant content.'
rel_levels <- setdiff(levels(ym), levels(ym)[1])

fit_models <- function(tok, yb, ym){
  vec <- fit_vectorizer(tok)
  X <- transform_vectorizer(vec, tok)
  svm_b <- svm(X, yb, kernel = 'linear', cost = svm_cost, probability = TRUE, scale = FALSE)
  rel <- ym != levels(ym)[1]
  svm_m <- svm(X[rel, ], droplevels(ym[rel]), kernel = 'linear', cost = svm_cost,
               probability = TRUE, scale = FALSE)
  list(vectorizer = vec, svm_binary = svm_b, svm_multi = svm_m,
       bclass_levels = levels(yb), mclass_levels = levels(ym))
}

predict_models <- function(bundle, tok){
  X <- transform_vectorizer(bundle$vectorizer, tok)
  pb <- attr(predict(bundle$svm_binary, X, probability = TRUE), 'probabilities')
  p_rel <- pb[, bundle$bclass_levels[2]]
  pm <- attr(predict(bundle$svm_multi, X, probability = TRUE), 'probabilities')
  m_rel <- colnames(pm)[max.col(pm, ties.method = 'first')]
  bpred <- ifelse(p_rel > 0.5, bundle$bclass_levels[2], bundle$bclass_levels[1])
  mpred <- ifelse(p_rel > 0.5, m_rel, bundle$mclass_levels[1])
  tibble(bclass.pred = factor(bpred, levels = bundle$bclass_levels),
         mclass.pred = factor(mpred, levels = bundle$mclass_levels),
         p_relevant = p_rel)
}

## 2. 5-fold stratified cross-validation (vectorizer refit within each fold)
if(run_cv){
  set.seed(seed)
  folds <- integer(length(ym))
  for(l in levels(ym)){ i <- which(ym == l); folds[i] <- sample(rep_len(1:5, length(i))) }
  cv_pred <- vector('list', 5)
  for(f in 1:5){
    tr <- folds != f
    b <- fit_models(tokens[tr], yb[tr], ym[tr])
    cv_pred[[f]] <- predict_models(b, tokens[!tr]) %>%
      mutate(row = which(!tr), fold = f)
    message('CV fold ', f, ' done')
  }
  cv_pred <- bind_rows(cv_pred) %>% arrange(row) %>%
    mutate(bclass = yb[row], mclass = ym[row])
  cv_summary <- cv_pred %>%
    group_by(fold) %>%
    summarize(binary_acc = mean(bclass.pred == bclass),
              multiclass_acc = mean(mclass.pred == mclass)) %>%
    summarize(binary_acc_mean = mean(binary_acc), binary_acc_se = sd(binary_acc) / sqrt(n()),
              multiclass_acc_mean = mean(multiclass_acc), multiclass_acc_se = sd(multiclass_acc) / sqrt(n()))
  print(cv_summary)
  cat('\nBinary confusion (rows = truth):\n'); print(table(cv_pred$bclass, cv_pred$bclass.pred))
  cat('\nMulticlass confusion (rows = truth):\n'); print(table(cv_pred$mclass, cv_pred$mclass.pred))
  cat('\nPer-class recall (multiclass):\n')
  print(cv_pred %>% group_by(mclass) %>% summarize(n = n(), recall = mean(mclass.pred == mclass)))
  write_csv(cv_summary, 'results/primary-cv-summary.csv')
}

## 3. fit final models on all labelled pages and export
set.seed(seed)
bundle <- fit_models(tokens, yb, ym)
bundle$make_tokens <- make_tokens
bundle$predict_models <- predict_models
bundle$info <- list(trained_on = nrow(parsed_train), vocab_size = length(bundle$vectorizer$vocab),
                    created = Sys.time())
saveRDS(bundle, 'results/primary-models.rds')
cat('saved results/primary-models.rds; vocabulary size', length(bundle$vectorizer$vocab), '\n')