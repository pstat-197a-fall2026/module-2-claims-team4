## Deliverable 4: use the saved models to classify new pages
##
## Run from the repository root (takes ~5 minutes, mostly HTML parsing):
##   Rscript scripts/prediction.R
##
## Needs: tidyverse, rvest, xml2, Matrix, SnowballC, tidytext, e1071
##
## Saved model: results/primary-models.rds (made by scripts/train-primary-jocelyn.R)
##   a list holding
##     vectorizer      tf-idf vocabulary + idf weights learned from the labeled pages
##     svm_binary      linear SVM: relevant claim content vs N/A
##     svm_multi       linear SVM: which of the 4 relevant categories
##     make_tokens()   turns parsed pages into tokens
##     predict_models() runs both SVMs; pages the binary SVM calls N/A get N/A
##                     for mclass too, so the two predictions always agree
##
## Output: results/preds-group4.RData containing pred_df with columns
##   .id, bclass.pred, mclass.pred   (one row per test page)

source('scripts/preprocessing-primary-jocelyn.R')   # HTML parsing + tokenizing functions
suppressPackageStartupMessages(library(e1071))       # needed to predict with the SVMs

# 1. load the saved models and the new pages
bundle <- readRDS('results/primary-models.rds')
load('data/claims-test.RData')                       # claims_test: .id, text_tmp (raw HTML)

# 2. preprocess exactly as in training: parse HTML -> tokens
#    (every page is kept, so every page gets a prediction)
parsed_test <- parse_pages(claims_test)
tokens_test <- bundle$make_tokens(parsed_test)

# 3. predict binary + multiclass labels
preds <- bundle$predict_models(bundle, tokens_test)

# 4. export in the required format
pred_df <- tibble(.id         = parsed_test$.id,
                  bclass.pred = unname(preds$bclass.pred),
                  mclass.pred = unname(preds$mclass.pred))

stopifnot(nrow(pred_df) == nrow(claims_test), !anyNA(pred_df))
save(pred_df, file = 'results/preds-group4.RData')

print(head(pred_df))
print(table(pred_df$bclass.pred))
print(table(pred_df$mclass.pred))
