## PRIMARY TASK: generate binary + multiclass predictions on the test pages
## Run from the repository root after scripts/train-primary-jocelyn.R.
## Output: results/preds-group4.RData containing pred_df with columns
##   .id, bclass.pred, mclass.pred   (one row for every test page)

source('scripts/preprocessing-primary-jocelyn.R')
suppressPackageStartupMessages(library(e1071))

bundle <- readRDS('results/primary-models.rds')
load('data/claims-test.RData')

# parse every test page (no pages are dropped), tokenize, predict
parsed_test <- parse_pages(claims_test)
preds <- bundle$predict_models(bundle, bundle$make_tokens(parsed_test))

pred_df <- parsed_test %>%
  select(.id) %>%
  bind_cols(preds %>% select(bclass.pred, mclass.pred))

stopifnot(nrow(pred_df) == nrow(claims_test), !anyNA(pred_df))
save(pred_df, file = 'results/preds-group4.RData')

print(table(pred_df$bclass.pred))
print(table(pred_df$mclass.pred))