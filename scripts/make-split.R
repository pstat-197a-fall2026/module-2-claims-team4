## Creates the ONE shared train / validation / test split used by every model
## (prelim tasks 1-3 and the primary task), so all models are trained on the
## same 70% of pages, tuned on the same 10%, and scored on the same 20%.
##
## Output: data/split-ids.csv with columns .id, bclass, mclass, split
##   split is one of 'train' (70%), 'val' (10%), 'test' (20%)
##
## The split is over ALL 2,165 labeled pages and is stratified by mclass
## (which also balances bclass, since bclass is a coarsening of mclass).
## Scripts that drop pages with no text simply filter this file; they must not
## re-split.
##
## Run from the repo root:  Rscript scripts/make-split.R

library(tidyverse)

seed  <- 110122
props <- c(train = 0.7, val = 0.1, test = 0.2)

load('data/claims-raw.RData')

set.seed(seed)
split_ids <- claims_raw %>%
  select(.id, bclass, mclass) %>%
  mutate(.id = as.character(.id)) %>%
  group_by(mclass) %>%
  mutate(split = {
    n <- n()
    n_train <- round(props[['train']] * n)
    n_val   <- round(props[['val']] * n)
    sample(rep(c('train', 'val', 'test'),
               c(n_train, n_val, n - n_train - n_val)))
  }) %>%
  ungroup() %>%
  arrange(.id)

write_csv(split_ids, 'data/split-ids.csv')

# sanity checks
stopifnot(!anyDuplicated(split_ids$.id), nrow(split_ids) == nrow(claims_raw))
cat('Pages per split:\n')
print(count(split_ids, split) %>% mutate(prop = round(n / sum(n), 3)))
cat('\nmclass proportions within each split:\n')
print(split_ids %>% count(split, mclass) %>% group_by(split) %>%
        mutate(prop = round(n / sum(n), 3)) %>%
        select(-n) %>% pivot_wider(names_from = split, values_from = prop))
