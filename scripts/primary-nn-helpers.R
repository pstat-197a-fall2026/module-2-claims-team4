## Helpers shared by the primary-task neural net training and prediction scripts.
## Intended to be sourced (after scripts/preprocessing-primary-jocelyn.R).

# Turn parsed pages (output of parse_pages(): columns .id, meta, hp, body) into
# one whitespace-separated string of stemmed tokens per page, which is what the
# saved model takes as input.
#   * hp   = headers + paragraphs (the prelim 1 winner)
#   * meta = page title + description tags; its tokens get an 'm_' prefix so a
#            word in the title is a different feature from the same word in the body
# Bigrams are NOT added here: the model's text-vectorization layer makes them.
nn_text_fn <- function(parsed){
  hp   <- tokenize_docs(parsed$hp,   bigrams = FALSE)
  meta <- tokenize_docs(parsed$meta, bigrams = FALSE, prefix = 'm_')
  map2_chr(meta, hp, ~ str_c(c(.x, .y), collapse = ' '))
}
