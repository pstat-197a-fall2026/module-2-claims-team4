## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

sel_paragraphs <- 'p'
sel_headers_paragraphs <- 'p, h1, h2, h3, h4, h5, h6'

# function to parse html and clean text
parse_fn <- function(.html, .selector = sel_paragraphs){
  read_html(.html) %>%
    html_elements(.selector) %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data <- function(.df, .selector = sel_paragraphs){
  safe_parse_fn <- function(.html){
    tryCatch(parse_fn(.html, .selector), error = function(e) NA_character_)
  }
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = safe_parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  n_fail <- sum(is.na(out$text_clean))
  if(n_fail > 0){
    warning(n_fail, ' page(s) failed to parse and were given empty text')
    out$text_clean[is.na(out$text_clean)] <- ''
  }
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

###

# ordered stream of cleaned tokens (stopwords removed, lemmatized, length > 2),
# one row per token occurrence, in document order. Same cleaning as nlp_fn().
token_stream_fn <- function(parse_data.out){
  stream <- parse_data.out %>%
    ungroup() %>%
    transmute(.id = as.character(.id), text_clean) %>%
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]'))
  # lemmatize each distinct token once (much faster than per occurrence)
  lemmas <- tibble(token = unique(stream$token)) %>%
    mutate(token.lem = lemmatize_words(token))
  stream %>%
    left_join(lemmas, by = 'token') %>%
    filter(str_length(token.lem) > 2) %>%
    select(.id, token.lem)
}

# document-term counts for single words
word_counts_fn <- function(.stream){
  .stream %>% count(.id, term = token.lem, name = 'n')
}

# document-term counts for bigrams: pairs of *adjacent tokens in the cleaned
# stream* (i.e. after stopword removal), formed within a document only
bigram_counts_fn <- function(.stream){
  .stream %>%
    group_by(.id) %>%
    mutate(next.lem = lead(token.lem)) %>%
    ungroup() %>%
    filter(!is.na(next.lem)) %>%
    count(.id, term = str_c(token.lem, next.lem, sep = ' '), name = 'n')
}

# tf-idf in long format; optionally drop terms appearing in < min_docs documents
# (document frequency uses no labels)
tfidf_long_fn <- function(.counts, min_docs = 1){
  .counts %>%
    bind_tf_idf(term = term, document = .id, n = n) %>%
    group_by(term) %>%
    filter(n_distinct(.id) >= min_docs) %>%
    ungroup() %>%
    filter(tf_idf > 0) %>%
    select(.id, term, tf_idf)
}

# long tf-idf -> sparse document-term matrix with one row per element of .ids
# (documents with no retained terms become all-zero rows, so rows line up
# across word and bigram matrices)
to_sparse_fn <- function(.long, .ids){
  .long <- .long %>% filter(.id %in% .ids)
  vocab <- sort(unique(.long$term))
  Matrix::sparseMatrix(i = match(.long$.id, .ids),
                       j = match(.long$term, vocab),
                       x = .long$tf_idf,
                       dims = c(length(.ids), length(vocab)),
                       dimnames = list(.ids, vocab))
}