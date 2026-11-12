## Preprocessing for the primary task (binary + multiclass models)
## Intended to be sourced. Depends only on tidyverse, rvest/xml2, Matrix, SnowballC.
##
## Differences from the starter scripts/preprocessing.R:
##  * every page is parsed (the starter drops pages without '<!' in the HTML,
##    which would leave 14 test pages without a prediction)
##  * <script>, <style> and <noscript> nodes are removed before extracting text
##  * page title and meta/og descriptions are captured as a separate field,
##    alongside headers + paragraphs and the full visible body text
##  * tokens are stemmed (SnowballC) instead of lemmatized (textstem)

suppressPackageStartupMessages({
  library(tidyverse)
  library(rvest)
  library(xml2)
  library(Matrix)
})

# ---- text cleaning ---------------------------------------------------------
clean_text <- function(x){
  x %>%
    str_replace_all('(https?://|www\\.)\\S+', ' ') %>%          # urls
    str_replace_all('\\S+@\\S+\\.\\S+', ' ') %>%                # emails
    str_remove_all("'") %>%
    str_replace_all('([a-z])([A-Z])', '\\1 \\2') %>%            # camelCase
    str_replace_all('[^A-Za-z]+', ' ') %>%                      # keep letters only
    tolower() %>%
    str_squish()
}

# ---- HTML parsing ----------------------------------------------------------
# returns a one-row tibble: meta (title + descriptions), hp (headers + paragraphs),
# body (all visible body text). Non-HTML pages (JSON, error strings) fall back to
# their raw text with tags stripped.
parse_fields <- function(.html){
  empty <- tibble(meta = '', hp = '', body = '')
  if(is.na(.html) || !nzchar(.html)) return(empty)
  doc <- tryCatch(read_html(.html), error = function(e) NULL)
  if(is.null(doc)){
    txt <- clean_text(str_replace_all(.html, '<[^>]+>', ' '))
    return(tibble(meta = '', hp = txt, body = txt))
  }
  xml_remove(html_elements(doc, 'script, style, noscript'))
  title <- html_text2(html_elements(doc, 'title'))
  desc  <- html_attr(html_elements(doc,
                                   'meta[name="description"], meta[property="og:description"], meta[property="og:title"], meta[name="keywords"]'),
                     'content')
  hp    <- html_text2(html_elements(doc, 'p, h1, h2, h3, h4, h5, h6'))
  body  <- html_text2(html_elements(doc, 'body'))
  collapse <- function(v) str_c(unique(na.omit(v)), collapse = ' ')
  tibble(meta = clean_text(collapse(c(title, desc))),
         hp   = clean_text(collapse(hp)),
         body = clean_text(collapse(body)))
}

# apply to a data frame with columns .id and text_tmp
parse_pages <- function(.df, verbose = TRUE){
  n <- nrow(.df)
  out <- vector('list', n)
  for(i in seq_len(n)){
    out[[i]] <- parse_fields(.df$text_tmp[i])
    if(verbose && i %% 250 == 0) message('parsed ', i, ' / ', n)
  }
  bind_cols(tibble(.id = as.character(.df$.id)), bind_rows(out))
}

# ---- tokenization ----------------------------------------------------------
stop_set <- unique(clean_text(tidytext::stop_words$word))

# character vector of documents -> list of token vectors (stopwords removed,
# length > 2, stemmed). With bigrams = TRUE, adjacent-token bigrams are added.
# `prefix` lets one field's tokens be kept distinct from another's (e.g. 'm_').
tokenize_docs <- function(docs, bigrams = TRUE, prefix = '', stem = TRUE){
  lapply(str_split(docs, ' '), function(tok){
    tok <- tok[nchar(tok) > 2 & !(tok %in% stop_set)]
    if(stem && length(tok)) tok <- SnowballC::wordStem(tok, language = 'english')
    out <- tok
    if(bigrams && length(tok) > 1)
      out <- c(out, paste(tok[-length(tok)], tok[-1], sep = '_'))
    if(nzchar(prefix) && length(out)) out <- paste0(prefix, out)
    out
  })
}

# ---- tf-idf vectorizer (fit on training documents only) ---------------------
# sublinear tf (1 + log tf), smoothed idf, rows L2-normalised
fit_vectorizer <- function(tokens, min_df = 3, max_df_prop = 0.95){
  df <- table(unlist(lapply(tokens, unique)))
  n <- length(tokens)
  keep <- df[df >= min_df & df <= max_df_prop * n]
  vocab <- sort(names(keep))
  idf <- log((1 + n) / (1 + as.numeric(keep[vocab]))) + 1
  list(vocab = vocab, idf = setNames(idf, vocab))
}

transform_vectorizer <- function(vec, tokens){
  ii <- rep(seq_along(tokens), lengths(tokens))
  jj <- match(unlist(tokens), vec$vocab)
  ok <- !is.na(jj)
  X <- sparseMatrix(i = ii[ok], j = jj[ok], x = 1,
                    dims = c(length(tokens), length(vec$vocab)))
  X <- as(X, 'CsparseMatrix')               # duplicates are summed -> raw counts
  X@x <- (1 + log(X@x))
  X <- X %*% Diagonal(x = vec$idf)
  nrm <- sqrt(rowSums(X^2)); nrm[nrm == 0] <- 1
  X <- Diagonal(x = 1 / nrm) %*% X
  colnames(X) <- vec$vocab
  as(X, 'CsparseMatrix')
}