##############################################################################
## PRIMARY TASK OVERVIEW: how the scripts fit together
## PSTAT 197A, Module 2 (claims classification)
##
## Run this file from the repository root (the folder
## that contains data/, scripts/ and results/). By default it only prints a
## summary; set the switches in STEP 0 to TRUE to actually train or predict.
##############################################################################


## ---------------------------------------------------------------------------
## THE BIG PICTURE
## ---------------------------------------------------------------------------
## We have two piles of web pages:
##
##   * 2,165 LABELLED pages   (data/claims-raw.RData)
##       We know the right answers. Used to BUILD and GRADE the model.
##
##   * 929 UNLABELLED pages   (data/claims-test.RData)
##       New observations. The finished model CATEGORIZES these.
##
## Three scripts handle this:
##   1. scripts/preprocessing-primary-jocelyn.R   shared toolbox (turns pages into numbers)
##   2. scripts/train-primary-jocelyn.R    builds + checks the model
##   3. scripts/prediction.R         uses the finished model on new pages


## ---------------------------------------------------------------------------
## SCRIPT 1: preprocessing-primary-jocelyn.R  (the shared toolbox)
## ---------------------------------------------------------------------------
## Never run directly; the other two scripts load it with source().
## It turns messy HTML into numbers a model can use:
##   1. Pull out readable text: title, meta description, headers, paragraphs
##   2. Clean it: remove code, links, numbers, punctuation; lowercase
##   3. Split into words, drop stopwords, stem words, add word pairs (bigrams)
##   4. Turn each page into a row of TF-IDF numbers
## Both other scripts use it so labelled and new pages get IDENTICAL treatment.


## ---------------------------------------------------------------------------
## SCRIPT 2: train-primary-jocelyn.R  (building and checking the model)
## ---------------------------------------------------------------------------
## Uses only the 2,165 labelled pages:
##   1. Preprocess them with the toolbox
##   2. Check accuracy with 5-fold cross-validation: hide 1/5 of the pages,
##      train on the rest, see how many hidden pages the model gets right;
##      repeat 5 times.
##        -> binary accuracy     ~84.5%
##        -> multiclass accuracy ~83.6%
##   3. Train the final models on all 2,165 pages:
##        * binary SVM: relevant or N/A?
##        * 4-class SVM (trained on relevant pages only): which kind of claim?
##   4. Save both, plus the word list, to results/primary-models.rds
## The model is now finished. Takes ~15 minutes (mostly cross-validation).


## ---------------------------------------------------------------------------
## SCRIPT 3: prediction.R  (using the finished model)
## ---------------------------------------------------------------------------
## Uses the 929 new pages:
##   1. Load the finished model (no learning happens here)
##   2. Preprocess the new pages with the same toolbox and same word list
##   3. Categorize each page:
##        * binary SVM: relevant or N/A?
##        * if relevant, the 4-class SVM picks the category;
##          if N/A, both labels are N/A
##   4. Save the answers as pred_df (.id, bclass.pred, mclass.pred)
##      to results/preds-group4.RData
## We don't know which answers are right (no labels). Based on cross-validation
## we expect about 84% to be correct; the instructors will check them.


## ---------------------------------------------------------------------------
## IN ONE LINE EACH
## ---------------------------------------------------------------------------
##   preprocessing-primary-jocelyn.R   turns web pages into numbers
##   train-primary-jocelyn.R    learns from labelled pages, measures accuracy,
##                             saves the model
##   prediction.R         uses the saved model to label new pages,
##                             saves the answers


## ---------------------------------------------------------------------------
## STEP 0: switches (set to TRUE to run that step)
## ---------------------------------------------------------------------------
run_training   <- FALSE   # ~15 min; only needed once or after changing the model
run_prediction <- FALSE   # ~2-5 min; needs results/primary-models.rds


## ---------------------------------------------------------------------------
## STEP 1: train the models (Script 2)
## ---------------------------------------------------------------------------
if(run_training){
  source('scripts/train-primary-jocelyn.R')
}


## ---------------------------------------------------------------------------
## STEP 2: predict on the test pages (Script 3)
## ---------------------------------------------------------------------------
if(run_prediction){
  source('scripts/prediction.R')
}


## ---------------------------------------------------------------------------
## STEP 3: look at what exists so far
## ---------------------------------------------------------------------------
if(file.exists('results/primary-cv-summary.csv')){
  cat('\nCross-validated accuracy estimates:\n')
  print(read.csv('results/primary-cv-summary.csv'))
}

if(file.exists('results/primary-models.rds')){
  bundle <- readRDS('results/primary-models.rds')
  cat('\nSaved model: trained on', bundle$info$trained_on, 'pages,',
      bundle$info$vocab_size, 'terms in the vocabulary\n')
}

if(file.exists('results/preds-group4.RData')){
  load('results/preds-group4.RData')
  cat('\nPredictions (pred_df):', nrow(pred_df), 'test pages\n')
  print(head(pred_df))
  cat('\nBinary predictions:\n');     print(table(pred_df$bclass.pred))
  cat('\nMulticlass predictions:\n'); print(table(pred_df$mclass.pred))
}