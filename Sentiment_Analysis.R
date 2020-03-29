# Executive Summary

# This is a sentiment analysis about an Amazon data sample provided 
# in the UCI Machine Learning Repository.1,000 reviews have received 
# sentiment orientation, i.e. "positive" or "negative". The challenge 
# in this project is maximize accuracy on predicting sentiment orientation 
# on a validation set of one third. Accuracy is representative since
# prevalence of positive sentiment orientation is 50 %. 
# An accuracy level of 84 % has been  reached on the validation set 
# at the end of a three-tier analysis.

# First, Natural Language Processing has been conducted in terms of
# lowercasing, punctuation removal, stemming, tokenization and bag of words,
# followed by a first CART prediction on the training set. Some fine tuning 
# proved necessary to cope with short forms (intra word contractions), 
# punctuation marks stuck to words, alternative grammar, etc. 

# Second, text mining has focused on word frequency, wordclouds, decision trees and
# analysis of false positives and of false negatives, which have 
# been pinpointed as a weak point. Text ming has opened up two
# avenues for improvement: reintegrating negational unigrams ("not", etc.)
# and text classification. Text classification applied to unigrams 
# or multigrams conveying some subjective information that were present 
# in false negatives or positives but didn't show in decision trees; 
# they have been classified as negative of positive sentiment orientation
# and replaced with one generic negative sentiment token and one
# generic positive sentiment token. Running CART again propelled 
# accuracy to higher levels. 

# Third, machine learning models were then compared in accuracy and,
# complementarily, in other performance metrics. Three out of ten 
# have been picked up: svmRadialCost, which delivered 
# the highest accuracy level, rf and xbgBoost, which produced 
# the highest specificity. To keep on the safe line, an ensemble model
# has been built up by majority vote. The ensemble model 
# has produced an accuracy level of 87 %, 
# to be compared with 50 % for the basiline model. 

# Downloading packages.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")
if(!require(SnowballC)) install.packages("SnowballC", repos = "http://cran.us.r-project.org")
if(!require(wordcloud)) install.packages("wordcloud", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(fastAdaboost)) install.packages("fastAdaboost", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(monmlp)) install.packages("monmlp", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(utf8)) install.packages("utf8", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(kernlab)
library(fastAdaboost)
library(randomForest)
library(gbm)
library(xgboost)
library(monmlp)
library(kableExtra)
library(gridExtra)
library(utf8)

# Data has been downloaded from the UCI Machine Learning Repository: 
# https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences 
# and have been uploaded onto my GitHub repository 
# https://github.com/Dev-P-L/Sentiment-Analysis
# under the name "amazon_cells_labelled.txt". 

# Let's retrieve amazon_cells_labelled.txt from my GitHub repository by accessing 
# https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/amazon_cells_labelled.txt.
myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis-In-Progress/master/amazon_cells_labelled.txt"
reviews <- read.delim(myfile, header = FALSE, sep = "\t", quote = "", 
                      stringsAsFactors = FALSE)
rm(myfile)

reviews <- reviews %>% 
  `colnames<-`(c("text", "sentiment")) %>%
  mutate(sentiment = as.factor(gsub("1", "Appreciating", 
         gsub("0", "Critisizing", sentiment)))) %>% as.data.frame()

## CREATING TRAINING INDEX AND VALIDATION INDEX

set.seed(1)
ind_train <- createDataPartition(y = reviews$sentiment, 
                                 times = 1, p = 2/3, list = FALSE)
ind_val <- as.integer(setdiff(1:nrow(reviews), ind_train))

# ind_train allows to select the reviews that will be used for training, 
# be it in natural language processing, in text mining or in 
# machine learning.

## NATURAL LANGUAGE PROCESSING 

# Creating corpus.
# Corpus is created on training reviews only to avoid any interference
# between training reviews and validation reviews. Otherwise, 
# validation token frequencies could slightly impact token selection
# when applying the sparsity threshold. 
reviews_train <- reviews[ind_train, ]
corpus <- VCorpus(VectorSource(reviews_train$text)) 

# Lowercasing, removing punctuation and stopwords, stemming document.
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

# Building up a bag of words in a Document Term Matrix.
dtm <- DocumentTermMatrix(corpus)

# Managing sparsity with sparsity threshold. 
sparse <- removeSparseTerms(dtm, 0.995)

# Converting sparse, which is a DocumentTermMatrix, 
# to a matrix and then to a data frame.
sentSparse <- as.data.frame(as.matrix(sparse)) 

# Making all column names R-friendly.
colnames(sentSparse) <- make.names(colnames(sentSparse))

# The results from the NLP process need checking. 
# Let's build up a wordcloud with the most frequent tokens.
set.seed(1)
wordcloud(colnames(sentSparse), colSums(sentSparse), min.freq = 10, 
          max.words = 50, random.order = FALSE, rot.per = 1/3, 
          colors = brewer.pal(8, "Dark2"), scale = c(4,.5))

# Some tokens were not expected, 
# such as "dont" or "ive" since they seem to originate in short forms 
# and were expected to have been eliminated by stopwords being removed. 

# Let's start with "dont". Which rows from the training reviews
# have a 1 in the column "dont"? This would mean that 
# the corresponding NLP-transformed reviews contain "dont".
bin <- which(sentSparse$dont == 1)

# How many rows contain " dont "?
df <- data.frame(length(bin)) %>% 
  `colnames<-`('Occurences of "dont"') %>%
  `rownames<-`("sentSparse")
knitr::kable(df, "pandoc")

# Which are the corresponding rows in untransformed reviews? 
# Let's see the first one.
df <- data.frame(reviews_train$text[bin[1]]) %>%
  `colnames<-`('First Review Generating "dont"') %>%
  `row.names<-`("reviews_train")
knitr::kable(df, "pandoc")

# "dont" contains a spelling error or is "alternative" grammar. 
# Nevertheless, ideally, it should be treated as a short form
# from standard grammar, i.e. as the short form "don't". 
# Consequently, if we want to eradicate short forms, we'll have to complement
# stopwords with variants such as "dont", "couldnt".
# This will be done under the form of an additional stopword file called 
# "extra_stopwords.csv". 

# Let's see the second review that, after NLP, generates "dont" .
df <- data.frame(reviews_train$text[bin[2]]) %>% 
  `colnames<-`('Second Review Generating "dont"')
knitr::kable(df, "pandoc")
rm(bin, df)

# This is another scenario: "don't" has been written in the standard way,
# but all punctuation marks have been removed, consequently it has become "dont"
# and is no longer identical to the stopword "don't" and will not
# be removed. Consequently, stopwords containing apostrophes
# should be removed before removing punctuation. Or punctuation marks should be
# removed except for apostrophes and hyphens by using parameters of 
# the function removePunctuation() or other coding. 
# A solution will be applied. 

# Let's now analyze all tokens emanating from reviews_train,
# not jus the most frequent ones. For brevity, only impactful
# results will be showcased. 

# Let's prepare a presentation table. 
# Collecting all tokens (in this way or in other ways).
tokens <- findFreqTerms(dtm, lowfreq = 1)

# Definign the number of columns of the presentation table. 
nc <- 5           

# Calculating the number of missing values to get a full matrix
# and adding hyphens accordingly to get a full matrix. 
mis <- ((ceiling(length(tokens) / nc)) * nc) - length(tokens)
tokens <- as.character(c(tokens, rep("-", mis)))
tokens <- data.frame(matrix(tokens, ncol = nc, byrow = TRUE)) %>%
  `colnames<-`(NULL)
rm(nc, mis)

# There are several unigrams that seem to originate from bigrams,
# e.g. "brokeni" at row 24.
knitr::kable(tokens[24, ], "pandoc")

# Where does "brokeni" come from? 
v <- 1:nrow(reviews_train)
string <- "brokeni"
for(i in 1:nrow(reviews_train)) {
  v[i] <- length(grep(string, corpus[[i]]$content))
}

df <- data.frame(reviews_train$text[which(v == 1)], stringsAsFactors = FALSE) %>%
  `colnames<-`('Review Producing "brokeni"')
knitr::kable(df, "pandoc")

# What happened? Well, "broken...I" was first lowercased to 
# "broken...i", then punctuation was removed 
# by the function removePunctuation, which does not insert any 
# white space character, and "broken...i" has become "brokeni". 

# In further text mining and machine learning steps, "brokeni" 
# will be treated differently than "broken", which can be 
# seen on the same row, i.e. on row 24 of the tokens from the training reviews.
# This has to be changed: instead of the function removePunctuation(),
# specific and different for loops will be developped, replacing 
# punctuation marks with white space characters instead of just removing punctuation
# marks and allowing for differentiated treatment of on the one hand 
# apostrophes and on the other hand other punctuation marks.

# A result similar to "brokeni" can be seen on row 3: it is "abovepretti".
knitr::kable(tokens[3, ], "pandoc")
rm(v, i)

# Where does "abovepretti" come from? 
v <- 1:nrow(reviews_train)
string <- "abovepretti"

for(i in 1:nrow(reviews_train)) {
  v[i] <- length(grep(string, corpus[[i]]$content))
}

df <- data.frame(reviews_train$text[which(v == 1)]) %>%
  `colnames<-`('Review Generating "abovepretti"')
knitr::kable(df, "pandoc")
rm(tokens, v, i, string, df)
rm(corpus, dtm, sparse, sentSparse)

# Therefore, the preprocessing process will be rerun
# with for loops that replace punctuation marks with white space characters
# and allowing for differentiated treatment of 
# on the one hand apostrophes and hyphens (dashes) 
# and on the other hand other punctuation marks.
# Stopwords will be split into stopwords with apostrophe,
# stopwords without apostrophe, negational stopwords
# and extra stopwords (short forms written without apostrophe). 

################################
################################

# COMPLEMENTING NLP

# Buildind up extra stopwords file: 19 extra stopwords,
# which are short forms without apostrophe like "isnt". 
# With this file, short forms such as "isnt" can be removed. 

# Moreover, stopwords have been split into 3 files: 
# - stopwords_with_apostrophe.csv,
# - stopwords_without_apostrophe.csv
# - and stopwords_negation.csv.

# The 4 files have been uploaded to my GitHub repository,
# https://github.com/Dev-P-L/Sentiment-Analysis.
# They are going to be downloaded now and integrated into
# the NLP transformation.

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/stopwords_with_apostrophe.csv"
stopwords_with_apostrophe <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
stopwords_with_apostrophe <- stopwords_with_apostrophe[, 2] %>% as.vector()
# Converting possible curly apostrophes to straight apostrophes. 
stopwords_with_apostrophe <- sapply(stopwords_with_apostrophe, utf8_normalize, map_quote = TRUE)

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/stopwords_without_apostrophe.csv"
stopwords_without_apostrophe <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
stopwords_without_apostrophe <- stopwords_without_apostrophe[, 2] %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/stopwords_negation.csv"
stopwords_negation <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
stopwords_negation <- stopwords_negation[, 2] %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/extra_stopwords.csv"
extra_stopwords <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
extra_stopwords <- extra_stopwords[, 2] %>% as.vector()
rm(myfile)

# Creating and preprocessing corpus.
corpus_av0 <- VCorpus(VectorSource(reviews_train$text)) 
corpus_av0 <- tm_map(corpus_av0, content_transformer(tolower))

# Replacing all punctuation marks other than apostrophe
# with white space characters,
# instead of simply suppressing punctuation marks, 
# to prevent proces from generating tokens like "brokeni".
# Keeping apostrophes to leave intact short forms such as "don't".
for (i in 1:nrow(reviews_train)) {
  corpus_av0[[i]]$content <- gsub("(?!')[[:punct:]]", " ", 
                              corpus_av0[[i]]$content, perl = TRUE)
}
rm(i)

# Removing stopwords containing one apostrophe. 
corpus_av0 <- tm_map(corpus_av0, removeWords, stopwords_with_apostrophe)

# Replacing all remaining apostrophes with white space characters (there might
# be other apostrophes than in short forms...). 
for (i in 1:nrow(reviews_train)) {
  corpus_av0[[i]]$content <- gsub("[[:punct:]]", " ", corpus_av0[[i]]$content)
}
rm(i)

# Removing stopwords from other files. 
corpus_av0 <- tm_map(corpus_av0, removeWords, stopwords_without_apostrophe)
corpus_av0 <- tm_map(corpus_av0, removeWords, stopwords_negation)
corpus_av0 <- tm_map(corpus_av0, removeWords, extra_stopwords)

# Stemming words.
corpus_av0 <- tm_map(corpus_av0, stemDocument)

# Removing numbers and extra white space characters (all white spaces except one
# of them in a sequence of white space characters).
corpus_av0 <- tm_map(corpus_av0, removeNumbers)
corpus_av0 <- tm_map(corpus_av0, stripWhitespace)

# Building up a bag of words in a Document Term Matrix.
dtm_av0 <- DocumentTermMatrix(corpus_av0)

# Managing sparsity with the sparsity threshold. 
sparse_av0 <- removeSparseTerms(dtm_av0, 0.995)

# Converting sparse, which is a DocumentTermMatrix, 
# to a matrix and then to a data frame.
sentSparse_av0 <- as.data.frame(as.matrix(sparse_av0)) 

# Making all column names R-friendly.
colnames(sentSparse_av0) <- make.names(colnames(sentSparse_av0))

# Let's check whether shortcomings have disappeared or not. 
# Let's build up a wordcloud with the most frequent tokens
# from the training reviews.
set.seed(1)
wordcloud(colnames(sentSparse_av0), colSums(sentSparse_av0), min.freq = 10, 
          max.words = 50, random.order = FALSE, rot.per = 1/3, 
          colors = brewer.pal(8, "Dark2"), scale = c(4,.5))

# In the wordcloud, there is no more token originating from short forms.
# Let's have a broader look at all tokens and build up
# a presentation table. 

# Retrieving all tokens.
tokens <- findFreqTerms(dtm_av0, lowfreq = 1)

# Choosing the number of columns of the presentation table. 
nc <- 5

# Calculating the number of missing tokens to have a full matrix. 
mis <- ((ceiling(length(tokens) / nc)) * nc) - length(tokens)

# Builing up presentation table.
tokens <- as.character(c(tokens, (rep("-", mis))))
tokens <- data.frame(matrix(tokens, ncol = nc, byrow = TRUE)) %>%
  `colnames<-`(NULL) %>% `rownames<-`(NULL)

# "dont" has disappeared:
knitr::kable(tokens[51, ], "pandoc")

# "ive" too
knitr::kable(tokens[96, ], "pandoc")

# As well as "brokeni"
knitr::kable(tokens[21, ], "pandoc")

# And "abovepretti"
knitr::kable(tokens[1, ], "pandoc")
rm(tokens, nc, mis)

# As well as many other oddities. 
#` I leave uncorrected some spelling errors, such as among others
# "disapoint" or "dissapoint".

# Let's have a first try at predicting sentiment on the basis of sentSparse_av0. 

# Adding dependent variable.
sentSparse_av0 <- sentSparse_av0 %>% mutate(sentiment = reviews_train$sentiment)

# Training CART with the algorithm rpart.
set.seed(1)
fit_cart_av0 <- rpart(sentiment ~., data = sentSparse_av0)
fitted_cart_av0 <- predict(fit_cart_av0, type = "class")
cm_cart_av0 <- confusionMatrix(fitted_cart_av0, sentSparse_av0$sentiment)

# The accuracy level is 
df <- data.frame(cm_cart_av0$overall["Accuracy"]) %>%
  `rownames<-`("Model rpart") %>% `colnames<-`("Accuracy")
knitr::kable(df, "pandoc")

# Training CART with the algorithm rpart, the train() function from caret
# which tuning on 15 values of cp parameter
# and automatically for each value training on 25 bootstrapped resamples. 
set.seed(1)
fit_cart_tuned_av0 <- train(sentiment ~ .,
                         method = "rpart",
                         data = sentSparse_av0,
                         tuneLength = 15,
                         metric = "Accuracy")
fitted_cart_tuned_av0 <- predict(fit_cart_tuned_av0)
cm_cart_tuned_av0 <- confusionMatrix(as.factor(fitted_cart_tuned_av0), 
                                     as.factor(sentSparse_av0$sentiment))

# The rpart tuned model delivers an accuracy level of
df <- data.frame(cm_cart_tuned_av0$overall["Accuracy"]) %>%
  `rownames<-`("Tuned CART model") %>% `colnames<-`("Accuracy")
knitr::kable(df, "pandoc")
rm(df)

# Accuracy in function of cp value is shown on the graph below.
graph <- ggplot(fit_cart_tuned_av0)
graph
rm(graph)

# Maximizing value is near zero. Getting more precision
# about cp of the final model.
fit_cart_tuned_av0$bestTune
df <- data.frame(fit_cart_tuned_av0$bestTune) %>%
  `rownames<-`("CART with Tuning on cp") %>% `colnames<-`("Value of cp from Final Model")
knitr::kable(df, "pandoc")
rm(df)

# Actually, finer tuning has been tried a little bit above 0 
# but with accuracy results a little bit lower. 
# At the end of the day, the tuning organized by train() has been kept. 

# Accuracy is 78-79 % with models rpart and rpart tuned, 
# which is already sensibly higher than with the baseline model. 
# Baseline model: predicting "Appreciating" everywhere.
# The baseline model would give an accuracy level of
df <- sentSparse_av0
pred_baseline <- 
  data.frame(sentiment = rep("Appreciating", nrow(df))) %>%
  mutate(sentiment = factor(sentiment, levels = levels(df$sentiment)))
cm_baseline <- confusionMatrix(pred_baseline$sentiment, 
                               as.factor(df$sentiment)) 
df <- data.frame(cm_baseline$overall["Accuracy"]) %>%
      `rownames<-`("Baseline Model") %>% `colnames<-`("Accuracy")
knitr::kable(df, "pandoc")
rm(df)

# Let's summarize results. 
colname <- c("MODEL", "SHORT DESCRIPTION", "ACCURACY", "SENSITIVITY", 
             "NEG PRED VAL", "SPECIFICITY", "POS PRED VAL")
models <- c("baseline", "cart_av0", "cart_tuned_av0")
description <- c("baseline model", "rpart", "rpart + cp tuning")
cm <- c("cm_baseline", "cm_cart_av0", "cm_cart_tuned_av0")
tab <- data.frame(matrix(1:(length(colname) * length(models)),
                         ncol = length(colname), nrow = length(models)) * 1)

for (i in 1:3) {
  tab[i, 1] <- models[i]
  tab[i, 2] <- description[i]
  tab[i, 3] <- eval(parse(text = paste(cm[i], "$overall['Accuracy']", sep = "")))
  tab[i, 4] <- eval(parse(text = paste(cm[i], "$byClass['Sensitivity']", sep = "")))
  tab[i, 5] <- eval(parse(text = paste(cm[i], "$byClass['Neg Pred Value']", sep = "")))
  tab[i, 6] <- eval(parse(text = paste(cm[i], "$byClass['Specificity']", sep = "")))
  tab[i, 7] <- eval(parse(text = paste(cm[i], "$byClass['Pos Pred Value']", sep = "")))
}                 
                  
tab_av0 <- tab %>% mutate_at(vars(3:7), funs(round(., 4))) %>%
           `colnames<-`(colname)
knitr::kable(tab_av0, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(1, bold = T, strikeout = T, 
           color = "#9bc4e2", background = "blue") %>%
  row_spec(2, bold = T, color = "white", background = "blue") %>%
  row_spec(3, bold = T, color = "white", background = "green")
rm(cm_baseline, pred_baseline, models, description, cm, tab)

# In the table above, on row 1, fonts have been blurred 
# into light blue and have been stricken through to indicate 
# that this model has been discarded.
# The other two models should be seen as a cumulative process 
# bringing accuracy improvement in a stepwise and incremental way, with the one 
# on green background being the best in accuracy.

# Accuracy with the baseline model is 50 %, which reflects prevalence.
# Models 2 and 3 are sensibly higher in accuracy.
# Model 2 gives 78 % and model 3 79 % in accuracy. 

# With model 2 or 3, sensitivity and negative predictive value are lower
# than specificity and positive predictive value. This reflects 
# the false negatives being more numerous than the false positives. 
# False negatives =  predictions pointing to "Critisizing" 
# while the reference value is "Appreciating".
# This is an insight for text mining: perusing the false negatives
# and coming with some improvement. 

# Model 3 is a little more balanced betwwen sensitivity/negative predictive value
# and specificity/positive predictive value. 

# Let's have a look at the confusion matrix for both models.
# First, the rpart model
tab <- table(fitted_cart_av0, sentSparse_av0$sentiment) %>% as.vector()
tab <- data.frame(matrix(tab, ncol = 2, nrow = 2, byrow = FALSE)) %>%
  `colnames<-`(c("Appreciating (ref)", "Criticizing (ref)")) %>%
  `rownames<-`(c("Appreciating (pred)", "Criticizing (pred)"))
knitr::kable(tab, "pandoc")
rm(tab)

# As expected, the number of false negatives is relatively high 
# and anyway substantially higher than the number of false positives. 

# Now, the CART model with cp tuning
tab <- table(fitted_cart_tuned_av0, sentSparse_av0$sentiment) %>% as.vector()
tab <- data.frame(matrix(tab, ncol = 2, nrow = 2, byrow = FALSE)) %>%
  `colnames<-`(c("Appreciating (ref)", "Criticizing (ref)")) %>%
  `rownames<-`(c("Appreciating (pred)", "Criticizing (pred)"))
knitr::kable(tab, "pandoc")
rm(tab)
rm(cm_cart_av0, fit_cart_av0, fitted_cart_av0)

# Imbalance between false negatives and false positives is less acute 
# but nevertheless false negatives are twice as numerous as
# false positives. 

# The sum of the elements on the main diagonal larger with the second 
# CART model, which corresponds to higher accuracy. 

# These estimations have been coded as av0 for avenue 0 or
# basic performance yardstick. It is just to make things easier
# for readers interested in the code. 

# Indeed, insights from avenue 0 
# will contribute to elaborating avenues for further 
# accuracy improvement through text mining in the next section. 

## TEXT MINING

# In this section, informational insights will be sought 
# from the tokens themselves through perusing 
# - word frequency,
# - the decision tree from the final CART model,
# - false negatives and false positives from the final CART model. 

## Word frequency

# Let's have a second look at the wordcloud that was already
# used to check tokenization results. This time to retrieve information
# about types of tokens. 
df <- sentSparse_av0[, - ncol(sentSparse_av0)]
set.seed(1)
wordcloud(colnames(df), colSums(df), 
          min.freq = 10, max.words = 50, random.order = FALSE, rot.per = 1/3, 
          colors = brewer.pal(8, "Dark2"), scale = c(4,.5))
rm(df)

# Among tokens depicted in the wordcloud, there are 
# - topic-related tokens ("phone", "batteri", "headset", "sound", "ear", etc.),
# - intent-related tokens ("purchas", "buy", "bought"),
# - compliance-related tokens, expressing compliance or incompliance with 
# expectations, requirements or advertisements ("fit", "comfort", "problem", etc.),
# - sentiment-related tokens other than in previous category ("love", "like", etc.).

# The difference between category 3 and 4 can sometimes be tricky and both categories
# will be referred to altogether in this project using phrases such as 
# "subjective information" or "tokens conveying subjective information". 

# "phone", which is topic-related, is the token with the 
# highest frequency. Other topic-related tokens rank among the highest 
# frequencies: "batteri", "product", "headset", "sound", etc. 

# Some of these tokens can be visualized in decreasing order of frequency
# in the graph below. 
df <- sentSparse_av0[, - ncol(sentSparse_av0)]
freq <- data.frame(to = colnames(df), fre = as.integer(colSums(df)), 
                   stringsAsFactors = FALSE) %>% 
        arrange(desc(fre)) %>% head(., 30)
graph <-  freq %>% mutate(to = reorder(to, fre)) %>%
  ggplot(aes(to, fre)) + 
  geom_bar(stat = "identity", width = 0.80, color = "#007ba7", fill = "#9bc4e2") + 
  coord_flip() +
  ggtitle("Token Frequency") +
  xlab("Token") + ylab("Frequency") +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title.x = element_text(size = 16), axis.title.y = element_text(size = 16), 
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12), 
        axis.text.y = element_text(size = 12))
graph
rm(df, freq, graph)

# "phone" has the highest frequency, i.e. 124, with "batteri" at 34,
# "headset" at 32, etc. Is it a similar ranking 
# in the decision tree produced by the final CART model? 

prp(fit_cart_tuned_av0$finalModel, uniform = TRUE, cex = 0.6, 
    box.palette = "auto")

# "phone", "batteri", "headset" do not even appear in the decision tree!
# This difference between wordcloud and decision tree might be insightful.
# Should topic-related tokens be discarded except the few ones that appear 
# in the decision tree?  To get some confirmation or infirmation,
# let's apply another algorithm. A Random Forests model is chosen 
# because it allows for some rough ranking of predictor impact. 

# Random Forests model
fit_rf_av0 <- train(sentiment ~ ., method = "rf", data = sentSparse_av0) 
fitted_rf_av0 <- predict(fit_rf_av0)
cm_rf_av0 <- confusionMatrix(as.factor(fitted_rf_av0), 
                             as.factor(sentSparse_av0$sentiment))

# Getting a ranking of predictor importance and saving it for further use.
df <- varImp(fit_rf_av0)
df <- data.frame(t = rownames(df$importance), 
                 i = round(df$importance$Overall, 2), 
                 stringsAsFactors = FALSE) %>% 
      arrange(desc(i)) %>% 
      `colnames<-`(c("Stemmed Token", "Importance in rf"))
importance_rf <- knitr::kable(head(df, 20), "pandoc")
importance_rf
rm(df)

# Actually, the ranking of predictors by impact is rather different:
# e.g. "phone" is ranked in the 11th position. In the 16 first positions, 
# there are three topic-related tokens ("price", "phone" and "product")
# while there is none in the 16 first positions in the decision tree.
# Since ranking depends on models
# and since several models will be trained in the machine learning section,
# topic-related tokens that do not show in one decision tree will not be discarded. 

# Let's now have a look at the false negatives from the last CART model,
# which constitute a challenge. 
df <- sentSparse_av0 %>% mutate(pred = fitted_cart_tuned_av0) %>% as.data.frame()
FN_train <- ifelse(df$sentiment == "Appreciating", 1, 0) - 
  ifelse(df$pred == "Appreciating", 1, 0)
FN_train <- ifelse(FN_train == 1, 1, 0)

# Let's build up a sample taken out of all false negatives. 
sample_size <- 12
set.seed(1)
seq <- sort(sample(which(FN_train == 1), sample_size, replace = FALSE))
rm(FN_train)

# Let's save seq under the name seq_FN for further use. 
seq_FN <- seq

# Let's build up a presentation table.
df <- data.frame(matrix(nrow = sample_size, ncol = 2) * 1)

for (i in 1:length(seq)) {
  row <- as.numeric(seq[i])
  df[i, 1] <- reviews_train[row, 1]
  df[i, 2] <- corpus_av0[[row]]$content
}
rm(seq, sample_size, i, row)

colname_token <- c("REVIEW", "TOKENIZATION", "USABLE SUBJECTIVE INFORMATION")
comment <- c("super", "fast + faster", "prettier + sharp", "infatu", "wise",
             "awesom", "rock", "troubl", "?", "?", "secur", "quick")
df <- df %>% mutate(com = comment) %>% `colnames<-`(colname_token)
rm(comment)

# Let's save df under a less anonymous name for further use and print it. 
df_FN_cart <- df
rm(df)
df_FN_cart <- kable(df_FN_cart, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(1:7, bold = T, color = "white", background = "blue") %>%
  row_spec(8, bold = T, color = "white", background = "purple") %>%
  row_spec(9:10, bold = T, color = "white", background = "magenta") %>%
  row_spec(11:12, bold = T, color = "white", background = "blue")
df_FN_cart

# There are three scenarios.

# First, on the blue background: in 9 cases out of 12, there is at least 
# one token with subjective information that is clear from a human point of view. 
# "Unfortunately", the decision tree does not split on these
# tokens in the final CART model. 

# On the first row, the review is simple, from a human point of view.
# But in the decision tree, none of the three stemmed tokens appears!
# Even "super"! It might be impactful to garner subjective information
# conveyed by tokens such as "super". Since CART doesn't do it, 
# why not replace such tokens with either a generic positive 
# sentiment token or a generic negative token? 
# Examples of positive minded tokens can be "super", "faster", 
# "prettier", "infatu", "awesom", etc.). This would
# reduce the number of tokens conveying subjective information
# and provide rather high frequencies for both generic tokens. 

# In this project, polarity of some tokens conveying subjective information 
# will be inserted in additional files. That is one avenue of research. 

# A second scenario is exemplified on the purple background: "no trouble" is
# clear from a human point of view but this bigram has 
# become "troubl", the negational token "no" having been removed
# with all other stopwords. Even if "troubl" is polarized
# under a generic negative token, as suggested above, the right polarity 
# of "no trouble" wouldn't show. Two avenues are opened up:
# the whole bigram "no trouble" could be converted into 
# a generic positive token or negational stopwords such as "not" 
# should no longer be removed, whihc is another avenue for
# improvement. 

# In this sample, there is only one case out of 12 with an ignored
# negational phrase. Is it worthwhile heading 
# for properly dealing with negational phrases?  Let's have a look at 
# word associations with "not", of course after having taken "not" back
# into the corpus.

# Alternate dtm with negational words, i.e. after building up
# a corpus without removing tokens from the file stopwords_negation.csv. 
corpus_2 <- VCorpus(VectorSource(reviews_train$text)) 
corpus_2 <- tm_map(corpus_2, content_transformer(tolower))

# Replacing all punctuation marks with white space characters 
# to prevent proces from generating tokens like "brokeni".
# Keeping apostrophes to leave intact short forms such as "don't".
for (i in 1:nrow(reviews_train)) {
  corpus_2[[i]]$content <- gsub("(?!')[[:punct:]]", " ", 
                                  corpus_2[[i]]$content, perl = TRUE)
}

corpus_2 <- tm_map(corpus_2, removeWords, stopwords_with_apostrophe)

# Removing apostrophes as well (there can be apostrophes outside of
# short forms). 
for (i in 1:nrow(reviews_train)) {
  corpus_2[[i]]$content <- gsub("[[:punct:]]", " ", corpus_2[[i]]$content)
}
rm(i)

# Further NLP
corpus_2 <- tm_map(corpus_2, removeWords, stopwords_without_apostrophe)
corpus_2 <- tm_map(corpus_2, removeWords, extra_stopwords)
corpus_2 <- tm_map(corpus_2, stemDocument)
corpus_2 <- tm_map(corpus_2, removeNumbers)
corpus_2 <- tm_map(corpus_2, stripWhitespace)

dtm_2 <- DocumentTermMatrix(corpus_2)
sparse_2 <- removeSparseTerms(dtm_2, 0.995)
sentSparse_2 <- as.data.frame(as.matrix(sparse_2)) 
colnames(sentSparse_2) <- make.names(colnames(sentSparse_2))

# Would one negational unigrams appear in a wordcloud? 
wordcloud(colnames(sentSparse_2), colSums(sentSparse_2), min.freq = 10, 
          max.words = 50, random.order = FALSE, rot.per = 1/3, 
          colors = brewer.pal(8, "Dark2"), scale = c(4,.5))
# "not" is indeed rather frequent.

# Looking for word associations with "not".
df <- findAssocs(sparse_2, "not", 0.05)
df <- as.data.frame(df$not) %>% mutate(token = names(df$not)) %>%
  `colnames<-`(c("correlation", "token")) %>% select(token, correlation) %>%
  `colnames<-`(c("Token", 'Correlation with "not"'))
knitr::kable(df, "pandoc")
rm(df)

# The first stemmed unigram on the list is "impress". 
# Let's localize the reviews producing "impress". 
v <- 1:nrow(reviews_train)
string <- "impress"
for(i in 1:nrow(reviews_train)) {
  v[i] <- length(grep(string, corpus_2[[i]]$content))
}
v <- which(v == 1)

# Let's print these reviews. 
df <- data.frame(matrix(1:length(v), ncol = 1))
for(i in 1:length(v)) {
  df[i, 1] <- reviews_train[v[i], 1]
}

colnames(df) <- 'Training Reviews Containing "impressed"'
kable(df, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(c(1, 3:4), bold = T, color = "white", background = "blue") %>%
  row_spec(c(2, 5:6), bold = T, color = "white", background = "purple") 

rm(string, df, v, i)
rm(corpus_2, dtm_2, sparse_2, sentSparse_2)

# Among the training reviews, there are three reviews containing "not impressed".
# Moreover, in the second of these reviews, there is also "not recommend"
# and "not laughing". This illustrates the usefulness of including negational words
# in one way or another. 

# Let's go back to the table with false negatives 
# coming from the final CART model.
df_FN_cart
rm(df_FN_cart)

# A third scenario is exemplified on the red background. 
# There is no unigram that clearly conveys on its own sentiment orientation 
# in these reviews, at least from a human point of view. Sentiment 
# orientation is generated by associations of words: "obviously 
# knows what they're doing" or "exactly as described". Polarity would still
# be clear without "obviously" or "exactly". Consequently, just as suggested
# above, n-grams could also be replaced with a generic token expressing polarity. 
# Variants of these n-grams could be insterted
# as well such as "know what they are doing". By the way, 
# as already pointed out, this approach could also
# be applied to n-grams involving negational unigrams such as "no trouble"... 

# What about false positives originating from the final CART model?
# Let's localize them. 
df <- sentSparse_av0 %>% mutate(pred = fitted_cart_tuned_av0) %>% as.data.frame()
FP_train <- ifelse(df$sentiment == "Appreciating", 1, 0) - 
  ifelse(df$pred == "Appreciating", 1, 0)
FP_train <- ifelse(FP_train == -1, 1, 0)

# Let's generate a sample index of false positives at random. 
sample_size <- 12
set.seed(1)
seq <- sort(sample(which(FP_train == 1), sample_size, replace = FALSE))
rm(FP_train)

# Let's organize a presentation table, retrieve reviews 
# corresponding to the index sample and print the presentation table.
df <- data.frame(matrix(nrow = sample_size, ncol = 2) * 1)

for (i in 1:length(seq)) {
  row <- as.numeric(seq[i])
  df[i, 1] <- reviews_train[row, 1]
  df[i, 2] <- corpus_av0[[row]]$content
}

colname_token <- c("REVIEW", "TOKENIZATION", "USABLE SUBJECTIVE INFORMATION")
comment <- c("not a good", "whine + ? the less", "shouldn't", 
             "slow + crawl + lock-up", "didn't work well", "still waiting", 
             "terrible", "difficult", "wasn't always easy", "sorry", 
             "not as good", "don't like")
df <- df %>% mutate(com = comment) %>% `colnames<-`(colname_token)
kable(df, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(c(8, 10), bold = T, color = "white", background = "blue") %>%
  row_spec(c(1, 5, 6, 9, 11, 12), bold = T, color = "white", 
           background = "purple") %>%
  row_spec(c(2, 3, 4, 7), bold = T, color = "white", background = "magenta") 

rm(df, i, row, sample_size, seq)

# First scenario, on the blue background: at least one token per review
# expresses sentiment orientation, from a human point of view:
# "difficult" or "sorry". Recommendation: replacing with one generic token
# expressing criticism. Avenue 2.

# Second scenario, on the purple background: there is at least 
# one negational word per review. There are 6 cases in this category, 
# out of 12 cases in the whole table. This clearly illustrates shows that negational
# words and short forms should be taken into account. Negational stopwords
# should be reintegrated, including negational short forms, and 
# impact on accuracy should be tested. Moreover, it might
# be worthwhile replacing some stereotyped phrases with one generic token 
# (usually) expressing negative sentiment: e.g. "not a good", "did not work",
# "did not work well", "did not work very well", "still waiting", 
# "wasn't always easy", "not as good", "don't like", etc. Consequently
# both avenue 1 and avenue 2.

# Third scenario, on the magenta background: reviews with figurative wording,
# sarcasm, irony, metaphores, multi-faceted reviews, etc. 
# Examples: figurative "whine" instead of being disappointed, frustrated; 
# figurative "crawl"; sarcasm and methaphores about monkeys;
# multi-faceted review such as "My experience was terrible..... 
# This was my fourth bluetooth headset, and while it was much more comfortable 
# than my last Jabra (which I HATED!!!". 
# Suggestion: replacing some unigrams or n-grams
# with one generic negative token: e.g. "whine", "shouldn't", "lock up",
# etc. So, avenue 2.

# Before developping avenues 1 and 2, let's go back to the results
# from the rf model, which were better in accuracy. 

tab <- data.frame(cm_rf_av0$overall["Accuracy"]) %>%
  `colnames<-`("Model rf") %>% `rownames<-`("Accuracy")
knitr::kable(tab, "pandoc")
rm(tab)

# Would this mean that this model solves all issues and that further text mining 
# and further machine learning are both useless?
# I do not think so: very often, at least on the basis of my experience, 
# rf has a tendency to overfitting. Results on the validation set 
# might be substantially lower. Consequently, let's go on. 

# Let's first ckeck whether there are more false negatives or false positives
# and let's generate a confusion matrix.

tab <- table(fitted_rf_av0, sentSparse_av0$sentiment) %>% as.vector()
tab <- data.frame(matrix(tab, ncol = 2, nrow = 2, byrow = FALSE)) %>%
  `colnames<-`(c("Appreciating (ref)", "Criticizing (ref)")) %>%
  `rownames<-`(c("Appreciating (pred)", "Criticizing (pred)"))
knitr::kable(tab, "pandoc")
rm(tab)

# Not surprisingly, the number of both false negatives and 
# false positives is more limited than with CART since accuracy
# is larger. Let's focus on false negatives.  

# While it didn't show in the CART decision tree,
# "awesom" is in 19th position on the importance list from the rf model. 

importance_rf
 
# Maybe due to this position or maybe due to some other reason, there is 
# no false negative in the rf model for the corresponding review. 
# Let's check up on it. 
df <- data.frame(reviews_train[seq_FN[6], ], fitted_rf_av0[seq_FN[6]], 
                 stringsAsFactors = FALSE) %>%
      `colnames<-`(c("Review", "Label (ref)", "Prediction by rf"))
knitr::kable(df, "pandoc")
rm(df, seq_FN)

# It makes sense to increase the impact of such tokens as proposed in avenue 2.

# There still remain false negatives and false positives with the rf model.
# Let's have a look at the false negatives, which are more numerous. 
# First localizing false negatives emanating from the rf model. 
df <- sentSparse_av0 %>% mutate(pred = fitted_rf_av0) %>% as.data.frame()
FN_train <- ifelse(df$sentiment == "Appreciating", 1, 0) - 
  ifelse(df$pred == "Appreciating", 1, 0)
FN_train <- ifelse(FN_train == 1, 1, 0)

# Second, creatig index at random to select some false negatives
# although the number of false negatives is rather limited,
# but it is a way to stick to a previously used procedure. 
sample_size <- 12
set.seed(1)
seq <- sort(sample(which(FN_train == 1), sample_size, replace = FALSE))

# Retrieving reviews and NLP-transformed reviews. 
df <- data.frame(matrix(nrow = sample_size, ncol = 2) * 1)
for (i in 1:length(seq)) {
  row <- as.numeric(seq[i])
  df[i, 1] <- reviews_train[row, 1]
  df[i, 2] <- corpus_av0[[row]]$content
}
rm(FN_train, seq, sample_size, i, row)

# Naming columns and commenting on models.
colname_token <- c("REVIEW", "TOKENIZATION", "USABLE SUBJECTIVE INFORMATION")
comment <- c("simpler", "job done", "incred", "shouldv invent sooner", 
             "rock", "incredi", "fix problem", "fabul", "perfect", 
             "thumbs up", "any problem", "wonder")

# Finalizing and printing.
df <- df %>% mutate(com = comment) %>% `colnames<-`(colname_token)
kable(df, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(c(1, 3, 5:6, 8:9, 12), bold = T, color = "white", background = "blue") %>%
  row_spec(11, bold = T, color = "white", background = "purple") %>%
  row_spec(c(2, 4, 7, 10), bold = T, color = "white", background = "magenta")

rm(colname_token, comment, df)
rm(cm_cart_tuned_av0, fit_cart_tuned_av0, fitted_cart_tuned_av0)
rm(cm_rf_av0, fit_rf_av0, fitted_rf_av0, importance_rf)
rm(corpus_av0, dtm_av0, sparse_av0, sentSparse_av0)
 
# In blue: simple unigrams such as "incred", "fabul", "perfect" or "wonder"
# should an (at least partial) trigger of positive polarity; avenue 2 could help.

# In purple, negational structure. Negational short forms 
# should be reintegrated and/or bigram such as "any problem" should
# be positively polarized. Avenue 1 and/or avenue 2. 

# In magenta, some unigrams (such as "fix") or multigrams (such as 
# "job done") should be an (at least partial) driver of 
# positive sentiment orientation: avenue 2.

# This is more confirmation of previous insights than innovation.
# This also shows that even the rf model could benefit from 
# text mining, even if rf results probably incorporate some overfitting. 

###############################
###############################

# Avenue 1: negational n-grams reintegrated

# There are some negational short forms in the file stopwords_with_apostrophe.csv
# such as "couldn't" and some negational unigrams in the file stopwords_negation.csv. 
# such as "not". 

# The stopwords from both files will be reinserted and tested. 
# It will be done separately, in two steps, and the merits of both
# will be established separately. 

# Building corpus and lowercasing. 
corpus_av1_a <- VCorpus(VectorSource(reviews_train$text)) 
corpus_av1_a <- tm_map(corpus_av1_a, content_transformer(tolower))

# Here, stopwords with apostrophe, i.e. short forms or contractions, 
# are no longer removed. 

# To clearly visualize the contractions (short forms) in a wordcloud 
# or a list, another piece of code will be used: it preserves apostrophes. 

for (i in 1:nrow(reviews_train)) {
  corpus_av1_a[[i]]$content <- gsub("(?!')[[:punct:]]", " ", 
                                        corpus_av1_a[[i]]$content, perl = TRUE)
}
rm(i)

# Removing other stopwords, stemming, removing numbers, digits 
# and multiple white space characters (leaving only 
# one white space character in a row). 
corpus_av1_a <- tm_map(corpus_av1_a, removeWords, 
                           stopwords_without_apostrophe)
corpus_av1_a <- tm_map(corpus_av1_a, removeWords, stopwords_negation)
corpus_av1_a <- tm_map(corpus_av1_a, removeWords, extra_stopwords)
corpus_av1_a <- tm_map(corpus_av1_a, stemDocument)
corpus_av1_a <- tm_map(corpus_av1_a, removeNumbers)
corpus_av1_a <- tm_map(corpus_av1_a, stripWhitespace)

# Building Document Term Matrix, introducing sparsity threshold,
# regularizing column names and inserting the dependent variable. 
dtm_av1_a <- DocumentTermMatrix(corpus_av1_a)
sparse_av1_a <- removeSparseTerms(dtm_av1_a, 0.995)
sentSparse_av1_a <- as.data.frame(as.matrix(sparse_av1_a)) 
colnames(sentSparse_av1_a) <- make.names(colnames(sentSparse_av1_a))
sentSparse_av1_a <- sentSparse_av1_a %>% 
  mutate(sentiment = reviews_train$sentiment)

# Wordcloud
df <- sentSparse_av1_a %>% select(- ncol(sentSparse_av1_a))
wordcloud(colnames(df), colSums(df), min.freq = 10, 
          max.words = 50, random.order = FALSE, rot.per = 1/3, 
          colors = brewer.pal(8, "Dark2"), scale = c(4,.5))
rm(df)

# There is only one short form with apostrophe in the wordcloud:
# "don't", transcribed as "don.t". Will it show in the CART decision
# tree and, more generally, would including short forms better accuracy? 

# Training CART with the algorithm rpart and tuning cp with the train() function 
# from caret.
set.seed(1)
fit_cart_tuned_av1_a <- train(sentiment ~ .,
                         method = "rpart",
                         data = sentSparse_av1_a,
                         tuneLength = 15,
                         metric = "Accuracy")
fitted_cart_tuned_av1_a <- predict(fit_cart_tuned_av1_a)
cm_cart_tuned_av1_a <- confusionMatrix(as.factor(fitted_cart_tuned_av1_a), 
                        as.factor(sentSparse_av1_a$sentiment))

tab <- data.frame(cm_cart_tuned_av1_a$overall["Accuracy"]) %>%
  `colnames<-`("Model rpart with cp Tuning and Negational Short Forms") %>%
  `rownames<-`("Accuracy")
knitr::kable(tab, "pandoc")
rm(tab)

# Which is the same accuracy level as before, i.e. without the negational
# short forms. This is not surprizing since "don't", or rather "don.t", 
# does not appear in the decision tree. 

prp(fit_cart_tuned_av1_a$finalModel, uniform = TRUE, cex = 0.6, 
    box.palette = "auto")

rm(corpus_av1_a, dtm_av1_a, sparse_av1_a, sentSparse_av1_a)
rm(fit_cart_tuned_av1_a, fitted_cart_tuned_av1_a)

# Results: there is no perceptible accuracy improvement. But now, 
# all negational short forms are included and make the 
# bag of words a little bit messier. 

# Anyway, multigrams such as "don't buy" or "don't like" are going
# to be used in the second avenue when polarizing multigrams... 

# Consequently, negational short forms 
# will not be maintained in the corpus: this attempt will
# be reverted and negational short forms will be removed again. 

###########################
###########################

# Reintegrating negational unigrams such as "not".
corpus_av1_b <- VCorpus(VectorSource(reviews_train$text)) 
corpus_av1_b <- tm_map(corpus_av1_b, content_transformer(tolower))

# Replacing all punctuation marks with white space characters,
# instead of just removing punctuation marks, 
# to prevent tokens like "brokeni" from being generated.
# Keeping apostrophes to leave intact short forms such as "don't"
# so that they can be removed by using contractions 
# from stopwords_with_apostrophe.csv. 
for (i in 1:nrow(reviews_train)) {
  corpus_av1_b[[i]]$content <- gsub("(?!')[[:punct:]]", " ", 
                                    corpus_av1_b[[i]]$content, perl = TRUE)
}
rm(i)

# Removing stopwords with apostrophe (so-called short forms or contractions)
corpus_av1_b <- tm_map(corpus_av1_b, removeWords, stopwords_with_apostrophe)

# Removing remaining apostrophes (there can be apostrophes outside of short forms). 
for (i in 1:nrow(reviews_train)) {
  corpus_av1_b[[i]]$content <- gsub("[[:punct:]]", " ", 
                                    corpus_av1_b[[i]]$content)
}
rm(i)

# Removing stopwords without apostrophe, extra stopwords, stemming, 
# removing numbers, digits and multiple white space characters (leaving only
# one white space character at a time).
corpus_av1_b <- tm_map(corpus_av1_b, removeWords, 
                         stopwords_without_apostrophe)
corpus_av1_b <- tm_map(corpus_av1_b, removeWords, extra_stopwords)
corpus_av1_b <- tm_map(corpus_av1_b, stemDocument)
corpus_av1_b <- tm_map(corpus_av1_b, removeNumbers)
corpus_av1_b <- tm_map(corpus_av1_b, stripWhitespace)

# Building bag of words, managing sparsity threshold, 
# regularizing column names and adding dependent variable.
dtm_av1_b <- DocumentTermMatrix(corpus_av1_b)
sparse_av1_b <- removeSparseTerms(dtm_av1_b, 0.995)
sentSparse_av1_b <- as.data.frame(as.matrix(sparse_av1_b)) 
colnames(sentSparse_av1_b) <- make.names(colnames(sentSparse_av1_b))
sentSparse_av1_b <- sentSparse_av1_b %>% 
  mutate(sentiment = reviews_train$sentiment)

# Training CART with the algorithm rpart with cp tuning.
set.seed(1)
fit_cart_tuned_av1_b <- train(sentiment ~ .,
                       method = "rpart",
                       data = sentSparse_av1_b,
                       tuneLength = 15,
                       metric = "Accuracy")
fitted_cart_tuned_av1_b <- predict(fit_cart_tuned_av1_b)
cm_cart_tuned_av1_b <- confusionMatrix(as.factor(fitted_cart_tuned_av1_b), 
                      as.factor(sentSparse_av1_b$sentiment))

tab <- data.frame(cm_cart_tuned_av1_b$overall["Accuracy"]) %>%
  `colnames<-`("Model rpart with cp Tuning and Negational Short Forms") %>%
  `rownames<-`("Accuracy")
knitr::kable(tab, "pandoc")
rm(tab)

# Accuracy has been upgraded from around 79 % 
# to approximately 81 %. Let's build up a comparative result table with
# outcomes from the various CART models already trained. 

# Result table
# The vector "colname" already exists and will be used again. 
models <- c("cart_tuned_av1_a", "cart_tuned_av1_b")
description <- c("negational short forms + rpart + cp tuning", 
                  "negational unigrams + rpart + cp tuning")
cm <- c("cm_cart_tuned_av1_a", "cm_cart_tuned_av1_b")
tab <- data.frame(matrix(1:(length(colname) * length(models)),
                         ncol = length(colname), nrow = length(models)) * 1)

for (i in 1:2) {
  tab[i, 1] <- models[i]
  tab[i, 2] <- description[i]
  tab[i, 3] <- eval(parse(text = paste(cm[i], "$overall['Accuracy']", sep = "")))
  tab[i, 4] <- eval(parse(text = paste(cm[i], "$byClass['Sensitivity']", sep = "")))
  tab[i, 5] <- eval(parse(text = paste(cm[i], "$byClass['Neg Pred Value']", sep = "")))
  tab[i, 6] <- eval(parse(text = paste(cm[i], "$byClass['Specificity']", sep = "")))
  tab[i, 7] <- eval(parse(text = paste(cm[i], "$byClass['Pos Pred Value']", sep = "")))
}                 

tab_av1 <- tab %>% mutate_at(vars(3:7), funs(round(., 4))) %>%
  `colnames<-`(colname)

# tab_av0 already exists and will be recalled and renamed to be on the safe side. 
tab_av0 <- tab_av0 %>% `colnames<-`(colname)

tab_av_0_1 <- rbind(tab_av0, tab_av1)
knitr::kable(tab_av_0_1, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(c(1, 4), bold = T, strikeout = T, 
           color = "#9bc4e2", background = "blue") %>%
  row_spec(2:3, bold = T, color = "white", background = "blue") %>%
  row_spec(5, bold = T, color = "white", background = "green")

# In the table above, on rows 1 and 4, fonts have been blurred 
# into light blue and have been stricken through to indicate 
# that these models have been discarded.
# The other three models should be seen as a cumulative process 
# bringing accuracy improvement in a stepwise and incremental way.

rm(cm_cart_tuned_av1_a, cm_cart_tuned_av1_b)
rm(models, description)
rm(tab_av0, tab_av1)

# Reintegrating negational unigrams has boosted accuracy from 78.59 %
# to 80.84 %. Reintegrating negational short forms has changed 
# nothing perceptible in accuracy.

# Consequently, negational unigrams will be taken on board and reintegrated,
# negational short forms won't because they don't improve accuracy
# and keeping them makes things a little bit messier. 

# Let's nevertheless express a caveat: sensitivity,
# which was already relatively low in comparison with specificity,
# has further sunk with negational unigrams being incorporated. 
# False negatives must be relatively numerous. Let's have 
# a look at the confusion matrix.

tab <- table(fitted_cart_tuned_av1_b, sentSparse_av1_b$sentiment) %>% as.vector()
tab <- data.frame(matrix(tab, ncol = 2, nrow = 2, byrow = FALSE)) %>%
  `colnames<-`(c("Appreciating (ref)", "Criticizing (ref)")) %>%
  `rownames<-`(c("Appreciating (pred)", "Criticizing (pred)"))
knitr::kable(tab, "pandoc")

rm(corpus_av1_b, dtm_av1_b, sparse_av1_b, sentSparse_av1_b)
rm(fit_cart_tuned_av1_b, fitted_cart_tuned_av1_b)

# The number of false negatives is obviously the weak point. 
# This is both a challenge and an opportunity:
# false negatives need perusing. That will get priority in the next text mining
# section. 

######################
######################

# Avenue 2 - Text classification and polarization

# In the final CART decision tree, there is a majority of 
# tokens conveying some kind of subjective information on their own
# and moreover they occupy the top nodes.
# But, as has been seen while analysing false negatives
# and false positives, many other tokens conveying on their own
# some type of subjective information have not been taken
# on board in the final decision tree. 

# After comparative analysis of false negatives/positives and decision trees,
# text mining can now evolve towards text classificiation.
# Tokens that figure in false negatives/positives, that convey subjective information
# and that do not show in decision trees will be classified in either 
# positive sentiment orientation or negative sentiment orientation. 
# Accordingly, they will be stopped, if they are positively oriented, into
# - subj_pos_unigrams.csv or
# - subj_pos_multigrams.csv;
# and if they are negatively oriented, they will be stopped into
# - subj_neg_unigrams.csv or
# - subj_neg_multigrams.csv.

# These files can be found in the GitHub repository 
# https://github.com/Dev-P-L/Sentiment-Analysis.

# Here are a few examples from each file of polarized n-grams.

# Posive sentiment oriented unigrams from subj_pos_unigrams.csv (stemmed):
# "super", "awesom", etc. 

# Some positive multigrams from the file sub_pos_multigrams.csv (not stemmed):
# "no trouble", "5 stars", "thumbs up", "it's a ten", "as described",
# "know what they're doing". 

# Possible variants have usually been added, including variants originating
# from spelling errors or "alternative" grammar: 
# "no troubles", "not any trouble", "not any troubles", "no problem", 
# "no problems", etc.; "five stars", "five star", "5-star", "5star", 5 star";
# "must have"; "it's a 10", "it's a ten", "its a 10", etc.; 
# "know what theyre doing", "know what they are doing", etc.

# Some negative unigrams (after stemming) from the file 
# subj_neg_unigrams.csv: "horribl", "crap", "whine", etc.

# Some negative multigrams (not stemmed) from the file sub_neg_multigrams.csv: 
# "1 star", "one star", "not good", "no good", 
# "shouldn't" (often associated with negative context),
# "pretty piece of junk", etc.

# This polarization will then be transfered to NLP-transformed reviews:
# in reviews, all occurences of the positively
# polarized n-grams will be replaced with " subjpo ";
# analogously negatively polarized n-grams will be replaced with " subjneg ". 

# Some efficacy-minded rules will be applied. 

# First, the polarized n-grams will be preceded and followed by 
# one white space character when looking for occurences in reviews.
# Otherwise, in the bag of words, the n-gram "most inconvi" would become
# "most in subjpo " (because "convi" is a polarized unigram 
# in subj_pos_unigrams.csv) and then "  subjpo " (because "most" and "in" are
# stopwords in stopwords_without_apostrophe.csv)! A negatively 
# oriented multigram would become a positively oriented unigram! 
# Consequently, one white space character is added in front of and 
# at the end of each polarized n-gram before looking for matching occurences in 
# NLP-transformed reviews, in order to avoid replacing substrings.

# Second, as a consequence, a white space character has to be added at the beginning
# and at the end of each NLP-transformed review! Otherwise, polarized n-grams, which 
# are preceded and followed by one white space character can never 
# match an occurence that is positioned at the beginning or at the end of a review. 

# Third, as already indicated, " subpos " and "subjneg " contain 
# one white space character at the beginning and at the end, in order to prevent 
# amalgamation. Indeed, what would happen if white space chatacters were not added? 
# Let's take our well known example of " conveni ": if it were replaced with 
# just "subjpo" in the n-gram " most conveni ", then it would produce 
# " mostsubjpo", which would no longer be a generic positive unigram!
# Transformation would be useless if not annoyingly counterproductive! 

# Fourth, multiple interword white space characters have to be reduced
# to a single interword white space character: indeed two multigrams 
# differing only in the number(s) of interword white space characters
# are treated as different multigrams. 

# Fifth, in reviews, matching negative n-grams have got to be replaced before 
# matching positive ones. Let's take the example of " poor fit ",
# which is a negatively polarized multigram in the file subj_neg_multigrams.csv: 
# if matching with occurences in reviews 
# begins with positively polarized n-grams, then " poor fit "
# becomes " poor subjpo ", with " poor " possibly not counterbalancing 
# " subpos " if it is not in a node of the decision tree 
# (or if it is at too low a level).

# Sixth, among positive or negative polarized multigrams, they should 
# be tentatively matched in decreasing order in for loops.
# Why? Let's take an example of " no good bargain " in one review. 
# In sub_neg_multigrams.csv, there are two negatively polarized multigrams:
# " no good bargain " and " no good"; if these are considered in 
# decreasing order, then, in the review, " no good bargain " is 
# replaced with " subjneg ", which looks good; otherwise 
# "no good bargain " is replaced with " subjneg bargain " and
# then " subjneg subjpo ": consequently, instead of having one negative generic 
# unigram we would get one positive and one negative generic unigrams! 

############################
############################

# Further step in NLP

# Downloading the four files with positive and negative polarized n-grams. 
# Reordering multigrams in decreasing order.
# Utilizing the utf8 package to normalize punctuation, in particular
# to convert possible curly apostrophes into straight ones (problem encountered).
# Adding one space at the beginning and at the end of these 
# polarized n-grams. 

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_pos_multigrams.csv"
subj_pos_multigrams <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_pos_multigrams <- sort(subj_pos_multigrams[, 2], decreasing = TRUE) %>% as.vector()
# Converting curly apostrophes to straight apostrophes. 
subj_pos_multigrams <- sapply(subj_pos_multigrams, utf8_normalize, map_quote = TRUE)
subj_pos_multigrams <- paste("", subj_pos_multigrams, "")

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_pos_unigrams.csv"
subj_pos_unigrams <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_pos_unigrams <- subj_pos_unigrams[, 2] %>% as.vector()
subj_pos_unigrams <- paste("", subj_pos_unigrams, "")

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_neg_multigrams.csv"
subj_neg_multigrams <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_neg_multigrams <- sort(subj_neg_multigrams[, 2], decreasing = TRUE) %>% as.vector()
# Converting curly apostrophes to straight apostrophes. 
subj_neg_multigrams <- sapply(subj_neg_multigrams, utf8_normalize, map_quote = TRUE)
subj_neg_multigrams <- paste("", subj_neg_multigrams, "")

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_neg_unigrams.csv"
subj_neg_unigrams <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_neg_unigrams <- subj_neg_unigrams[, 2] %>% as.vector()
subj_neg_unigrams <- paste("", subj_neg_unigrams, "")

rm(myfile)

# Creating and lowercasing corpus.
corpus_av2 <- VCorpus(VectorSource(reviews_train$text)) 
corpus_av2 <- tm_map(corpus_av2, content_transformer(tolower))

# Replacing all punctuation marks by spaces except for apostrophes and hyphens. 
for (i in 1:nrow(reviews_train)) {
  corpus_av2[[i]]$content <- 
    gsub("[.?!]", " ", gsub("(?![-.?!'])[[:punct:]]", " ", 
                            corpus_av2[[i]]$content, perl=T))
}

# Removing spaces at the beginning and at the end of reviews
# to get apostrophes in first or last position if they are at 
# the beginning or at the end of a review.
for (i in 1:nrow(reviews_train)) {
  corpus_av2[[i]]$content <- trimws(corpus_av2[[i]]$content, which = "l")
  corpus_av2[[i]]$content <- trimws(corpus_av2[[i]]$content, which = "r")
}

# Removing apostrophes and hyphens at the beginning and at the end of reviews,
# with repetition (in case there are several of them).
for (i in 1:nrow(reviews_train)) {
  for (j in 1:12) {
    corpus_av2[[i]]$content <- sub("^[[:punct:]]","", corpus_av2[[i]]$content)
    corpus_av2[[i]]$content <- sub("[[:punct:]]$","", corpus_av2[[i]]$content)
  }
}

# Adding one space at the beginning and at the end of reviews.
for (i in 1:nrow(reviews_train)) {
  corpus_av2[[i]]$content <- paste("", corpus_av2[[i]]$content, "") 
}

# Reducing interword white space to one single character. 
corpus_av2 <- tm_map(corpus_av2, stripWhitespace)

# Matching multigrams from reviews with polarized multigrams 
# from subj_neg_multigrams.csv or subj_pos_multigrams.csv.
# If matching works, replacing multigrams from reviews 
# with generic polarized unigram " subjpo " or " subneg ".
for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_neg_multigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_neg_multigrams[j], " subjneg ", 
                                    corpus_av2[[i]]$content)
  }
}

for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_pos_multigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_pos_multigrams[j], " subjpo ", 
                                    corpus_av2[[i]]$content)
  }
}

# Removing stopwords with apostrophe.
corpus_av2 <- tm_map(corpus_av2, removeWords, stopwords_with_apostrophe)

# Replacing each remaining apostrophe or hyphen with one single white space 
# character. 
for (i in 1:nrow(reviews_train)) {
  corpus_av2[[i]]$content <- gsub("[[:punct:]]", " ", corpus_av2[[i]]$content)
}

# Polarizing multigrams again (some apostrophes or hyphens 
# might have prevented taking some n-grams into account).
for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_neg_multigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_neg_multigrams[j], " subjneg ", 
                                    corpus_av2[[i]]$content)
  }
}

for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_pos_multigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_pos_multigrams[j], " subjpo ", 
                                    corpus_av2[[i]]$content)
  }
}

# Stemming reviews.
corpus_av2 <- tm_map(corpus_av2, stemDocument)

# Challenge: the function stemDocument suppresses spaces at the beginning 
# and at the end of each review. Consequently, one space has to be added again
# at the beginning and at the end of each review.

for (i in 1:nrow(reviews_train)) {
  corpus_av2[[i]]$content <- paste("", corpus_av2[[i]]$content, "") 
}

# Polarizing multigrams again after stemming. Some multigrams might have 
# become eligible after stemming. 
for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_neg_multigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_neg_multigrams[j], " subjneg ", 
                                    corpus_av2[[i]]$content)
  }
}

for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_pos_multigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_pos_multigrams[j], " subjpo ", 
                                    corpus_av2[[i]]$content)
  }
}

# Matching polarized unigrams with unigrams in reviews and,
# if it is the case, replacing matching unigrams from reviews 
# with a generic polarized unigram. 
for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_neg_unigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_neg_unigrams[j], " subjneg ", 
                                    corpus_av2[[i]]$content)
  }
}

for (i in 1:nrow(reviews_train)) {
  for (j in 1:length(subj_pos_unigrams)) {
    corpus_av2[[i]]$content <- gsub(subj_pos_unigrams[j], " subjpo ", 
                                    corpus_av2[[i]]$content)
  }
}

# Removing remaining stopwords.
corpus_av2 <- tm_map(corpus_av2, removeWords, extra_stopwords)
corpus_av2 <- tm_map(corpus_av2, removeWords, stopwords_without_apostrophe)

# Removing numbers, digits and extra white space characters.
corpus_av2 <- tm_map(corpus_av2, removeNumbers)
corpus_av2 <- tm_map(corpus_av2, stripWhitespace)

# Creating document term matrix. 
dtm_av2 <- DocumentTermMatrix(corpus_av2)

# Removing sparse terms. 
sparse_av2 <- removeSparseTerms(dtm_av2, 0.995)

# Converting sparse, which is a Document Term Matrix, to a matrix 
# and then to a data frame.
sentSparse_av2 <- as.data.frame(as.matrix(sparse_av2)) 
rownames(sentSparse_av2) <- 1:nrow(sentSparse_av2)

# Making all variable names R-friendly.
colnames(sentSparse_av2) <- make.names(colnames(sentSparse_av2))

# Adding dependent variable.
sentSparse_av2 <- sentSparse_av2 %>% mutate(sentiment = reviews_train$sentiment)

# Building a CART model.
set.seed(1)
fit_cart_tuned_av2 <- train(sentiment ~ .,
                      method = "rpart",
                      data = sentSparse_av2,
                      tuneLength = 15, 
                      metric = "Accuracy")
fitted_cart_tuned_av2 <- predict(fit_cart_tuned_av2)
cm_cart_tuned_av2 <- 
  confusionMatrix(as.factor(fitted_cart_tuned_av2), 
                  as.factor(sentSparse_av2$sentiment))

# Result table
# The vector "colname" has been previously defined and is available. 
models <- c("cart_tuned_av2")
description <- c("negational unigrams + polarization + rpart + cp tuning")
accuracies <- cm_cart_tuned_av2$overall["Accuracy"]
sensitivities <- cm_cart_tuned_av2$byClass["Sensitivity"]
specificities <- cm_cart_tuned_av2$byClass["Specificity"] 
posPredValues <- cm_cart_tuned_av2$byClass["Pos Pred Value"]  
negPredValues <- cm_cart_tuned_av2$byClass["Neg Pred Value"]

tab_av2 <- data.frame(models, 
                      description, 
                      round(accuracies, 4),
                      round(sensitivities, 4),
                      round(negPredValues, 4),
                      round(specificities, 4),
                      round(posPredValues, 4)) %>%
           `colnames<-`(colname) %>% `rownames<-`(NULL)

tab_av_0_1 <- tab_av_0_1 %>% `colnames<-`(colname)

tab_av_0_1_2 <- rbind(tab_av_0_1, tab_av2)
knitr::kable(tab_av_0_1_2, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(c(1, 4),  bold = T, strikeout = T, 
           color = "#9bc4e2", background = "blue") %>%
  row_spec(c(2:3, 5), bold = T, color = "white", background = "blue") %>%
  row_spec(6, bold = T, color = "white", background = "green")

# In the table above, on rows 1 and 4, fonts have been blurred 
# into light blue and have been stricken through to indicate 
# that these models have been discarded.
# The other four models should be seen as a cumulative process 
# bringing accuracy improvement in a stepwise and incremental way. 

# As shown in the last two rows, thanks to polarization, 
# accuracy has jumped from 80 % up to 92 %, which is impressive. 
# More impressive: sensitivity has sprung from 66 % to 89 %. 

# False negatives have been a recurrent weak point in machine learning results
# up to now. But special attention has been paid to them in debriefing
# previous machine learning results and in text classification of 
# subjective information n-grams figuring in false negatives 
# and not in decision trees, i.e. in classifying positively oriented n-grams
# as " subjpo ". Some attention has also been paid to false positives, 
# less numerous though. 

# Let's have a look at remaining false negatives and positives
# in a confusion matrix.
tab <- table(fitted_cart_tuned_av2, sentSparse_av2$sentiment) %>% as.vector()
tab <- data.frame(matrix(tab, ncol = 2, nrow = 2, byrow = FALSE)) %>%
  `colnames<-`(c("Appreciating (ref)", "Criticizing (ref)")) %>%
  `rownames<-`(c("Appreciating (predicted AFTER POLARIZING)", 
                 "Criticizing (predicted AFTER POLARIZING)"))
knitr::kable(tab, "pandoc")

# At the previous step, results were less good, especially for false negatives,
# as already shown in a table. The number of false negatives has been crushed from 114 to 38; parallelwise,
# the number of true negatives has climbed from 220 to 296. The decrease
# in false positives is much less impressive, from 22 to 17. 

# This is linked to completely different bags of words.
# Let's have a look at the new wordcloud obtained after polarizing.
df <- sentSparse_av2 %>% select(- ncol(.))
set.seed(1)
wordcloud(colnames(df), colSums(df), min.freq = 10, 
          max.words = 50, random.order = FALSE, rot.per = 1/3, 
          colors = brewer.pal(8, "Dark2"), scale = c(4,.5))

# The n-grams with the highest frequencies are now "subjpo" and
# "subjneg" and by far. This can also visualized in another way in
# the next graph.
freq <- data.frame(to = colnames(df), fre = as.integer(colSums(df)), 
                   stringsAsFactors = FALSE) %>% 
        arrange(desc(fre)) %>% head(., 12)
graph <-  freq %>% mutate(to = reorder(to, fre)) %>%
  ggplot(aes(to, fre)) + 
  geom_bar(stat = "identity", width = 0.80, color = "#007ba7", fill = "#9bc4e2") + 
  coord_flip() +
  ggtitle("Token Frequency") +
  xlab("Token") + ylab("Frequency") +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title.x = element_text(size = 16), 
        axis.title.y = element_text(size = 16), 
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12), 
        axis.text.y = element_text(size = 12))
graph
rm(graph)

# What did that change in machine learning? Let's have a look at 
# the decision tree.  
prp(fit_cart_tuned_av2$finalModel, uniform = TRUE, cex = 0.6, 
    box.palette = "auto")

rm(corpus_av2, dtm_av2, sparse_av2)
rm(fit_cart_tuned_av2, fitted_cart_tuned_av2, cm_cart_tuned_av2)
rm(df, freq, i, j)
rm(models, description, accuracies, sensitivities, negPredValues,
   specificities, posPredValues, colname)
rm(tab, tab_av_0_1, tab_av2, tab_av_0_1_2)

# Predominance of "subjpos" and "subneg" is crystal clear.
# The decision tree is simpler. The two generic tokens, 
# with polarized sentiment orientation, are in the top nodes of the tree. 
# "not" is also highly ranked.
# Many false negatives have disappeared but many individual tokens 
# that were previously in nodes of the tree, have 
# disappeared, which has cause some new false negatives and positives. 
# For instance the tokens "happi", "better". For brevity, this is not shown.

# New improvements could be reached by ranking "happi", "better", etc. 
# in the positive cluster. The same holds for some tokens with negative sentiment
# orientation. Whole lists coming e.g. from dictionaries could be imported.

# But the experiment is going to be stopped here. 
# Anyway, there can be other tokens in the validation set, possibly partially
# different from the tokens from the training reviews. 

############################################################
############################################################

# MACHINE LEARNING 
# I - TESTING 

# Several other models are going to be applied.

# Because of high overfitting risk with some models, 
# coparing models on accuracy performance will probably
# require more tolls than just accuracy on training set. 

# Before, the corpus, and consequently the training set, 
# was built only on the training rows (ind_train).
# It was to be on the safe side: frequencies from the validation rows
# could not impact the selection of tokens when applying the 
# sparsity threshold. Insulation prevented cross-effects, probably very 
# marginal though.

# Here, we have to work on the whole data set reviews to have the same
# number of columns in the training set and in the validation set. 
# To remain on the safe side, we'll measure the possible impact
# of these cross-effects by comparing results from CART already obtained
# and new results from CART.

# Creating and lowercasing corpus.
corpus <- VCorpus(VectorSource(reviews$text)) 
corpus <- tm_map(corpus, content_transformer(tolower))

# Replacing all punctuation marks by spaces except for apostrophes and hyphens. 
for (i in 1:nrow(reviews)) {
  corpus[[i]]$content <- 
    gsub("[.?!]", " ", gsub("(?![-.?!'])[[:punct:]]", " ", 
                            corpus[[i]]$content, perl=T))
}

# Removing spaces at the beginning and at the end of reviews.
for (i in 1:nrow(reviews)) {
  corpus[[i]]$content <- trimws(corpus[[i]]$content, which = "l")
}

for (i in 1:nrow(reviews)) {
  corpus[[i]]$content <- trimws(corpus[[i]]$content, which = "r")
}

# Removing apostrophes and hyphens at the beginning 
# and at the end of reviews, with repetition. 
for (i in 1:nrow(reviews)) {
  for (j in 1:12) {
    corpus[[i]]$content <- sub("^[[:punct:]]","", corpus[[i]]$content)
    corpus[[i]]$content <- sub("[[:punct:]]$","", corpus[[i]]$content)
  }
}

# Adding one space at the beginning and at the end of reviews.
for (i in 1:nrow(reviews)) {
  corpus[[i]]$content <- paste("", corpus[[i]]$content, "") 
}

# Reducing multispaces to unispaces: otherwise, a multigram 
# with multispaces between some words can't match a multigram
# neither from subj_pos_multigrams.csv nor subj_neg_multigrams.csv. 
corpus <- tm_map(corpus, stripWhitespace)

# Polarizing negative and positive multigrams.
for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_neg_multigrams)) {
    corpus[[i]]$content <- gsub(subj_neg_multigrams[j], " subjneg ", 
                                corpus[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_pos_multigrams)) {
    corpus[[i]]$content <- gsub(subj_pos_multigrams[j], " subjpo ", 
                                corpus[[i]]$content)
  }
}

# Removing stopwords with apostrophe.
corpus <- tm_map(corpus, removeWords, stopwords_with_apostrophe)

# Replacing all apostrophes and hyphens with spaces. 
for (i in 1:nrow(reviews)) {
  corpus[[i]]$content <- gsub("[[:punct:]]", " ", corpus[[i]]$content)
}

# Polarizing multigrams again 
# (hyphens might have prevented taking some strings into account).
for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_neg_multigrams)) {
    corpus[[i]]$content <- gsub(subj_neg_multigrams[j], " subjneg ", 
                                corpus[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_pos_multigrams)) {
    corpus[[i]]$content <- gsub(subj_pos_multigrams[j], " subjpo ", 
                                corpus[[i]]$content)
  }
}

# Stemming reviews.
corpus <- tm_map(corpus, stemDocument)

# Challenge: the function stemDocument suppresses spaces at the beginning 
# and at the end of each review. Consequently, one space has to be added again
# at the beginning and at the end of each review.

for (i in 1:nrow(reviews)) {
  corpus[[i]]$content <- paste("", corpus[[i]]$content, "") 
}

# Polarizing multigrams again after stemming. Some multigrams migh have 
# become eligible after stemming. 
for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_neg_multigrams)) {
    corpus[[i]]$content <- gsub(subj_neg_multigrams[j], " subjneg ", 
                                corpus[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_pos_multigrams)) {
    corpus[[i]]$content <- gsub(subj_pos_multigrams[j], " subjpo ", 
                                corpus[[i]]$content)
  }
}

# Polarizing unigrams.
for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_neg_unigrams)) {
    corpus[[i]]$content <- gsub(subj_neg_unigrams[j], " subjneg ", 
                                corpus[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_pos_unigrams)) {
    corpus[[i]]$content <- gsub(subj_pos_unigrams[j], " subjpo ", 
                                corpus[[i]]$content)
  }
}

# Removing stopwords.
corpus <- tm_map(corpus, removeWords, extra_stopwords)
corpus <- tm_map(corpus, removeWords, stopwords_without_apostrophe)

# Removing numbers, digits and extra white space characters.
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, stripWhitespace)

# Creating document term matrix. 
dtm <- DocumentTermMatrix(corpus)

# Removing sparse terms, keeping only words that appear in 0.5 % of reviews. 
sparse <- removeSparseTerms(dtm, 0.995)

# Converting sparse, which is a Document Term Matrix to a matrix 
# and then to a data frame.
sentSparse <- as.data.frame(as.matrix(sparse)) 

# Making all variable names R-friendly.
colnames(sentSparse) <- make.names(colnames(sentSparse))

# Adding dependent variable.
sentSparse <- sentSparse %>% 
  mutate(sentiment = reviews$sentiment) %>% as.data.frame()

# Splitting into training set and validation set. 
train <- sentSparse[ind_train, ]
val <- sentSparse[ind_val, ]

# TRAINING MACHINE LEARNING MODELS ON TRAINING SET

# 10 models will be trained on the training set.
# They are trained with the train() function from the caret package. 
# By default, training is done on 25 bootstrapped resamples and
# on 3 values of the parameters tuned. The names of the parameters
# tuned by the train() function are available in 
# http://topepo.github.io/caret/available-models.html .

IDs <- c("cart", "cart_tuned", "svm", "svm_tuned", "adaboost",
            "rf", "gbm", "gbm_tuned", "xgb", "monmlp")
models <- c("CART",
           "CART",
           "Support Vector Machines with Radial Basis Function Kernel",
           "Support Vector Machines with Radial Basis Function Kernel",
           "AdaBoost Classification Trees",
           "Random Forest",
           "Stochastic Gradient Boosting",
           "Stochastic Gradient Boosting",
           "eXtreme Gradient Boosting",
           "Monotone Multi-Layer Perceptron Neural Network")
caret_names <- c("rpart", "rpart", "svmRadialCost", "svmRadialCost",
             "adaboost", "rf", "gbm", "gbm", "xgbLinear", "monmlp")
tunings <- c(3, 15, 3, 15, 3, 3, 3, 15, 3, 3)
nr_resamples <- rep(25, 10)
colname_methods <- c("MODEL ID", "METHOD", "NAME IN CARET", 
                    "# TUNING VALUES", "# BOOTSTRAPPED RESAMPLES")

tab <- data.frame(IDs, models, caret_names, tunings, nr_resamples) %>%
       `colnames<-`(colname_methods)
knitr::kable(tab, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(c(1, 3, 5:7, 9:10),  bold = T, 
           color = "blue", background = "powderblue") %>%
  row_spec(c(2, 4, 8), bold = T, color = "white", background = "blue") 

rm(tunings, nr_resamples, colname_methods, tab)

rows_standard_tuning <- c(1, 3, 5:7, 9:10)
rows_standard_tuning <- c(1, 3)
rows_extra_tuning <- setdiff(1:length(IDs), rows_standard_tuning)
IDs_standard_tuning <- IDs[rows_standard_tuning]
IDs_extra_tuning <- IDs[rows_extra_tuning]
caret_names_standard_tuning <- caret_names[rows_standard_tuning]
caret_names_extra_tuning <- caret_names[rows_extra_tuning]

fits_standard_tuning <- 
  lapply(caret_names_standard_tuning, function(meth){
    set.seed(1)
    train(sentiment ~ ., method = meth, data = train, metric = "Accuracy")
  }) 

fits_extra_tuning <- 
  lapply(caret_names_extra_tuning, function(meth){
    set.seed(1)
    train(sentiment ~ ., method = meth, data = train, tuneLength = 15, 
          metric = "Accuracy")
  })

names(fits_standard_tuning) <- IDs_standard_tuning
names(fits_extra_tuning) <- IDs_extra_tuning

fits <- list(fits_standard_tuning, fits_extra_tuning)


# CART 
# Here, rpart is trained by the train() function instead of the 
# function rpart, because resample data will be collected
# to compare models on accuracy performance. 
set.seed(1)
fit_cart <- train(sentiment ~ ., method = "rpart", data = train, metric = "Accuracy") 
fitted_cart <- predict(fit_cart)
cm_train_cart <- confusionMatrix(as.factor(fitted_cart), 
                                 as.factor(train$sentiment))

# CART + tuning
set.seed(1)
fit_cart_tuned <- train(sentiment ~ .,
                    method = "rpart",
                    data = train,
                    tuneLength = 15, 
                    metric = "Accuracy") 
fitted_cart_tuned <- predict(fit_cart_tuned)
cm_train_cart_tuned <- confusionMatrix(as.factor(fitted_cart_tuned), 
                                       as.factor(train$sentiment))

# SVM
set.seed(1)
fit_svm <- train(sentiment ~ ., method = "svmRadialCost", data = train) 
fitted_svm <- predict(fit_svm)
cm_train_svm <- confusionMatrix(as.factor(fitted_svm), 
                                as.factor(train$sentiment))

# SVM + tuning
set.seed(1)
fit_svm_tuned <- train(sentiment ~ .,
                  method = "svmRadialCost",
                  data = train,
                  tuneLength = 15, 
                  metric = "Accuracy")
fitted_svm_tuned <- predict(fit_svm_tuned)
cm_train_svm_tuned <-
  confusionMatrix(as.factor(fitted_svm_tuned), 
                  as.factor(train$sentiment))

# AdaBoost
set.seed(1)
fit_adaboost <- train(sentiment ~ ., method = "adaboost", data = train) 
fitted_adaboost <- predict(fit_adaboost)
cm_train_ada <- confusionMatrix(as.factor(fitted_adaboost), 
                                as.factor(train$sentiment))

# Random forest model
set.seed(1)
fit_rf <- train(sentiment ~ ., method = "rf", data = train) 
fitted_rf <- predict(fit_rf)
cm_train_rf <- confusionMatrix(as.factor(fitted_rf), 
                               as.factor(train$sentiment))

# GBM
set.seed(1)
fit_gbm <- train(sentiment ~ ., method = "gbm", data = train, verbose = FALSE) 
fitted_gbm <- predict(fit_gbm)
cm_train_gbm <- confusionMatrix(as.factor(fitted_gbm), 
                                as.factor(train$sentiment))

# GBM + tuning
set.seed(1)
fit_gbm_tuned <- train(sentiment ~ .,
                    method = "gbm",
                    data = train,
                    tuneLength = 15, 
                    metric = "Accuracy",
                    verbose = FALSE)
fitted_gbm_tuned <- predict(fit_gbm_tuned)
cm_train_gbm_tuned <- confusionMatrix(as.factor(fitted_gbm_tuned), 
                                      as.factor(train$sentiment))

# XGBoost
set.seed(1)
fit_xgb <- train(sentiment ~ ., method = "xgbLinear", data = train) 
fitted_xgb <- predict(fit_xgb)
cm_train_xgb <- confusionMatrix(as.factor(fitted_xgb), 
                                as.factor(train$sentiment))

# MONMLP
set.seed(1)
fit_monmlp <- train(sentiment ~ ., method = "monmlp", data = train) 
fitted_monmlp <- predict(fit_monmlp)
cm_train_mlp <- confusionMatrix(as.factor(fitted_monmlp), 
                                as.factor(train$sentiment))

# Table to compare model accuracy on the training set.

models <- c("rpart", "rpart + extra tuning", "smvRadialCost", 
            "svmRadialCost + extra tuning", "adaboost", "rf", 
            "gbm", "gbm + extra tuning", "xgbLinear", "monmlp")

accs <- c(cm_train_cart$overall["Accuracy"],
               cm_train_cart_tuned$overall["Accuracy"],
               cm_train_svm$overall["Accuracy"],
               cm_train_svm_tuned$overall["Accuracy"],
               cm_train_ada$overall["Accuracy"],
               cm_train_rf$overall["Accuracy"],
               cm_train_gbm$overall["Accuracy"],
               cm_train_gbm_tuned$overall["Accuracy"],
               cm_train_xgb$overall["Accuracy"],
               cm_train_mlp$overall["Accuracy"])

colname_train <- c("MODEL", "ACCURACY ON THE TRAINING SET")
tab_test <- data.frame(models, accs = round(accs, 4)) %>%
            arrange(- accs) %>% `colnames<-`(colname_train)

kable(tab_test, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(1:10, bold = T, color = "white", background = "blue")
  
# Not surprizingly, most models seem to overfit. As pointed out in 
# literature, it is a rather widespread tendency. Moreover,
# NLP has been extensively adjusted on the training set. 

# Consequently, the ranking figuring in the table above is no appropriate 
# tool to separate models. 

# Accuracy cannot be tested on the validation set
# since the validation set is only to be used as a last step;
# if some testing is performed on the validation set, it is 
# no longer a validation set. 

# Moreover, the whole sample being very limited 
# (1,000 reviews), it was potentially counter-productive 
# to divide it into training, test and validation sets. 

# But there are some unused resources. The caret train() function 
# has automatically trained the models on 25 bootstrapped resamples, 
# for all values of the tuned parameters, i.e. on three values of the 
# tuned parameters except where extra tuning on 15 values had been asked, 
# i.e. CART, svm and gbm (chosen because running time was limited). 
# Actually, there is tuning on all models but in three cases
# there is extra tuning. 

# Accuracy on 25 resamples is available for the 10 models 
# for the value of the paramter(s) tuned that deliver 
# on average on all the 25 resamples the highest accuracy level. 
# This produces for each model a distribution of accuracy, 
# which is depicted below for the model with the highest level for
# the resample accuracy mean.   

# Let's first extract accuracy distributions from the 10 models. 
# Let's draw graphs with the resample accuracy distribution from
# each model. 
# The list of models has already been defined under the name "models". 
distributions <- data.frame(matrix(1:(10 * 25), nrow = 25, ncol = 10) * 1) %>%
  `colnames<-`(models) 
for (i in 1:length(models)) {
  expressio <- paste("fit_", models[i], "$resample$Accuracy", sep = "")
  distributions[, i] <- eval(parse(text = expressio))
}

l <- list(1:10)
for (i in 1:length(models)) {
  graph <- distributions %>% select(i) %>% as.data.frame() %>% 
    `colnames<-`("dist") %>%
    ggplot(aes(dist)) + 
    geom_histogram(bins = 7, color = "#007ba7", fill = "#9bc4e2") + 
    geom_vline(aes(xintercept = mean(dist)), col = "magenta", size = 2) +
    geom_vline(aes(xintercept = median(dist)), col = "yellow", 
               linetype = "dashed", size = 2) +
    ggtitle(models[i]) +
    theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
          axis.title.x = element_blank(), axis.title.y = element_blank(), 
          axis.text.x = element_text(size = 12), axis.text.y =element_text(size = 12))
  l[[i]] <- graph
}

marrangeGrob(l, nrow = 5, ncol = 2, 
             top = quote(paste("RESAMPLE ACCURACY DISTRIBUTION PER MODEL")))

# On these graphs, the mean appears as a vertical magenta line 
# and the median as a vertical dashed yellow line. 

# These distributions do not seem to be really centered around the means.

# Most models have a mean and a median above 90 % except for cart and monmlp, 
# which thereby do not rank among the favorite models. 

# There is a right-skewed distribution: cart, which is anyway already out of 
# competition. 

# There are also 3 left-skewed distributions: monmlp (already out), 
# svm and svm_tuned. A long left tail is problematic from a risk management
# viewpoint since some resamples show accuracy levels that are 
# substantially lower than the average and the median, even if
# these resamples are a minority. Thus svm and svm_tuned will also be
# excluded from the validation process. This criterion will even 
# be generalized: models will be ranked on the basis of their accuracy minimum
# and the top 3 will be combined into an ensemble model. 

# To evaluate the merits of the remaining models, let's produce a 
# comparative table.

# The function resamples() will be used, which implies separating 
# the 3 models with extra tuning since the tuned values of parameters 
# is 15 instead of 3 and aggreagting them to the other ones 
# in the resamples() function would disrupt output. 
list_fit_standard <- list(cart = fit_cart, 
                          svm = fit_svm,
                          adaboost = fit_adaboost,
                          rf = fit_rf,
                          gbm = fit_gbm,
                          xgb = fit_xgb,
                          monmlp = fit_monmlp)
list_fit_extra_tuning <- list(cart_tuned = fit_cart_tuned,
                              svm_tuned = fit_svm_tuned,
                              gbm_tuned = fit_gbm_tuned)

tab_standard <- summary(resamples(list_fit_standard))
tab_extra_tuning <- summary(resamples(list_fit_extra_tuning))

tab_standard <- tab_standard$statistics$Accuracy
tab_extra_tuning <- tab_extra_tuning$statistics$Accuracy

# Let's immediately order all models on the basis of the accuracy minimum.
tab <- rbind(tab_standard, tab_extra_tuning) %>% as.data.frame() %>%
       mutate(model = rownames(.)) %>%
       select(model, everything()) %>%
       mutate_at(vars(2:7), funs(round(., 4))) %>%
       rename(Model = model) %>%
       arrange(- Min.)

kable(tab, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(1:3, bold = T, color = "white", background = "green") %>%
  row_spec(4:10, bold = T, color = "white", background = "blue")

# The 3 top models are xgb, rf and gbm_tuned. 
# For the record, cart-tuned doesn't score badly: it is rather close to 
# gbm_tuned in minimum; it had been used as a practical yardstick in
# the NLP and text mining part, running relatively quickly. 

#######################
#######################

# VALIDATION

pred_val_xgb <- predict(fit_xgb, newdata = val)
pred_val_rf <- predict(fit_rf, newdata = val)
pred_val_gbm_tuned <- predict(fit_gbm_tuned, newdata = val)

cm_val_xgb <- confusionMatrix(as.factor(pred_val_xgb), 
                              as.factor(val$sentiment))
cm_val_rf <- confusionMatrix(as.factor(pred_val_rf), 
                              as.factor(val$sentiment))
cm_val_gbm_tuned <- confusionMatrix(as.factor(pred_val_gbm_tuned), 
                              as.factor(val$sentiment))

mean(pred_val_xgb == val$sentiment)

# €nsemble model xgb / rf / gbm_tuned
df <- data.frame((as.integer(pred_val_xgb) - 2) * -1,
                 (as.integer(pred_val_rf) - 2) * -1,
                 (as.integer(pred_val_gbm_tuned) - 2) * -1)
votes <- rowSums(df)
votes <- ifelse(votes > 1, 1, 0)                 
votes <- ifelse(votes == 1, "Appreciating", "Critisizing")
mean(votes == val$sentiment)

table(pred_val_xgb, val$sentiment)
table(pred_val_rf, val$sentiment)
table(pred_val_gbm_tuned, val$sentiment)

#######################################################
#######################################################



