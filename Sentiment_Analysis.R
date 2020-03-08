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
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(wsrf)) install.packages("wsrf", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(mgcv)) install.packages("mgcv", repos = "http://cran.us.r-project.org")
if(!require(fastAdaboost)) install.packages("fastAdaboost", repos = "http://cran.us.r-project.org")
if(!require(monmlp)) install.packages("monmlp", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

# Data has been downloaded from the UCI Machine Learning Repository: 
# https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences 
# and have been uploaded onto GitHub under the name "amazon_cells_labelled.txt". 

# Now, let's retrieve amazon_cells_labelled.txt from my GitHub repository by accessing 
# https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/amazon_cells_labelled.txt.

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis-In-Progress/master/amazon_cells_labelled.txt"
reviews <- read.delim(myfile, header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE)
rm(myfile)

# Preprocessing data frame.
colnames(reviews) <- c("text", "sentiment") 
reviews <- reviews %>% mutate(sentiment = 
  as.factor(gsub("1", "Appreciating", gsub("0", "Critisizing", reviews$sentiment)))) %>%
  as.data.frame()
rownames(reviews) <- 1:nrow(reviews)
as_tibble(head(reviews))

# Running a first model to create a critical yardstick

# Creating corpus.
corpus <- VCorpus(VectorSource(reviews$text)) 

# Preprocessing corpus: converting to lower-case, 
# removing punctuation and English stopwords, stemming document
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

# Checking tokens from preprocessing.
# Building up Document Term Matrix
dtm <- DocumentTermMatrix(corpus)

freq <- findFreqTerms(dtm, lowfreq = 1)
nc <- 5
mis <- ((ceiling(length(freq) / nc)) * nc) - length(freq)
addendum <- as.character(rep("-", mis))
freq <- as.character(c(freq, addendum))
freq <- data.frame(matrix(freq, ncol = nc, byrow = TRUE))
colnames(freq) <- NULL
rownames(freq) <- NULL
freq
tail(freq, 90)

# From a formal point of view, we notice "ive" at line 112. 
# How many " ive " did we have in the data frame reviews? 
nr <- data.frame(nr = length(grep(" ive ", reviews$text)))
knitr::kable(nr, "pandoc")

# Let's find the first occurence of " I've "
first_occurence <- (grep(" I've ", reviews$text))[1]
# What's the corresponding token after preprocessing?
corpus[[first_occurence]]$content

# After tokenization, the "short form" "I've" has not
# disappeared although it is included in English stopwords 
# and English stopwords have been removed! 

# Why? The text has first been lowercased, 
# then punctuation has been removed, stopwords have been removed and 
# stemming has been operated. 

# Consequently, "I've" has become "i've" and then "ive", which is no stopword
# and has very logically not been removed along with all stopwords. 
# In order to remove "I've", the preprocessing order should be alterated:
# lowercasing, removing stopwords and only then removing punctuation and
# stemming. That is what will be done later on. 

# Rationale is similar, mutatis mutandis, for "couldnt". 
# Let's find the occurences of " couldn't ".

first_occurence <- (grep(" couldn't ", reviews$text))[1]

# What's the corresponding token after preprocessing?

corpus[[first_occurence]]$content

# Not disappeared, of course!

# Moreover, there can also be " couldnt " in reviews.
# And, of course, it is not going to disappear 
# through preprocessing since " couldnt " is no stopword! 
length(grep(" couldnt ", reviews$text))

# Consequently, short forms without apostrophe should be
# removed as well in addition to stopwords. 

# Furthermore, removing punctuation without inserting any space
# can cause strings to fuse. Several examples of it can easily
# be pinpointed. 

# " iam " at line 122. Let's check it up on it.
v <- 1:nrow(reviews)
string <- "iam"
for(i in 1:nrow(reviews)) {
  v[i] <- length(grep(string, corpus[[i]]$content))
}
reviews$text[which(v == 1)]
# Ideally, "Iam" could be corrected to be normally removed. 
# Alternatively, "Iam" can be added to words to be removed. 

# Moreover, on line 199, "rebootsoveral"! Where does it come from?
v <- 1:nrow(reviews)
string <- "rebootsoveral"
for(i in 1:nrow(reviews)) {
  v[i] <- length(grep(string, corpus[[i]]$content))
}
reviews$text[which(v == 1)] 

# Actually, it was "reboots.Overall" but the function 
# removePunctuation() removes punctuation without inserting 
# any white space! Consequently, some words get fused, just as 
# in this case. It can also be so in the case of an apostrophe
# between two words, as we have already pinpointed it. 
# Therefore, the function the preprocessing process will be rerun
# with a for loop that replaces punctuation marks with white spaces 
# instead of the removePunctuation() function. 

# To complete corpus cleaning, in case of 
# multiple white spaces in a row, all of them will be scrapped but one. 

################################
################################

# Pinpointing extra stopwords. Let's have a look at all tokens.
freq

# Scrolling through the data frame freq has provided 13 extra stopwords.
# They have been stored in the file extra_stopwords.csv, 
# which has been uploaded to my GitHub repository,
# https://github.com/Dev-P-L/Sentiment-Analysis.
# It is going to be downloaded now and integrated into
# the preprocessing process. 

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/stopwords_with_apostrophe.csv"
stopwords_with_apostrophe <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
stopwords_with_apostrophe <- stopwords_with_apostrophe[, 2] %>% as.vector()

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
corpus <- VCorpus(VectorSource(reviews$text)) 
corpus <- tm_map(corpus, content_transformer(tolower))

for (i in 1:nrow(reviews)) {
  corpus[[i]]$content <- gsub("[[:punct:]]", " ", corpus[[i]]$content)
}

corpus <- tm_map(corpus, removeWords, stopwords_with_apostrophe)
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, removeWords, stopwords_without_apostrophe)
corpus <- tm_map(corpus, removeWords, stopwords_negation)
corpus <- tm_map(corpus, removeWords, extra_stopwords)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, stripWhitespace)

# Let's have a look at the new Document Term Matrix, to check 
# whether shortcomings have disappeared or not. 
dtm <- DocumentTermMatrix(corpus)

freq <- findFreqTerms(dtm, lowfreq = 1)
nc <- 5
mis <- ((ceiling(length(freq) / nc)) * nc) - length(freq)
addendum <- as.character(rep("-", mis))
freq <- as.character(c(freq, addendum))
freq <- data.frame(matrix(freq, ncol = nc, byrow = TRUE))
colnames(freq) <- NULL
rownames(freq) <- NULL

# "couldnt" has disappeared:
knitr::kable(freq[53, ], "pandoc")

# "iam" too
knitr::kable(freq[116, ], "pandoc")

# As well as "ive"
knitr::kable(freq[125, ], "pandoc")

# And "rebootsoveral"
knitr::kable(freq[188, ], "pandoc")

# As well as many other oddities

# Other issues need adressing, among others sparsity threshold and 
# dichotomy between subjective and objective information. That dichotomy
# will be addressed after running a first machine learning model
# and garnering first results. 

# Let's immediately address sparsity. 
# Inspecting part of dtm. 
inspect(dtm[500:510, 50:60])

# This matrix is sparse: it contains many zeroes. 

# Removing sparse terms, keeping only words that appear in 0.5 % of reviews. 
sparse <- removeSparseTerms(dtm, 0.995)

# Converting sparse, which is a list, to a matrix and then to a data frame.
sentSparse <- as.data.frame(as.matrix(sparse)) 
rownames(sentSparse) <- 1:nrow(sentSparse)

# Making all variable names R-friendly.
colnames(sentSparse) <- make.names(colnames(sentSparse))

# Adding dependent variable.
sentSparse <- sentSparse %>% mutate(sentiment = reviews$sentiment)

# Splitting into training set and validation set.
set.seed(1)
ind_train <- createDataPartition(y = sentSparse$sentiment, times = 1, p = 2/3, list = FALSE)
train <- sentSparse[ind_train,]

# Creating test set from training set by bootstrapping to save observations. 
set.seed(1)
ind_test <- sample(1:nrow(train), size = nrow(train)/2, replace = TRUE)
set.seed(1)
ind_test <- sample(ind_test, size = nrow(train)/2, replace = TRUE)
test <- train[ind_test, ]

# Building a CART model.
set.seed(1)
trControl = trainControl(method = "cv", number = 10, p = .9)
fit_cart <- train(sentiment ~ .,
                  method = "rpart",
                  data = train,
                  tuneLength = 15, 
                  metric = "Accuracy",
                  trControl = trControl)

fitted_cart <- predict(fit_cart)
confusionMatrix(as.factor(fitted_cart), as.factor(train$sentiment))

# Making predictions.
pred_test <- predict(fit_cart, newdata = test)
confusionMatrix(as.factor(pred_test), as.factor(test$sentiment))

# Baseline model: predicting "Appreciating" everywhere
# On the training set
pred_train_baseline <- rep("Appreciating", nrow(train))
acc_train_baseline <- mean(pred_train_baseline == train$sentiment) 
as_tibble(acc_train_baseline)

# On the test set
pred_test_baseline <- rep("Appreciating", nrow(test))
acc_test_baseline <- mean(pred_test_baseline == test$sentiment) 
as_tibble(acc_test_baseline)

# Comment 
# Accuracy is 78 % on the training set. 

# Statement 1: it is already much better than the baseline model which would give 50%
# on the training set and on the test set, which relects prevalence.

# Statement 2: sensitivity and negative predictive value are a bit lower.
# Indeed, proportionately, there are many false negatives, i.e. predictions
# pointing to "Critisizing" while the reference value is "Appreciating". 
# Let's have a look at the false negatives from the training set. 

df <- train %>% mutate(pred = fitted_cart) %>% as.data.frame()
FP_train <- ifelse(df$sentiment == "Appreciating", 1, 0) - 
  ifelse(df$pred == "Appreciating", 1, 0)
FP_train <- ifelse(FP_train == 1, 1, 0)

sample_size <- 12
set.seed(1)
seq <- sort(sample(which(FP_train == 1), sample_size, replace = FALSE))

df <- data.frame(matrix(nrow = sample_size * 2, ncol = 2) * 1)

for (i in 1:length(seq)) {
  row <- as.numeric(rownames(train[seq[i], ]))
  df[(i*2)-1, 1] <- reviews[row, 1]
  df[i*2, 1] <- corpus[[row]]$content
}

colnames(df) <- c("REVIEW in blue, TOKENIZATION in green", "SUBJECTIVE INFORMATION")
df[, 2] <- c("seamlessly", "seamless",
             "tremendous", "tremend", 
             "fantastic + perfectly", "fantast + perfect",
             "excited + cute", "excit + cute",
             "prompt", "prompt",
             "rocks", "rock",
             "no trouble", "troubl",
             "cool", "cool",
             "perfectly", "perfect",
             "happier", "happier",
             "saved", "save",
             "wonderfully", "wonder")

kable(df, "html", align = "c") %>% 
  kable_styling(bootstrap_options = c("bordered", "condensed"), 
                full_width = F, font_size = 16) %>%
  row_spec(1:12, bold = T, color = "white", background = "green") %>%
  row_spec(13:14, bold = T, color = "white", background = "#D7261E") %>%
  row_spec(15:24, bold = T, color = "white", background = "green")

# From a contentual point of view, in 11 cases out of 12, 
# polarity is cristal clear from a human point of view:
# there is subjective information conveyed by at least one or two words:
# it can express some feeling/emotion ("tremendous", "fantastic", 
# "perfect", "magical", etc.) or a statement of practicality ("seamlessly", 
# "understanding", "patient", "saved", etc.). 
# In one case out of 12, the review is cristal clear as well 
# but the token is misleading: a negation mark ("no") has
# been dropped because it is a stopword. 
# At least two questions need asking:
# - why have these 11 pieces of subjective information not been utilized?
# - what could be done for negations? 

# First question: let's have a look at the decision tree 
# of the final model from CART. 

prp(fit_cart$finalModel, uniform = TRUE, cex = 0.6)

# Effectively, the decision tree does not split on tokens that 
# have been mentioned in the right-handed column from the table above. 
# Let's have a look at cp values (complexity parameter values) used by model.

ggplot(fit_cart)

# According to the graph above, cp = 0 is the best option to maximize accuracy. 
# It has automatically been picked up by caret in the final model, whose 
# results have already been seen. Let's check that cp value is zero in the final
# model.

fit_cart$bestTune
fit <- rpart(sentiment ~., data = train, cp = fit_cart$bestTune)
pred <- predict(fit, type = "class")
mean(pred == train$sentiment)

# If we don't specify cp, result is less performing: 
fit <- rpart(sentiment ~., data = train)
fitted <- predict(fit, type = "class")
mean(fitted == train$sentiment)
pred <- predict(fit, newdata = test, type = "class")
mean(pred == test$sentiment)

# What about excluding some tokens? What about 
# "factual" tokens (tokens designating devices, etc.)?
# Let's have a look! Let's build up a wordcloud 
# on the basis of the training set. 

temp <- sentSparse[, -ncol(sentSparse)]
temp <- temp[as.numeric(rownames(train)), ]
set.seed(1)
wordcloud(colnames(temp),colSums(temp), min.freq = 10, max.words=50,
          random.order = FALSE, rot.per = 1/3, colors = brewer.pal(8, "Dark2"), 
          scale = c(4,.5))

# In the wordcloud, lots of tokens not conveying subjective information
# are present: "phone", "headset", "product", "batteri", etc.,
# as well as numerous tokens conveying some form of subjective information. 

# Regarding the issue of negations, let's have a look at
# word associations for "not". We've got to reprocess
# the data frame reviews without removing the stopwords "not", "no" or "nor",
# which are in stopwords_negation.csv. 

# Alternate dtm
corpus_2 <- VCorpus(VectorSource(reviews$text)) 
corpus_2 <- tm_map(corpus_2, content_transformer(tolower))

for (i in 1:nrow(reviews)) {
  corpus_2[[i]]$content <- gsub("[[:punct:]]", " ", corpus_2[[i]]$content)
}

corpus_2 <- tm_map(corpus_2, removeWords, stopwords_with_apostrophe)
corpus_2 <- tm_map(corpus_2, stemDocument)
corpus_2 <- tm_map(corpus_2, removeWords, stopwords_without_apostrophe)
corpus_2 <- tm_map(corpus_2, removeWords, extra_stopwords)
corpus_2 <- tm_map(corpus_2, removeNumbers)
corpus_2 <- tm_map(corpus_2, stripWhitespace)

dtm_2 <- DocumentTermMatrix(corpus_2)
findAssocs(dtm_2, "not", 0.1)
knitr::kable(findAssocs(dtm_2, "not", 0.1), "pandoc")

# Associated 
# - with positive words: "work", "impressed", "match", "advise", "operate", etc.
# - and with negative words: "cracked", "snug", "bother", "bottom".
# This avenue of research will not be investigated in this project.

# When globalising the previous analysis of false negatives, 
# the decision tree above and the wordcloud above, 
# several statements can be made:
# - statement 1: in the decision tree, there is a majority of tokens conveying 
# some kind of subjective information such as "great", "good" and "excel"
# but also a minority of factual tokens such as "cell" and "headset";
# - statement 2: in the wordcloud, some high frequncy tokens are present
# but they have not been taken over to the decision tree, such as 
# "phone", "batteri", "ear", "call", "bluetooth", etc.;
# - statement 3: in the analysis of false negatives, there are 
# some subjective information tokens that have not been retained
# in the decision tree, such as "tremend", "fantast", "incred",
# "favorit", "magic", "cool", "happier" and "wonder", 
# which could contribute to explain why these are false negatives;
# - statement 4: in the analysis of false negatives again, 
# it appeared that the negation "not" has been removed 
# as being a stopword; the same holds for "no", "neither" and 
# "nor".

# These statements open up three avenues for text mining:
# - avenue 1: numerous factual tokens that are not present 
# in the decision tree will be removed from the corpus and 
# the whole process will be rerun up to and including 
# the CART model; then, numbers and digits
# will be removed; 
# - avenue 2: numerous subjective information tokens that
# have not been taken on board in the decision tree will be 
# aggregated into two generic tokens: subjpos and subjneg 
# and the whole process will be rerun; in the same line,
# two synonyms ("cellular" and "cellphone") are going 
# to be changed into "cell", token that is used to split
# the decision tree; 
# - avenue 3: the negations "not", "no", "neither" and "nor"
# will be removed from stopwords and will remain 
# in the corpus;
# - avenue 4: previous avenues that deliver accuracy improvement
# will be combined and the whole process will be rerun. 

# Avenue 1: removing factual tokens (including numbers and digits).

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/factual.csv"
factual <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
factual <- sort(factual[, 2], decreasing = TRUE) %>% as.vector()
factual <- paste("", factual, "") 
rm(myfile)

# Creating and preprocessing corpus.
corpus_av1 <- VCorpus(VectorSource(reviews$text)) 
corpus_av1 <- tm_map(corpus_av1, content_transformer(tolower))

for (i in 1:nrow(reviews)) {
  corpus_av1[[i]]$content <- gsub("[[:punct:]]", " ", corpus_av1[[i]]$content)
}

corpus_av1 <- tm_map(corpus_av1, removeWords, stopwords_with_apostrophe)
corpus_av1 <- tm_map(corpus_av1, stemDocument)
corpus_av1 <- tm_map(corpus_av1, removeWords, stopwords_without_apostrophe)
corpus_av1 <- tm_map(corpus_av1, removeWords, stopwords_negation)
corpus_av1 <- tm_map(corpus_av1, removeWords, extra_stopwords)

for (i in 1:nrow(reviews)) {
  for (j in 1:length(factual)) {
    corpus_av1[[i]]$content <- gsub(factual[j], " ", corpus_av1[[i]]$content)
  }
}

corpus_av1 <- tm_map(corpus_av1, removeNumbers)
corpus_av1 <- tm_map(corpus_av1, stripWhitespace)

dtm_av1 <- DocumentTermMatrix(corpus_av1)

# Removing sparse terms, keeping only words that appear in 0.5 % of reviews. 
sparse_av1 <- removeSparseTerms(dtm_av1, 0.995)

# Converting sparse, which is a list, to a matrix and then to a data frame.
sentSparse_av1 <- as.data.frame(as.matrix(sparse_av1)) 
rownames(sentSparse_av1) <- 1:nrow(sentSparse_av1)

# Making all variable names R-friendly.
colnames(sentSparse_av1) <- make.names(colnames(sentSparse_av1))

# Adding dependent variable.
sentSparse_av1 <- sentSparse_av1 %>% mutate(sentiment = reviews$sentiment)

# Splitting into training set and validation set.
set.seed(1)
ind <- createDataPartition(y = sentSparse_av1$sentiment, 
                           times = 1, p = 1/3, list = FALSE)
test_av1 <- sentSparse_av1[ind,]
train_av1 <- sentSparse_av1[-ind,]

# Creating test set from training set by bootstrapping to save observations. 
set.seed(1)
ind <- sample(rownames(train), size = nrow(train)/2, replace = TRUE)
set.seed(1)
ind <- sample(ind, size = nrow(train)/2, replace = TRUE)
test <- train[ind, ]

# Building a CART model.
set.seed(1)
trControl = trainControl(method = "cv", number = 10, p = .9)
fit_cart_av1 <- train(sentiment ~ .,
                  method = "rpart",
                  data = train_av1,
                  tuneLength = 15, 
                  metric = "Accuracy",
                  trControl = trControl)

fitted_cart_av1 <- predict(fit_cart_av1)
confusionMatrix(as.factor(fitted_cart_av1), as.factor(train_av1$sentiment))

# Making predictions.
pred_test_av1 <- predict(fit_cart_av1, newdata = test_av1)
confusionMatrix(as.factor(pred_test_av1), as.factor(test_av1$sentiment))

# Some accuracy has been lost.

# Avenue 2
df <- sentSparse %>% select(- sentiment)
df <- data.frame(token = colnames(df), frequency = colSums(df)) %>%
  arrange(- frequency)
nrow(df)

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_pos_words.csv"
subj_pos_words <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_pos_words <- sort(subj_pos_words[, 2], decreasing = TRUE) %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_pos_tokens.csv"
subj_pos_tokens <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_pos_tokens <- sort(subj_pos_tokens[, 2], decreasing = TRUE) %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_neg_words.csv"
subj_neg_words <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_neg_words <- sort(subj_neg_words[, 2], decreasing = TRUE) %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_neg_tokens.csv"
subj_neg_tokens <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_neg_tokens <- sort(subj_neg_tokens[, 2], decreasing = TRUE) %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/synonyms.csv"
synonyms <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
synonyms <- sort(synonyms[, 2], decreasing = TRUE) %>% as.vector()
rm(myfile)

# Creating and preprocessing corpus.
corpus_av2 <- VCorpus(VectorSource(reviews$text)) 
corpus_av2 <- tm_map(corpus_av2, content_transformer(tolower))
corpus_av2 <- tm_map(corpus_av2, removeWords, stopwords("english"))

for (i in 1:nrow(reviews)) {
  corpus_av2[[i]]$content <- gsub("[[:punct:]]", " ", corpus_av2[[i]]$content)
}

corpus_av2 <- tm_map(corpus_av2, stemDocument)
corpus_av2 <- tm_map(corpus_av2, removeWords, stopwords("english"))

for (i in 1:nrow(reviews)) {
  for (j in 1:length(extra_stopwords)) {
    corpus_av2[[i]]$content <- gsub(extra_stopwords[j], " ", corpus_av2[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_pos)) {
    corpus_av2[[i]]$content <- gsub(subj_pos[j], "subj_pos", corpus_av2[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(subj_neg)) {
    corpus_av2[[i]]$content <- gsub(subj_neg[j], "subj_neg", corpus_av2[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(synonyms)) {
    corpus_av2[[i]]$content <- gsub(synonyms[j], "cell", corpus_av2[[i]]$content)
  }
}

corpus_av2 <- tm_map(corpus_av2, removeNumbers)
corpus_av2 <- tm_map(corpus_av2, stripWhitespace)

dtm_av2 <- DocumentTermMatrix(corpus_av2)

# Removing sparse terms, keeping only words that appear in 0.5 % of reviews. 
sparse_av2 <- removeSparseTerms(dtm_av2, 0.995)

# Converting sparse, which is a list, to a matrix and then to a data frame.
sentSparse_av2 <- as.data.frame(as.matrix(sparse_av2)) 
rownames(sentSparse_av2) <- 1:nrow(sentSparse_av2)

# Making all variable names R-friendly.
colnames(sentSparse_av2) <- make.names(colnames(sentSparse_av2))

# Adding dependent variable.
sentSparse_av2 <- sentSparse_av2 %>% mutate(sentiment = reviews$sentiment)

# Splitting into training set and validation set.
set.seed(1)
ind <- createDataPartition(y = sentSparse_av2$sentiment, 
                           times = 1, p = 1/3, list = FALSE)
test_av2 <- sentSparse_av2[ind,]
train_av2 <- sentSparse_av2[-ind,]

# Building a CART model.
set.seed(1)
trControl = trainControl(method = "cv", number = 10, p = .9)
fit_cart_av2 <- train(sentiment ~ .,
                      method = "rpart",
                      data = train_av2,
                      tuneLength = 15, 
                      metric = "Accuracy",
                      trControl = trControl)

fitted_cart_av2 <- predict(fit_cart_av2)
confusionMatrix(as.factor(fitted_cart_av2), as.factor(train_av2$sentiment))

# Making predictions.
pred_test_av2 <- predict(fit_cart_av2, newdata = test_av2)
confusionMatrix(as.factor(pred_test_av2), as.factor(test_av2$sentiment))

# Avenue 3

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/reduced_stopwords.csv"
reduced_stopwords <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
reduced_stopwords <- sort(reduced_stopwords[, 2], decreasing = TRUE) %>% as.vector()

# Creating and preprocessing corpus.
corpus_av3 <- VCorpus(VectorSource(reviews$text)) 
corpus_av3 <- tm_map(corpus_av3, content_transformer(tolower))
corpus_av3 <- tm_map(corpus_av3, removeWords, stopwords("english"))

for (i in 1:nrow(reviews)) {
  corpus_av3[[i]]$content <- gsub("[[:punct:]]", " ", corpus_av3[[i]]$content)
}

corpus_av3 <- tm_map(corpus_av3, stemDocument)

for (i in 1:nrow(reviews)) {
  for (j in 1:length(reduced_stopwords)) {
    corpus_av3[[i]]$content <- gsub(reduced_stopwords[j], "", corpus_av3[[i]]$content)
  }
}

for (i in 1:nrow(reviews)) {
  for (j in 1:length(extra_stopwords)) {
    corpus_av3[[i]]$content <- gsub(extra_stopwords[j], " ", corpus_av3[[i]]$content)
  }
}

corpus_av3 <- tm_map(corpus_av3, removeNumbers)
corpus_av3 <- tm_map(corpus_av3, stripWhitespace)

dtm_av3 <- DocumentTermMatrix(corpus_av3)

# Removing sparse terms, keeping only words that appear in 0.5 % of reviews. 
sparse_av3 <- removeSparseTerms(dtm_av3, 0.995)

# Converting sparse, which is a list, to a matrix and then to a data frame.
sentSparse_av3 <- as.data.frame(as.matrix(sparse_av3)) 
rownames(sentSparse_av3) <- 1:nrow(sentSparse_av3)

# Making all variable names R-friendly.
colnames(sentSparse_av3) <- make.names(colnames(sentSparse_av3))

# Adding dependent variable.
sentSparse_av3 <- sentSparse_av3 %>% mutate(sentiment = reviews$sentiment)

# Splitting into training set and validation set.
set.seed(1)
ind <- createDataPartition(y = sentSparse_av3$sentiment, 
                           times = 1, p = 1/3, list = FALSE)
test_av3 <- sentSparse_av3[ind,]
train_av3 <- sentSparse_av3[-ind,]

# Building a CART model.
set.seed(1)
trControl = trainControl(method = "cv", number = 10, p = .9)
fit_cart_av3 <- train(sentiment ~ .,
                      method = "rpart",
                      data = train_av3,
                      tuneLength = 15, 
                      metric = "Accuracy",
                      trControl = trControl)

fitted_cart_av3 <- predict(fit_cart_av3)
confusionMatrix(as.factor(fitted_cart_av3), as.factor(train_av3$sentiment))

# Making predictions.
pred_test_av3 <- predict(fit_cart_av3, newdata = test_av3)
confusionMatrix(as.factor(pred_test_av3), as.factor(test_av3$sentiment))

# Loss of accuracy

############################################################
############################################################

# CART model will work as a critical yardstick. Several 
# other models are going to be applied.

# SVM
fit_svm <- train(sentiment ~ ., method = "svmRadialCost", data = train) 
fitted_svm <- predict(fit_svm)
confusionMatrix(as.factor(fitted_svm), as.factor(train$sentiment))

# Making predictions.
pred_svm <- predict(fit_svm, newdata = test)
confusionMatrix(as.factor(pred_svm), as.factor(test$sentiment))

# SVM + CV
trControl = trainControl(method = "cv", number = 10, p = .9)
fit_svm_cv <- train(sentiment ~ .,
                  method = "svmRadialCost",
                  data = train,
                  tuneLength = 15, 
                  metric = "Accuracy",
                  trControl = trControl)
fitted_svm_cv <- predict(fit_svm_cv)
confusionMatrix(as.factor(fitted_svm_cv), as.factor(train$sentiment))

# Making predictions.
pred_svm_cv <- predict(fit_svm_cv, newdata = test)
confusionMatrix(as.factor(pred_svm_cv), as.factor(test$sentiment))

# SVM + AV1
fit_svm_av1 <- train(sentiment ~ ., method = "svmRadialCost", data = train_av1) 
fitted_svm_av1 <- predict(fit_svm_av1)
confusionMatrix(as.factor(fitted_svm_av1), as.factor(train_av1$sentiment))

# Making predictions.
pred_svm_av1 <- predict(fit_svm_av1, newdata = test_av1)
confusionMatrix(as.factor(pred_svm_av1), as.factor(test_av1$sentiment))

# SVM + AV2
fit_svm_av2 <- train(sentiment ~ ., method = "svmRadialCost", data = train_av2) 
fitted_svm_av2 <- predict(fit_svm_av2)
confusionMatrix(as.factor(fitted_svm_av2), as.factor(train_av2$sentiment))

# Making predictions.
pred_svm_av2 <- predict(fit_svm_av2, newdata = test_av2)
confusionMatrix(as.factor(pred_svm_av2), as.factor(test_av2$sentiment))

# SVM + AV3
fit_svm_av3 <- train(sentiment ~ ., method = "svmRadialCost", data = train_av3) 
fitted_svm_av3 <- predict(fit_svm_av3)
confusionMatrix(as.factor(fitted_svm_av3), as.factor(train_av3$sentiment))

# Making predictions.
pred_svm_av3 <- predict(fit_svm_av3, newdata = test_av3)
confusionMatrix(as.factor(pred_svm_av3), as.factor(test_av3$sentiment))

# AdaBoost
fit_adaboost <- train(sentiment ~ ., method = "adaboost", data = train) 
fitted_adaboost <- predict(fit_adaboost)
confusionMatrix(as.factor(fitted_adaboost), as.factor(train$sentiment))

# Making predictions.
pred_adaboost <- predict(fit_adaboost, newdata = test)
confusionMatrix(as.factor(pred_adaboost), as.factor(test$sentiment))

# Random forest model
fit_rf <- train(sentiment ~ ., method = "rf", data = train) 
fitted_rf <- predict(fit_rf)
confusionMatrix(as.factor(fitted_rf), as.factor(train$sentiment))

# Making predictions.
pred_rf <- predict(fit_rf, newdata = test)
confusionMatrix(as.factor(pred_rf), as.factor(test$sentiment))

# GBM
fit_gbm <- train(sentiment ~ ., method = "gbm", data = train) 
fitted_gbm <- predict(fit_gbm)
confusionMatrix(as.factor(fitted_gbm), as.factor(train$sentiment))

# Making predictions.
pred_gbm <- predict(fit_gbm, newdata = test)
confusionMatrix(as.factor(pred_gbm), as.factor(test$sentiment))

# GBM + CV
trControl = trainControl(method = "cv", number = 10, p = .9)
fit_gbm_cv <- train(sentiment ~ .,
                    method = "gbm",
                    data = train,
                    tuneLength = 15, 
                    metric = "Accuracy",
                    trControl = trControl)
fitted_gbm_cv <- predict(fit_gbm_cv)
confusionMatrix(as.factor(fitted_gbm_cv), as.factor(train$sentiment))

# Making predictions.
pred_gbm_cv <- predict(fit_gbm_cv, newdata = test)
confusionMatrix(as.factor(pred_gbm_cv), as.factor(test$sentiment))

# XGBoost
fit_xgb <- train(sentiment ~ ., method = "xgbLinear", data = train) 
fitted_xgb <- predict(fit_xgb)
confusionMatrix(as.factor(fitted_xgb), as.factor(train$sentiment))

# Making predictions.
pred_xgb <- predict(fit_xgb, newdata = test)
confusionMatrix(as.factor(test$sentiment), as.factor(pred_xgb))

# MONMLP
fit_monmlp <- train(sentiment ~ ., method = "monmlp", data = train) 
fitted_monmlp <- predict(fit_monmlp)
confusionMatrix(as.factor(fitted_monmlp), as.factor(train$sentiment))

# Making predictions.
pred_monmlp <- predict(fit_monmlp, newdata = test)
confusionMatrix(as.factor(pred_monmlp), as.factor(test$sentiment))

# Pre-attentive features and insights with wordclouds

#########################
##################

x <- read.delim("test.txt", header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE)

colnames(x) <- c("text", "sentiment") 
x <- x %>% mutate(sentiment = 
  as.factor(gsub("1", "Appreciating", gsub("0", "Critisizing", x$sentiment)))) %>%
  as.data.frame()
x <- VCorpus(VectorSource(x$text)) 

# Preprocessing corpus: converting to lower-case, 
# removing punctuation and English stopwords, stemming document
x <- tm_map(x, content_transformer(tolower))

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_pos.csv"
subj_pos <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_pos <- sort(subj_pos[, 2], decreasing = TRUE) %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/subj_neg.csv"
subj_neg <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
subj_neg <- sort(subj_neg[, 2], decreasing = TRUE) %>% as.vector()

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/synonyms.csv"
synonyms <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
synonyms <- sort(synonyms[, 2], decreasing = TRUE) %>% as.vector()
rm(myfile)

myfile <- "https://raw.githubusercontent.com/Dev-P-L/Sentiment-Analysis/master/reduced_stopwords.csv"
reduced_stopwords <- read.csv(myfile, header = FALSE, stringsAsFactors = FALSE) 
reduced_stopwords <- sort(reduced_stopwords[, 2], decreasing = TRUE) %>% as.vector()
rm(myfile)

subj_pos <- read.csv("subj_pos.csv", header = FALSE, stringsAsFactors = FALSE) 
subj_pos <- sort(subj_pos[, 2], decreasing = TRUE) %>% as.vector()

subj_pos <- paste("", subj_pos, "") %>% as.vector()
subj_neg <- paste("", subj_neg, "") %>% as.vector()
synonyms <- paste("", synonyms, "") %>% as.vector()

x <- tm_map(x, content_transformer(tolower))

for (i in 1:3) {
  x[[i]]$content <- gsub("[[:punct:]]", " ", x[[i]]$content)
}

x <- tm_map(x, stripWhitespace)
x <- tm_map(x, removeWords, reduced_stopwords)
x <- tm_map(x, removeWords, extra_stopwords)

x <- tm_map(x, stripWhitespace)

which(subj_pos %in% x)
for (i in 1:3) {
  for (j in 1:length(subj_pos)) {
    x[[i]]$content <- gsub(subj_pos[j], " subj_pos ", x[[i]]$content)
  }
}

for (i in 1:3) {
  for (j in 1:length(subj_neg)) {
    x[[i]]$content <- gsub(subj_neg[j], " subj_neg ", x[[i]]$content)
  }
}

for (i in 1:3) {
  for (j in 1:length(synonyms)) {
    x[[i]]$content <- gsub(synonyms[j], " cell ", x[[i]]$content)
  }
}

x <- tm_map(x, removeWords, stopwords("english"))
x <- tm_map(x, removeNumbers)

x[[1]]$content
x[[2]]$content
x[[3]]$content

v  <- sort(stopwords("english"))
w <- read.csv("stopwords_with_apostrophe.csv", header = FALSE, stringsAsFactors = FALSE)
w <- w %>% as.data.frame() 
w <- w[, 2]
w <- as.character(w)
x <- setdiff(v, w) 
write.csv(x, "stopwords_without_apostrophe.csv")

try <- sample(rownames(train), size = nrow(train)/2, replace = TRUE)
try[1:10]
length(unique(try))
length(try)

try <- sample(try, size = nrow(train)/2, replace = TRUE)
try[1:10]
length(unique(try))
length(try)


