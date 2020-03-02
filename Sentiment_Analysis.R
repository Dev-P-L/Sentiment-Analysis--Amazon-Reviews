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
df <- read.delim(myfile, header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE)
rm(myfile)

# Preprocessing data frame.
colnames(df) <- c("text", "sentiment") 
df <- df %>% mutate(sentiment = 
  as.factor(gsub("1", "Appreciating", gsub("0", "Critisizing", df$sentiment))))
as_tibble(head(df))

# Creating corpus.
corpus <- VCorpus(VectorSource(df$text)) 

# Preprocessing corpus: converting to lower-case, 
# removing punctuation and English stopwords, stemming document
corpus <- tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, c("amazon", stopwords("english")))
corpus = tm_map(corpus, stemDocument)

# Creating matrix with rows corresponding to documents (corpus)
# and columns corresponding to words.
dtm <- DocumentTermMatrix(corpus)

# Looking at matrix.
inspect(dtm[500:520,500:510])
# This matrix is sparse: it contains many zeroes. 
# Checking for sparsity.
findFreqTerms(dtm, lowfreq = 10)

# Removing sparse terms, keeping only words that appear in 0.5 % of documents (sent). 
sparse <- removeSparseTerms(dtm, 0.99)

# Converting sparse, which is a list, to a matrix and then to a data frame.
sentSparse <- as.data.frame(as.matrix(sparse))

# Making all variable names R-friendly.
colnames(sentSparse) <- make.names(colnames(sentSparse))

# Adding dependent variable.
sentSparse <- sentSparse %>% mutate(sentiment = df$sentiment)

# Splitting into training set and validation set.
set.seed(1)
ind <- createDataPartition(y = sentSparse$sentiment, 
                           times = 1, p = 0.5, list = FALSE)
val <- sentSparse[ind,]
train <- sentSparse[-ind,]

# Building a CART model.
fit_cart <- train(sentiment ~ ., method = "rpart", data = train) 
fitted_cart <- predict(fit_cart)

# Making predictions.
pred_cart <- predict(fit_cart, newdata = val)
mean(val$sentiment == pred_cart)
table(as.factor(val$sentiment), as.factor(pred_cart))
confusionMatrix(as.factor(val$sentiment), as.factor(pred_cart))

############################################################

# Exploratory analysis and insights

# Removing factual words
temp <- sentSparse[, -ncol(sentSparse)]
set.seed(1)
wordcloud(colnames(temp),colSums(temp), min.freq = 10, max.words=50,
  random.order = FALSE, rot.per = 1/3, colors = brewer.pal(8, "Dark2"), 
  scale = c(4,.5))
# Remove: phone, headset, ear, batteri, item, use, servic, 
# time, sound, item, car, charg

factual <- c("phone", "headset", "ear", "batteri", "item", "use", "servic", 
             "time", "sound", "item", "car", "charg")


# Random forest model
fit <- train(sentiment ~ ., method = "monmlp", data = train) 
fitted <- predict(fit)

# Making predictions.
pred <- predict(fit, newdata = val)
mean(val$sentiment == pred)
table(as.factor(val$sentiment), as.factor(pred))

# Accuracy
confusionMatrix(as.factor(val$sentiment), as.factor(pred))

