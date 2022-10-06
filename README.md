# Access to HTML Document

https://dev-p-l.github.io/Sentiment-Analysis--Amazon-Reviews/SA_Amazon_Insights&Results.html

<br>
<br>

# Presentation of Sentiment Analysis Project on Prediction of Amazon Reviews Polarity

## Files

- This project is comprised of twelve files. 
- All code is included in *SA_Amazon_Code.Rmd*. 
- Insights, results and methods are in *SA_Amazon_Insights&Results.html*.

- Wishing to visualize *SA_Amazon_Insights&Results.pdf* or *SA_Amazon_Insights&Results.oxps*? Very easy: just download them from GitHub.

- Wishing to visualize *SA_Amazon_Insights&Results.html*?
  * You could activate the hyperlink https://dev-p-l.github.io/Sentiment-Analysis--Amazon-Reviews/SA_Amazon_Insights&Results.html .
  * Alternatively, you could open it in GitHub Desktop.
  * Alternatively again, you could knit *SA_Amazon_Code.Rmd* or ask me by email for a copy of *SA_Amazon_Insights&Results.html* .

<br>

## Executive Summary

* **88 % prediction accuracy** has been reached on the validation set, against 50 % with a baseline model. Data is an Amazon sample provided in UCI Machine Learning Repository.

* In this sentiment analysis project, which factors have contributed towards that improvement with 38 percentage points?

* **Natural Language Processing** has contributed 21.7 percentage points: corpus, lowercasing, punctuation handling, stopword removal, stemming, tokenization from sentences into words, bag of words. 

* **Text mining** has brought additional accuracy improvement with 12.7 percentage points. The following insights have been determinant. 

* In decision trees predominate some tokens conveying subjective information; but other tokens containing subjective information have not been used in false negatives and false positives. Such ignored subjective information has been retrieved from random samples of false negatives and false positives, exclusively on the training set; customized lists have been established with tokens sorted as having either positive or negative sentiment orientation; occurrences of these tokens in reviews have been replaced with either a positive or a negative generic token. Polarization and text substitution have brought 10.3 percentage points out of the 12.7.

* Another insight has been about negation impact: negation has been fruitfully integrated, contributing 2.4 percentage points towards the 12.7 improvement from text mining. 

* **Machine learning optimization** has been performed across 10 models. Testing has been conducted on accuracy distributions across bootstrapped resamples. eXtreme Gradient Boosting has emerged as the most performing model in this project and has boosted accuracy with 3.6 additional percentage points. 
 
<br>

## TAGS
* sentiment analysis, natural language processing, text mining, subjective information, tokenization, bag of words, word frequency, wordcloud, decision trees, false negatives, false positives, text classification, polarization, lists of positive n-grams, lists of negative n-grams, text substitution, machine learning, binary classification, eXtreme Gradient Boosting, Monotone Multi-Layer Perceptron Neural Network, Random Forest, Stochastic Gradient Boosting, Support Vector Machines with Radial Basis Function Kernel, AdaBoost Classification Trees, bootstrapping, accuracy distribution across resamples, R

<br>

## Warning
* I reran the code in 2021 and obtained better results! Actually, this is due to the function *createDataPartition()* delivering other results with R version 4.0.3 (2020-10-10): the function creates the training set and the validation set in a different way and this introduces a bias! This is no improvement!

* Consequently, if you wish to run the code, I suggest that you rather run the code in the new version of this project, which you can find in another repository in this GitHub account. Sorry for the inconvenience. 
