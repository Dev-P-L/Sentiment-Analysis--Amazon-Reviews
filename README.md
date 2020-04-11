# Sentiment-Analysis
In progress
## Executive Summary

In this project, **sentiment analysis** has been conducted on produced an Amazon data sample provided in UCI Machine Learning Repository. Predictive accuracy of 88 % has been reached on the validation set, against 50 % with the baseline method. 

First, **Natural Language Processing** has been performed: corpus, lowercasing, punctuation handling, stopword removal, stemming, tokenization from sentences into words and bag of words. With NLP, accuracy has gained 25 percentage points.

Second, **text mining** has brought additional accuracy improvement with 10 percentage points. Two insights have been determinant: in decision trees tokens conveying subjective information predominate; but other pieces of subjective information are not used in numerous false negatives and false positives. Such ignored subjective information has been retrieved from random samples of false negatives and false positives, exclusively on the training set; customized lists have been established with tokens having either positive or negative sentiment orientation; occurrences of these tokens in reviews have been replaced either with a positive or a negative generic token.

Third, **machine learning optimization** has boosted accuracy with 4 additional percentage points. Testing has been conducted on accuracy distributions across bootstrapped resamples. eXtreme Gradient Boosting has emerged as the most performing model in this project. 


## TAGS
sentiment analysis, natural language processing, text mining, subjective information, tokenization, bag of words, word frequency, wordcloud, decision trees, false negatives, false positives, text classification, polarization, lists of positive n-grams, lists of negative n-grams, text substitution, machine learning, binary classification, eXtreme Gradient Boosting, Monotone Multi-Layer Perceptron Neural Network, Random Forest, Stochastic Gradient Boosting, Support Vector Machines with Radial Basis Function Kernel, AdaBoost Classification Trees, bootstrapping, accuracy distribution across resamples, R
