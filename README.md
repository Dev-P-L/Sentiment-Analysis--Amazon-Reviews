# Sentiment-Analysis
In progress
## Executive Summary


First, Natural Language Processing has been conducted in terms of
lowercasing, punctuation removal, stemming, tokenization and bag of words,
followed by a first CART prediction on the training set. Some fine tuning 
proved necessary to cope with short forms (intra word contractions), 
punctuation marks stuck to words, alternative grammar, etc. 

Second, text mining has focused on word frequency, wordclouds, decision trees and
analysis of false positives and of false negatives, which have 
been pinpointed as a weak point. Text mining has opened up two
avenues for improvement: reintegrating negational unigrams ("not", etc.)
and text classification. Text classification applied to unigrams 
or multigrams conveying some subjective information that were present 
in false negatives or positives but didn't show in decision trees; 
they have been classified as negative of positive sentiment orientation
and replaced with one generic negative sentiment token and one
generic positive sentiment token. Running CART again propelled 
accuracy to higher levels. 

Third, machine learning models were then compared in accuracy and,
complementarily, in other performance metrics. Three out of ten 
have been picked up: svmRadialCost, which delivered 
the highest accuracy level, rf and xbgBoost, which produced 
the highest specificity. To keep on the safe line, an ensemble model
has been built up by majority vote. The ensemble model 
has produced an accuracy level of 87 %, 
to be compared with 50 % for the baseline model.
