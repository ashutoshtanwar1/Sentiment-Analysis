# Sentiment-Analysis
Sentiment Analysis: To determine, from a text corpus, whether the  sentiment towards any topic or product etc. is positive, negative, or neutral.


# Purpose
The main purpose of this analysis to build a model that predict whether the review given by the user is postive or negative.
To do so, we will work on Restaurant Review dataset, we will load it into predicitve algorithms Gaussian Naive Bayes.


# Dataset
Dataset: Restaurant_Reviews.tsv is a dataset from Kaggle datasets which consists of 1000 reviews on a restaurant.


# To build a model to predict if review is positive or negative, following steps are performed.

-->Importing Dataset : Importing the Restaurant Review dataset using pandas library.

-->Preprocessing Dataset : Each review undergoes through a preprocessing step, where all the vague information is removed.

-->Vectorization : From the cleaned dataset, potential features are extracted and are converted to numerical format. 
                   The vectorization techniques are used to convert textual data to numerical format. Using vectorization, a matrix is created where each column represents a feature and each row represents an individual review.

-->Training and Classification : Further the data is splitted into training and testing set using Cross Validation technique. 
                                 This data is used as input to classification algorithm.
                                 
-->Analysis Conclusion : In this study, an attempt has been made to classify sentiment analysis for restaurant reviews using machine learning techniques. Two algorithms namely Multinomial Naive Bayes and Bernoulli Naive Bayes are implemented.
                         Accuracy : 71 %.
