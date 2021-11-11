

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Business Proposition](#business-proposition)
- [Executive Summary](#executive-summary)
- [Modeling Process](#modeling-process)
  - [Data Scraping and Cleaning](#data-scraping-and-cleaning)
  - [EDA](#eda)
  - [Building Models](#building-models)
  - [Model Results](#model-results)
- [Appendix](#appendix)


# Business Proposition

<hr 
 style="
 border:none;
 height:4px;
 background-color:DarkGray;
 ">

A baby food startup company has decided to use Reddit for their social media campaign to market their yet to be released baby food product. Your job as a data scientist is to guide the marketing team to design a data-driven marketing campaign.


# Executive Summary

<hr 
 style="
 border:none;
 height:4px;
 background-color:DarkGray;
 ">

 Reddit is consistently ranked amongst the top 10 popular social media sites in the USA. Based on Alexa ranking, it is currently ranked [#20](https://www.alexa.com/siteinfo/reddit.com) in global internet traffic and engagement. Ad competition on Reddit is low and the inventory is cheap. However, advertising on Reddit is notoriously difficult. The users of Reddit (redditors) in general value authenticity a lot. So to make a connection with the redditors, advertisers need to engage with the community. A good way to communicate with the community is to understand the relevant topics of discussions, their interests, trends being followed and the subreddits lingos.
 Reddit posts are a great resource for data mining. Natural language processing tools can be utilized to analyze the subreddits of interest to discover hidden trends, ad-keywords, to generate relevant promotional posts and content, make sure the ads are posted to relevant subreddits.
 The two subreddits of interest for this promotional campaign are [r/BabyBumps](https://www.reddit.com/r/BabyBumps/) and [r/beyondthebump](https://www.reddit.com/r/beyondthebump/). To help the marketing team design a campaign to advertise on reddit, a [LogisticRegression ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classification model is built to correctly identify which subreddit a post belongs to so that the relevant ads show up on the subreddits. Additionally, data analysis of these subreddits using NLP would help guide the decisions around generating relevant content that can keep the redditors engaged and interested. 

# Modeling Process

<hr 
 style="
 border:none;
 height:4px;
 background-color:DarkGray;
 ">

## Data Scraping and Cleaning

To build a classification model for this NLP project, I scraped  reddit posts each from r/BabyBumps and r/beyondthebump subreddits.  [Pushshift](https://github.com/pushshift/api) Reddit API is used to scrape the data from the website. The API was set up to scrape 5000 reddit posts from each of these subreddits. After scraping the data, texts were cleaned to

- Remove any emojis, hyperlinks and other non-unicode characters
- Empty posts either in the title or body of the posts were removed
- Regular expressions is used to substitute certain abbreviated texts such as 1w2d = 1 week 2 days, 2mos = 2 months
- Certain posts were filtered such as stickied posts, posts with zero comments, deleted posts either by the moderator or the author

## EDA

To get a better understanding of the posts on each of the subreddits, NLP tools such as [Spacy](https://spacy.io) and [NLTK](https://www.nltk.org) are used. These EDA analysis gave much needed insight into the posts and to understand  how similar and different the posts of the subreddits are from each other. Text data EDA is done to investigate:

- Frequently used words
- Understand the lingos common to each of them
- Named entity recognition using spacy and compare the characteristics of each of the entities
- Sentiment Analysis

## Building Models

Once the data was scraped and cleaned, a base-line was established. For our dataset, the baseline accuracy is 41%. Four different classification models are built to identify the subreddits based on the content of posts. These are

- [LogisticRegression](https://www.reddit.com/r/beyondthebump/)
- [MultinomialNB Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [RandomForest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [C-Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

All of these models were overfit when run with default hyperparameters

[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) libraries are used to tune the hyperparameters and reduce the overfitting. [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) of transformers such as [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and final estimators are used too.

## Model Results

- RandomForest Classifier was the most overfit model even after tuning the hyperparameters
- SVC was relatively less overfit as compared to the RandomForest
- MNB model had a reasonable bias-variance tradeoff but its testing scores were the lowest
- LogisticRegression model is the best model and its traning and testing scores are almost similar
  
The final LogisticRegression model has a training score and testing score of 85.5%. The AUC score which informs the separation between True Positives and True Negatives is 0.92. In general, the model does relatively better in identifying the posts belonging to the r/beyondthebump subreddit as compared to r/BabyBumps subreddit. The recall score for r/beyondthebump is 0.9 and for r/BabyBumps is 0.79. The f1-score for the model is 0.817. The model coefficients of the LogisticRegression model are analyzed to identify the top predictor words for each of the subreddits. 'trimester', 'pregnant', 'week' are the top 3 predictor words for r/BabyBumps and 'month', 'old', 'LO' are the top 3 predictor words for r/beyondthebump.


# Appendix

<hr 
 style="
 border:none;
 height:4px;
 background-color:DarkGray;
 ">

<details>
  <summary><b>Directory Tree Structure</b></summary>

```
├── README.md
├── code
│   ├── 01.Data_Scraping.ipynb
│   ├── 02.EDA.ipynb
│   ├── 03.Classifier_Models.ipynb
│   └── 04.Models_Summary_and_Evaluation.ipynb
├── data
│   ├── df.csv
│   └── df_raw.csv
├── models
│   ├── logreg_gs.sav
│   ├── logreg_gs_v10.sav
│   ├── logreg_gs_v11.sav
│   ├── logreg_gs_v12.sav
│   ├── logreg_gs_v13.sav
│   ├── logreg_gs_v14.sav
│   ├── logreg_gs_v15.sav
│   ├── logreg_gs_v2.sav
│   ├── logreg_gs_v3.sav
│   ├── logreg_gs_v4.sav
│   ├── logreg_gs_v5.sav
│   ├── logreg_gs_v6.sav
│   ├── logreg_gs_v7.sav
│   ├── logreg_gs_v9.sav
│   ├── mnb_gs_v1.sav
│   ├── mnb_gs_v2.sav
│   ├── mnb_gs_v3.sav
│   ├── mnb_gs_v4.sav
│   ├── mnb_gs_v5.sav
│   ├── rscv_v1.sav
│   ├── rscv_v2.sav
│   ├── svc_cv_v1.sav
│   └── svc_cv_v2.sav
└── presentation
    └── NLP_Subreddit_Classification.pdf
```
</details>


