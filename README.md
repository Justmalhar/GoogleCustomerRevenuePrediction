# GoogleCustomerRevenuePrediction
Performed exploratory data analysis on the datasets and reduced the number of features by feature extraction and feature selection. Designed LightGBM, XGBoost, Catboost and ensemble models to predict the revenue that a customer will generate. Analyzed the prediction results by evaluating factors such as validation score, accuracy, mean absolute error, and RMSE. Altered the dataset and identified which features play significant roles in generating revenue.

# INTRODUCTION
Businesses constantly face a challenge in converting visitors into customers, only a small fraction of customers produce most of the revenue. Marketing teams need to rethink how to target the appropriate audience and assign budgets, thorough data analysis can prove to be a solution to this problem.
However, data mining is a difficult task and requires understanding of the data and how to model the data to find the underlying relations, which can assist us to perform predictions. In this project we have extensively worked to first understand the underlying data as it involves over 12 columns compressed and 100 uncompressed. So more work has been done in exploration of the dataset and after which we used existing prediction models to make predictions for the revenue per customer.

# Project Objective
- Alter and analyze dataset to identify which features play significant roles in generating revenue, and how they affect the prediction model.
- Build prediction models to predict the revenue that a customer will generate based on training dataset
- Compare the prediction models by evaluating factors such as accuracy, mean absolute error, RMSE etc.

# Exploratory Data Analysis
The dataset has a 903653 rows and a total of 12 features and 1 target which is the revenue. In the dataset, out of the 12 features, there are several features which are dictionaries that contain other features. Thus, if Merchandise stores are well-known users of data mining techniques. Many stores offer free loyalty cards to customers that give them access to reduced prices not available to non-members. The cards make it easy for stores to track who is buying what, when they are buying it and at what price. After analyzing the data, stores can then use this data to offer customers coupons targeted to their buying habits and decide when to put items on sale or when to sell them at full price.

The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies. In this kaggle competition,we are challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer. Hopefully, the outcome will be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.

# About the Dataset
We try to analyze the data by considering them as separate features, the total number of features in the dataset would be around 60. Each row in the dataset is one visit to the store. We are predicting the natural log of the sum of all transactions per user which we will test using RMSE.

We are given two datasets: train.csv and test.csv The data fields in the given files are:
_fullVisitorId_ - A unique identifier for each user of the Google Merchandise Store. _channelGrouping_ - The channel via which the user came to the Store.
_date_ - The date on which the user visited the Store.
_device_ - The specifications for the device used to access the Store.
_geoNetwork_ - This section contains information about the geography of the user.
_sessionId_ - A unique identifier for this visit to the store.
_socialEngagementType_ - Engagement type, either "Socially Engaged" or "Not Socially Engaged". _totals_ - This section contains aggregate values across the session.
_trafficSource_ - This section contains information about the Traffic Source from which the session originated.
_visitId_ - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
_visitNumber_ - The session number for this user. If this is the first session, then this is set to 1. _visitStartTime_ - The timestamp (expressed as POSIX time).

In our case, we know that some columns have json blobs. Our target feature is _transactionRevenue_ which we got from _totals_ column in the original training dataset. Thus, while loading the dataset into a dataframe we converted all the json blobs into individual columns.

We first group the users according to the _transactionRevenue_ feature. Then we plot the distribution for the same and observe the following results:
 
This graph confirms the 80/20 rule which we mentioned as part of the introduction - only a small percentage of customers produce most of the revenue.

## Unique and Constant Values:

We observe that there are several constant and NaN columns which do not add any meaning to the analysis of the dataset. We are primarily removing them to save memory and time.

After this we focus on keeping and understanding all the unique values present in the dataset. These have been summarized below:
- Feature: transactionRevenue
Number of instances in train set with non-zero revenue : 11515 and ratio is : 0.01274272314704 Number of unique customers with non-zero revenue : 9996 and the ratio is : 0.013996726255903731
- Feature: fullVisitorId
- Feature: channelGrouping
  Number of unique visitors in train set : 714167 out of rows : 903653 Number of unique visitors in test set : 617242 out of rows : 804684 Number of common visitors in train and test set : 7679
 Number of unique channelGroupings in train set : 8
Number of unique channelGroupings in test set : 8
Number of common channelGroupings in train and test set : 8
Similarly we have performed the analysis for several features which can be observed in the Python notebo ok.
 
## Device Information Analysis:
A lot of meaningful information is present in the _device_ column of the dataset. We plot the following g raphs for them:
  Device Browser
  Device Category 
  Device Operating System
 We draw the following conclusions from the analysis of data:
- Device browser distribution looks similar on both the count and count of non-zero revenue plots.
- Desktop seem to have higher percentage of non-zero revenue counts compared to mobile devices.
- In case of device operating system, though the number of counts is more from windows, the number of counts where revenue is not zero is more for Macintosh.
  o Chrome OS also has higher percentage of non-zero revenue counts
  o On the mobile OS side, iOS has more percentage of non-zero revenue counts compared to Android
  
## Date and Time Analysis:
Train Set Test Set
From the above analysis in train set we can see that:
• We have data from 1 Aug 2016 to 31 July 2017 in our training dataset
• In Nov 2016, though there is an increase in the count of visitors, there is no increase in non-zero revenue counts during that time period (relative to the mean).
 While in the test set we see that:
 
• We have dates from 2 Aug 2017 to 30 Apr 2018.
• So, there are no common dates between train and test set.
• So, it might be a good idea to do time based validation for this dataset.
 
## Geographic Information Analysis:
Continent
Sub-continent
• In the sub-continent plot, we can see that America has both higher number of counts as well as highest number of counts where the revenue is non-zero
• Though Asia and Europe has high number of counts, the number of non-zero revenue counts from these continents are comparatively low.
• The mean revenue of eastern Asia is the highest even though the non-zero revenue count is very less.
City
• In the city plot, we can see that most entries are not available, which can cause infering incoreect information from the plot.
Country
• In the country plot, we can see that United states has the most entries while the mean revenue of
Japan is the highest. Network Domain
Totals Analysis:
PageViews
Traffic Source Analysis:
Source
Medium
• "referral" has more number of non-zero revenue count compared to "organic" medium.
• In the continent plot, we can see that America has both higher number of counts as well as highest number of counts where the revenue is non-zero
• Though Asia and Europe has high number of counts, the number of non-zero revenue counts from these continents are comparatively low.
• The mean revenue of Africa is the highest even though the count is very less.
 • The number of counts with non-zero revenue for "unknown.unknown" is lower than that for "(not set)".
  • totalCount and Non variables look very predictive.
• Count plot shows decreasing nature i.e. we have a very high total count for less number of hits
and page views per visitor transaction and the overall count decreases when the number of hits
per visitor transaction increases.
• On the other hand, we can clearly see that when the number of hits / pageviews per visitor
transaction increases, we see that there is a high number of non-zero revenue counts.
 • Though Youtube has high number of counts, the number of non-zero revenue counts are very less.
• Google plex has a high ratio of non-zero revenue count to total count in the traffic source plot.
  
#Modeling Dataset
Before we start building models, we see if there are any features which are there in training dataset but not in test dataset. The feature "trafficSource.campaignCode" was not present in test dataset and hence we removed it from the dataset. Also we drop the constant variables which we got earlier. Also we drop the "sessionId" as it is a unique identifier of the visit. We impute 0 for missing target values as a part of preprocessing.

After performing some preprocessing on the dataset, we split it into development and validation sets based on the time. We have used the following three models for the modeling and prediction:

## LightGBM
Next we create a LightGBM Model which we train using our preprocessed dataset.Our task is to calculate the sum for all the transactions of the user and then do a log transformation on top. The result of the Gradient Boosting is seen below: [Image can be seen in the Python notebook]
We can see here that according to LightGBM model, "totals.pageviews" is the most important feature followed by "totals.hits" and "visitStartTime" and so on.
Now let's try to further analize our model and see what happens when we increase the boosting rounds to 2000. We also changed verbose_eval to 50 to see the difference in values in a better way.
 
 Let's compute the evaluation metric on this new validation data.
1.709925685736863
Here, we got a validation score of 1.70 using the new method, which is the same as the previous one. We
can see that our best iteration remains to be iteration number 254. XGBoost (Extreme Grandient Boosting):
2.224021832325654
Here, we got a validation score of 2.22 using XGBoost.
## Catboost
1.7094433862016563
Here, we got a validation score of 1.70 using CatBoost.
  
## Ensemble Method
The best ensemble is: ensemble1 with weights 0.7 for LightGBM, 0.3 for CatBoost and 0.0 for XGBoost

#Summary
The following results were obtained after the training and testing of the dataset using ensemble classifiers:
The LightGBM model achieved the best results with regards to the training, validation. On the other hand we observed that, Catboost is sensitive to model overfitting and that is one of the reason why the model was shrinked to first 197 iterations. XGBoost did not perform much better on this dataset and took much more time as compared to others (from minutes to hours).
