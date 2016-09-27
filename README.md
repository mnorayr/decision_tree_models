#Gradient Boosting and Random Forests Models for ComEd Outage Prediction
###Prepared by Martin Copello (martin.copello@comed.com)

These scripts have been developed with the purpose of using existing weather and infrastructure data to predict outages in the 
forseeable future. Decision trees were tested as possible models due to their increasing popularity and efficiency in fitting 
non-linear data points. In doing so, the possibility of using Gradient Boosting Models and Random Forests emerged. Lines in scripts starting with '#' are simply comments. Lines starting with '##' are commented-out pieces of code.

There are two ways to run these scripts:

1. The first one is 'as is'. They scripts, when run, will create a uniformly distributed random variable across all data points to
create a split of data between training and validation sets.
2. The second way involves editing the commented-out portions of the code. This will enable the option of setting default start and
end years for training and validation, or prompting the user to input these values.

####Gradient Boosting Models (GBMs)
This model is based on an iterative process of developing decision stumps (one-level decision tree), and correcting for errors through boosting. The algorithm develops a stump, scores for errors, and develops more stumps to correct for these errors. The process continues until the number of trees is that which was specified by the user, or the error score is smaller than a certain threshold for a certain number of iterations (also specified by the user). Visually, each decision stump fits a step function to the remaining errors. Because of this, they can approach data points fairly well, but fall short when extrapolating to test with parameters outside of its training range.


![A Simple Decision Stump](http://i.imgur.com/0bMkBK9.png)

This image depicts a simple decision stump


####Random Forests (RFs)
Similar to what GBMs do, RFs develop a series of decision trees using a process called bootstrap aggregation (or bagging), and averages the predictions of these trees. Bagging consists of drawing random data points with replacement from the training and validation data set. In RFs, decision trees are developed for these randomly drawn samples. The user may specify how many trees the model may develop. The extrapolated prediction that the user seeks will then be calculated by finding the average extrapolation of all models. The advantage that RFs have over GBMs is that they do not tend to overfit the data. However, they still find weakness in extrapolation predictions. 

An additional feature, available in H2O, is to run a model of Extremely Random Forests (ERFs), where another layer of randomness is added. GBMs and RFs find the optimal split at each decision node. However, ERFs find a random threshold by which to create the split. This helps in further reducing variance, but maintains the same weakness in extrapolating as GBMs or RFs. 



