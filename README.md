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
Are based on an iterative process of developing decision stumps (one-level decision tree). The algorithm develops a stump, scores for errors, and develops more stumps to correct for these errors. The process continues until the number of trees is that which was specified by the user, or the error score is smaller than a certain threshold for a certain number of iterations (also specified by the user). Visually, each decision stump fits a step function to the remaining errors. Because of this, they can approach data points fairly well, but fall short when tested with parameters outside of its training range.
![A Simple Decision Stump](http://i.imgur.com/seFXv2h.png)


