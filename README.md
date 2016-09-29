#Gradient Boosting and Random Forests Models for ComEd Outage Prediction
##Prepared by Martin Copello (martin.copello@comed.com)

These scripts have been developed using Python and the H2O library with the purpose of using existing weather and infrastructure data to predict outages in the forseeable future. Decision trees were tested as possible models due to their increasing popularity and efficiency in fitting non-linear data points. In doing so, the possibility of using Gradient Boosting Models and Random Forests emerged. Outputs of yearly and daily outage predictions are generated. Lines in scripts starting with '#' are simply comments. Lines starting with '##' are commented-out pieces of code.


###A Brief Overview of Concepts

####Gradient Boosting Models (GBMs)
This model is based on an iterative process of developing decision stumps (one-level decision tree), and correcting for errors through boosting. The algorithm develops a stump, scores for errors, and develops more stumps to correct for these errors. The process continues until the number of trees is that which was specified by the user, or the error score is smaller than a certain threshold for a certain number of iterations (also specified by the user). Visually, each decision stump fits a step function to the remaining errors. Because of this, they can approach data points fairly well, but fall short when extrapolating to test with parameters outside of its training range.

![A Simple Decision Stump](/decision_stump.png)

This image depicts a simple decision stump


####Random Forests (RFs)
Similar to what GBMs do, RFs develop a series of decision trees using a process called bootstrap aggregation (or bagging), and averages the predictions of these trees. Bagging consists of drawing random data points with replacement from the training and validation data set. In RFs, decision trees are developed for these randomly drawn samples. The user may specify how many trees the model may develop. The extrapolated prediction that the user seeks will then be calculated by finding the average extrapolation of all models. The advantage that RFs have over GBMs is that they do not tend to overfit the data. However, they still find weakness in extrapolation predictions. 

An additional feature, available in H2O, is to run a model of Extremely Random Forests (ERFs), where another layer of randomness is added. GBMs and RFs find the optimal split at each decision node. However, ERFs find a random threshold by which to create the split. This helps in further reducing variance, but maintains the same weakness in extrapolating as GBMs or RFs. 

###Training and Validation Sets: Random vs. User-Defined

The scripts allow for partioning of sets of data points for training and validation in two ways. The first one is 'as is'. The scripts will create a uniformly distributed random variable in the range of [0, 1] for each data point in the data set. Further, an 80/20 split based on generated values of this random variable will define the training and validation. This means that 80% of total observations are randomly selected to be part of the training set, and 20% of the validation set. This way, there will be training and validation data points throughout the entire data set. For consistency in modelling, H2O offers the option of setting a random number seed, resulting in the same training and validation sets. 

The second way to to run the scripts is by manually defining the training and validation sets, which can be useful if the user has an idea of what observations should comprise the testing, and validation set. The script allows the user to input start and end years for training and validation sets. This option can be commented out, so that the start and end years are constant across script runs.

![GBM with 100 trees, using random partitioning for training and validation sets](/gbm_yearly_random.png) 

GBM with 100 trees, using random partitioning for training and validation sets

![GBM with 100 trees, using user-defined partitioning for training and validation sets](/gbm_yearly_user_defined.png)

GBM with 100 trees, using user-defined partitioning for training and validation sets

As can be seen, the definition of the training and validation set may have a significant effect on the extrapolated predictions on the testing set.

Note: Testing is performed against a weather-year data CSV file. However, a testing set may be defined within this data file in the same way, by random or user-defined partitioning. 


###Tuning

H2O offers a variety of tuning parameters for both GBMs and RFs. For a comprehensive list of parameters, click [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html#defining-a-gbm-model) for GBMs, and [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html#defining-a-drf-model) for RFs. Some parameters that were found important to our analysis are described briefly below:


####Parameters for GBMs and RFs:

x: Vector containing the names of predictors used to build model

y: Vector containing the name of the response variable

training_frame: an H2O Frame containing the data to be used for model training

validation_frame: an H2O Frame containing the data to be used for model validation

ntrees: Defines the number of trees the model will generate. Default is 50.

max_depth: Defines the tree depth. Default is 5.

seed: Especially useful when looking to replicate sampling in model development

score_each_iteration: Indicates whether to score during each iteration of model training

stopping_metric: Determines metric used for early stopping. Can be AUTO, deviance, logloss, MSE, AUC, 2r, or misclassification

stopping_rounds: Early stopping based on convergence of stopping_metric.

####Parameters for GBMs only:

distribution: GBMs support multiple distributions (AUTO, bernoulli, multinomial, gaussian, poisson, gamma, or tweedie)

####Parameters for RFs only:

histogram_type: By default (AUTO) DRF bins from min...max in steps of (max-min)/N. Random split points or quantile-based split points can be selected as well. RoundRobin can be specified to cycle through all histogram types (one per tree). Use this option to specify the type of histogram to use for finding optimal split points: AUTO, UniformAdaptive, Random, QuantilesGlobal, RoundRobin.

Note: H2O supports extremely randomized trees via histogram_type="Random". In extremely randomized trees (Extra-Trees), randomness goes one step further in the way splits are computed. As in Random Forests, a random subset of candidate features is used, but instead of looking for the best split, thresholds (for the split) are drawn at random for each candidate feature, and the best of these randomly-generated thresholds is picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias.

###Outcome

Decision tree training is based on the iterative fitting of step functions into data sets, each time optimizing for error correction at all points. Because of this, they can fit data very well. However, it was found that decision trees are a viable model to implement for future outage prediction. Because some predictors (such as bad cable mileage), which we expect to be especially significant in reducing outages, find themselves out of range from the values on which the model was trained and validated, the model fails at recognizing the effect this predictor might have in the future if taken out of range. Because of this, the results of the extrapolation that decision trees may predict cannot be trusted. 

![Example of Decision Trees' Data Fitting Through Step Functions](/error_residual.PNG) 

Example of Decision Trees' Data Fitting Through Step Functions

As previously noted, it was decided to use random partitioning for training and validation sets.
