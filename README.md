#Gradient Boosting and Random Forests Models for ComEd Outage Prediction
###Prepared by Martin Copello (martin.copello@comed.com)

These scripts have been developed with the purpose of using existing weather and infrastructure data to predict outages in the 
forseeable future. Decision trees were tested as possible models due to their increasing popularity and efficiency in fitting 
non-linear data points. Lines in scripts starting with '#' are simply comments. Lines starting with '##' are commented-out pieces of code.

There are two ways to run these scripts:

1. The first one is 'as is'. They scripts, when run, will create a uniformly distributed random variable across all data points to
create a split of data between training and validation sets.
2. The second way involves editing the commented-out portions of the code. This will enable the option of setting default start and
end years for training and validation, or prompting the user to input these values.


