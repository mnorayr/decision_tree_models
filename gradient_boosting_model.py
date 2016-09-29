######################################################################################
#H2O Gradient Boosting Model Implementation for ComEd Outage Prediction
#Prepared by Martin Copello (martin.copello@comed.com)
#
#Lines starting with '#' are simply comments
#Lines starting with '##' are commented-out pieces of code.
#
#There are two ways to run this script:
#
#The first one is 'as is'. This will create a uniformly distributed
#random variable across all data points to create a split of data
#for creating training and validation sets.
#
#The second way is by editing the commented-out regions. This will enable
#the option of setting default start and end years for training and validation
#or allowing the user to input these values.


######################################################################################
#Import libraries
######################################################################################

import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import plotly
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import Scatter


######################################################################################
#Initialize H2O and input data file
######################################################################################

h2o.init()
h2o.remove_all()
#Create H2O and Pandas frame with input file for training and validation
df = h2o.import_file(path="C:\\Users\E056899\Desktop\Plotly\ACT.csv")
di = pd.read_csv('C:\\Users\E056899\Desktop\Plotly\ACT.csv')
n = df.nrow


######################################################################################
#Create empty H2O Frames for Training and Validation
######################################################################################

train = h2o.H2OFrame (zip([0]*n))
valid = h2o.H2OFrame (zip([0]*n))


######################################################################################
#Enter start and end years
######################################################################################

#Allows user to input years for start and end of training and validation. These values have been set below. This may be commented out for flexibility.

##syt = input ('Enter start year for training set: ')
##while syt <1999:
##    print "Invalid Year"
##    syt = input ('Enter start year for training set: ')
##print "Start year for training set set to: ", syt
##
##eyt = input ('Enter end year for training set: ')
##while eyt>2015:
##    print "Invalid Year"
##    eyt = input ('Enter end year for training set: ')
##print "End year for training set set to: ", eyt
##
##syv = input ('Enter start year for validation set: ')
##while syv<1999:
##    print "Invalid Year"
##    syv = input ('Enter start year for validation set: ')
##print "Start year for validation set set to: ", syv
##
##eyv = input ('Enter end year for validation set: ')
##while eyv>2015:
##    print "Invalid Year"
##    eyv = input ('Enter end year for validation set: ')
##print "End year for validation set set to: ", eyv
syt = 2006
eyt = 2014
syv = 2014
eyv = 2015


######################################################################################
#Creates data for train, valid, and test vector, sends it to empty H2O frames
######################################################################################

#Splits input data based on years entered

##while (syt<=eyt):
##    stryt = str(syt)
##    date_train_vector = df['Date1']
##    search_train_vector = date_train_vector.countmatches(stryt)
##    train = train +search_train_vector
##    syt = syt +1
##while (syv<=eyv):
##    stryv = str(syv)
##    date_valid_vector = df['Date1']
##    search_valid_vector = date_valid_vector.countmatches(stryv)
##    valid = valid + search_valid_vector
##    syv = syv +1

#Create H2O frame for the testing set
df_test = h2o.import_file(path="C:\\Users\E056899\Desktop\data_2015\ExportFileWeather_2000.csv")

#There are two options for selecting training, validation, and testing:
    
#a) Creates frames for training and validationin sequential order according to the start and end years that have been picked

##df_train = df[train > 0]
##df_valid = df[valid > 0]
    
#b) Creates a 80/20 split for training and validation frames

r=df.runif()
df_train = df[r<0.8]
df_valid = df[r>=0.8]


######################################################################################
#Set up and run the model
######################################################################################

df_model = H2OGradientBoostingEstimator(distribution="gaussian",
                                        ntrees=100, max_depth=4, learn_rate=0.4,
                                        score_each_iteration = True, sample_rate = 0.6,
                                        col_sample_rate = 0.6)

#Run model with training and validation sets. 
df_model.train(x= range(2, df.ncol), y="ACT", training_frame=df_train, validation_frame=df_valid)
#Prints model information
print df_model
#Exports predictions to file pred_gbm.csv
pred_gbm = df_model.predict(df_test)
h2o.export_file(pred_gbm, path="C:\Users\E056899\Desktop\Plotly\GBM\pred_gbm.csv", force= True)

######################################################################################
#Adjust search parameters if performing random grid search
######################################################################################

##ntrees_opt = range(0, 1000, 1)
##max_depth_opt = range(0,10, 1)
##learn_rate_opt=[s/float(1000) for s in range (1, 101)]
##hyper_parameters = {"ntrees":ntrees_opt, "max_depth": max_depth_opt, "learn_rate":learn_rate_opt}
##search_criteria = {"strategy":"RandomDiscrete", "max_models":10, "max_runtime_secs":100, "seed":123456}
##from h2o.grid.grid_search import H2OGridSearch
##gs = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params=hyper_parameters, search_criteria = search_criteria)
##gs.train(x=range(2,df.ncol), y="ACT", training_frame=df_train, validation_frame = df_valid)
##print gs

######################################################################################
#Set up to plot, and plot
######################################################################################

#Read prediction data
da = pd.read_csv('pred_gbm.csv')
#Set up the scatter plot traces
Prediction = go.Scatter(x = di['Date1'], y = da['predict'], name="Prediction")
ACT = go.Scatter (x = di['Date1'], y = di['ACT'], name="Measured ACT")
#Create plotting data frame
data = [Prediction, ACT]
#Plot by date
plot(data, filename='GBM_Daily.html')

#Creates date vector based on 'Date1' column
datevector = di['Date1']
#Creates year vector based on the last four digits of the 'Date1' column
year = datevector.str[:4].astype(int)
#Adds columns to prediction file for 'Year' and 'ACT'
da['Year'] = year
da['ACT'] = di['ACT']
#Aggregates results by year
da = da.groupby("Year").sum()
#Creates additional year vector (necessary only because some distortion occurs to the 'Year' column when aggregated. Haven't figured it out yet.
da['Year2'] = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
               2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
               2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
#Set up the scatter plot traces
predictionyear2 = go.Scatter(x=da['Year2'], y = da['predict'], name="Prediction")
actyear2 = go.Scatter(x=da['Year2'], y = da['ACT'], name="Measured ACT")
#Create plotting data frame
data2 = [predictionyear2, actyear2]
#Plot by year
plot(data2, filename = 'GBM_Yearly.html')
