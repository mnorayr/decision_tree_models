######################################################################################
#H2O Random Forest Model Implementation for ComEd Outage Prediction
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
h2o.init()
from h2o.estimators.random_forest import H2ORandomForestEstimator
import plotly
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Bar, Scatter, Figure, Layout
import os


######################################################################################
#Initialize H2O and input data file
######################################################################################

h2o.init()
h2o.remove_all()
#Create H2O and Pandas frame with input file for training and validation
df = h2o.import_file(path="C:\\Users\E056899\Desktop\RDF\ACT.csv")
di = pd.read_csv('C:\\Users\E056899\Desktop\RDF\ACT.csv')
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
#syt = Start Year Training Set; eyt = End Year Training Set;
#syv = Start Year Validation Set; eyv = End Year Validation Set;

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

##syt = 2006
##eyt = 2014
##syv = 2014
##eyv = 2015


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

#Create H2O and Pandas frames for the testing set
df_test = h2o.import_file(os.path.realpath("C:\Users\E056899\Desktop\RDF\ExportFileWeather_2000.csv"))
df_test_pyt = pd.read_csv("C:\Users\E056899\Desktop\RDF\ExportFileWeather_2000.csv")

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

#Define the Random Forest model. Set histogram_type = 'random' for extremely random forests. Otherwise, set to 'AUTO'
df_model = H2ORandomForestEstimator(ntrees =100, max_depth=10, min_rows = 1, nbins = 20,
                                    seed = -1, mtries = -1, sample_rate = 0.6320000290870667,
                                    stopping_rounds=0, stopping_tolerance = 0.001,
                                    score_each_iteration=True,
                                    stopping_metric = 'AUTO',col_sample_rate_per_tree = 0.63,
                                    min_split_improvement = 0, col_sample_rate_change_per_level = 0.63,
                                    histogram_type="random")
#Run model with training and validation sets.
df_model.train(x=range(2,df.ncol), y="ACT", training_frame=df_train, validation_frame = df_valid)
#Prints model information
print df_model
#Exports predictions to file pred_rf.csv
pred_rf = df_model.predict(df_test)
h2o.export_file(pred_rf, path="C:\Users\E056899\Desktop\RDF\pred_rf.csv", force= True)

######################################################################################
#Set up to plot, and plot
######################################################################################

#Read prediction data
da = pd.read_csv('C:\Users\E056899\Desktop\RDF\pred_rf.csv')
#Set up the scatter plot traces
Prediction = go.Scatter(x = di['Date1'], y = da['predict'], name ='Prediction')
ACT = go.Scatter (x = di['Date1'], y = di['ACT'], name='Measured ACT')
#Create plotting data frame
data = [Prediction, ACT]
#Plot by date
plot(data, filename='RF_Daily.html')

#Creates date vector based on 'Date1' column
datevector = df_test_pyt['Date1']
#Creates year vector based on the last four digits of the 'Date1' column
year = datevector.str[:4].astype(int)
#Adds columns to prediction file for 'Year' and 'ACT'
da['Year'] = year
da['ACT'] = di['ACT']
#Aggregates results by year
da = da.groupby("Year").sum()
#Creates additional year vector (necessary only because some distortion occurs to the 'Year' column when aggregated. Haven't figured it out yet.
da['Year2'] = [1999, 2000, 2001, 2002, 2003, 2004, 2005,
               2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
               2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
               2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,2030, 2031]
#Set up the scatter plot traces
predictionyear2 = go.Scatter(x=da['Year2'], y = da['predict'], name='Prediction')
actyear2 = go.Scatter(x=da['Year2'], y = da['ACT'], name='Measured ACT')
#Create plotting data frame
data2 = [predictionyear2, actyear2]
#Plot by year
plot(data2, filename = 'RF_Yearly.html')
