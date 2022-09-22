#importing all the libraries used for the analysis 
from turtle import speed
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 


import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Model
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor

#loading the csv files of the weather station data 
CG17O = pd.read_csv("CG17 Oxenholme Stn 18.1724X October.csv")
CGJ6 = pd.read_csv("CGJ6 Barton Broughton 4.0985X October.csv")
CG17P = pd.read_csv("CGJ7 Penrith 51.0660X October.csv")
DJH = pd.read_csv("DJH Hellifield 34.1457X October.csv")
WCM1 = pd.read_csv("WCM1 Quintshill 10.0434X October.csv")
ULR = pd.read_csv("ULR Carlisle Upperby 0.0309X October.csv")
CG17S = pd.read_csv("CGJ7 Shap Summit 37.1341X October.csv")

#creating csv file for the combined weather data 
weatherDataPath = 'weather_data_combined.csv'
#creating csv file for the combined delay data 
combinedDelayDataPath = 'delay_data_combined.csv'

#read the csv files of the delays 
customisedDelaysOther = pd.read_csv("Customised List of Delays - NR Other Weather Codes - periods 2101-2208.csv")
customisedDelaysSevere = pd.read_excel("Customised List of Delays - NR Severe Weather Codes - periods 2101-2208.xlsx")
customisedDelaysVW = pd.read_excel("Customised List of Delays - VW Weather Code for TOC on Self and Operator on Operator - periods 2101-2208.xlsx")

#turn string to date formate for all the journey dates in the delay files 
customisedDelaysOther["Jny Dt Rw"] =pd.to_datetime(customisedDelaysOther["Jny Dt Rw"])
customisedDelaysSevere["Jny Dt Rw"] =pd.to_datetime(customisedDelaysSevere["Jny Dt Rw"])
customisedDelaysVW["Jny Dt Rw"] =pd.to_datetime(customisedDelaysVW["Jny Dt Rw"])

#combine all the weather station data 
combinedWeatherStationData = pd.concat([CG17O, CGJ6, CG17P, DJH, WCM1, ULR, CG17S], axis=0)
#save combined data in a csv file 
combinedWeatherStationData.to_csv(weatherDataPath)

#combine all the delay data
combinedDelayData = pd.concat([customisedDelaysSevere, customisedDelaysVW, customisedDelaysOther], axis=0)
#rename some of the columns such as journey date or budgle profit center code to make them easier to interpter 
combinedDelayData.rename(columns={'Jny Dt Rw' : 'JourneyDate', 'Bgl Prft Cent': 'LineCode', 'Hcde': 'IndividualTrainCode'}, inplace=True)
#sort the delay data by the journey dates 
combinedDelayData = combinedDelayData.sort_values(by = 'JourneyDate')
#turn all the memo notes upper case to make filtering easier later 
combinedDelayData['Memo'] = combinedDelayData['Memo'].str.upper()

#save combined delay data in a csv file 
combinedDelayData.to_csv(combinedDelayDataPath)

#read in the combined weather csv file 
weatherDf = pd.read_csv(weatherDataPath)
#read in the combined delay csv file 
delayDf = pd.read_csv(combinedDelayDataPath) 

 #turn all the journey dates from string format to date format 
delayDf['JourneyDate'] = pd.to_datetime(delayDf['JourneyDate'])  
#filter out journeys with speed restrictions based on the memo notes. Terms such as 'blanket' 'restriction' and 'ESR' were used 
blanketRestrictionDf = delayDf[delayDf['Memo'].str.contains("BLANKET") | delayDf['Memo'].str.contains("RESTRICTION") | delayDf['Memo'].str.contains("ESR")] 
#filter out journeys with no speed restrictions based on the memo notes.
noRestrictionDf = delayDf[~delayDf['Memo'].str.contains("BLANKET") & ~delayDf['Memo'].str.contains("RESTRICTION") & ~delayDf['Memo'].str.contains("ESR")]

#save filtered out journeys with speed restriction to a csv file 
blanketRestrictionDf.to_csv('blanket_restriction_data.csv')

#################### Structure of the Graph generating function and user guide #############
# journeyName (String) - Graph Title
# lineToExamine (Integer) - The specific line to examine by numerical code, if examining all lines set this to 0
# startDate - The date from which to examine rows in the dataset
# endDate - The date in which to end the examination of rows in the dataset
####### If examining entire dataset, leave startDate and endDate empty ==> ''
##### ##### ##### ##### ######
##### ##### Example Inputs: ##### #####
#    #journeyName = 'Euston to Edinburgh and Glasgow'
#    #lineToExamine = 22114000
#    #startDate = '2021-10-01'
#    #endDate = '2021-10-31'
##### ##### ##### ##### #####
##### ##### ##### ##### #####
#    #journeyName = 'Euston to Edinburgh and Glasgow'
#    #lineToExamine = 22114000
#    #startDate = ''
#    #endDate = ''
#for reference:
# 22114000 -> Euston to Edinburgh & Glasgow
# 23121000 -> Euston to Glasgow
##### ##### ##### ##### #####
##### ##### ##### ##### #####
#creating function to generate graph data to answer question 2 . This function was created to make the graph generation easier for multiple 
#different journeys and dates 
def generateGraphData(journeyName, lineToExamine, startDate, endDate):
    print('[Log]: Running generateGraphData()') 
    print('[Log]: Parameters - ' + journeyName + ' ' + lineToExamine + ' ' + startDate + ' ' + endDate)
    #printing titles of the graph with the line examined, dates and the journey name
#if line to examine is set to 0 
    if (lineToExamine == 0):
        journeyName = 'ALL JOURNIES' #set journeyName to be All Journeys 
        lineToExamine = 'ALL LINES' # set line to examine to be All lines 
        EtgRes = blanketRestrictionDf
        EtgNoRes = noRestrictionDf
    else: #otherwise filter dataframes with blanket and no blanket restrictions based on the line code
        EtgRes = blanketRestrictionDf[blanketRestrictionDf['LineCode'] == lineToExamine]
        EtgNoRes = noRestrictionDf[noRestrictionDf['LineCode'] == lineToExamine]

#if start data and end date is set 
    if (startDate != '' and endDate != ''):
        titleStr = ' (between ' + str(startDate) + ' & ' + str(endDate) + ')' #add dates in title 
         #filter dataframe based on journey date 
        restrictionRange = EtgRes[EtgRes["JourneyDate"].isin(pd.date_range(startDate, endDate))]
         #filter dataframe based on journey date 
        noRestrictionRange = EtgNoRes[EtgNoRes["JourneyDate"].isin(pd.date_range(startDate, endDate))]
    #if dates are not set 
    else:
        titleStr = ' (all dates)' #add all dates to title 
        restrictionRange = EtgRes  #include all dates from df
        noRestrictionRange = EtgNoRes #include all dates from df
#plotting boxplot 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')' + titleStr)
    axes[0].boxplot(restrictionRange['Dly Mins']) #box plot for delays with restriction
    axes[0].set_title('Restricted Speed Delay (Minutes)') #set title
    axes[0].set_xlabel('Blanket Restrictions') #set x label 
    axes[0].set_ylabel('Minutes') #set y label
    axes[1].boxplot(noRestrictionRange['Dly Mins']) #box plot for delays with restriction
    axes[1].set_title('No Restrictions Speed Delay (Minutes)') #set title
    axes[1].set_xlabel('No Blanket Restrictions') #set x lable 
    axes[1].set_ylabel('Minutes') #set y label
    fig.tight_layout()
    plt.show()
#plotting histogram 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')' + titleStr)
    axes[0].hist(restrictionRange['Dly Mins']) #histogram for delays with restriction
    axes[0].set_title('Restricted Speed Delay Frequency') #set title 
    axes[0].set_xlabel('Frequency')  #set x lable 
    axes[0].set_ylabel('Minutes') #set y label
    axes[1].hist(noRestrictionRange['Dly Mins']) #histogram for delays with no restriction
    axes[1].set_title('No Restrictions Speed Delay Frequency') #set title 
    axes[1].set_xlabel('Frequency')  #set x lable 
    axes[1].set_ylabel('Minutes') #set y label
    fig.tight_layout()
    plt.show()
#plotting scatterplot 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')' + titleStr)
    axes[0].plot(restrictionRange['JourneyDate'].dt.strftime('%m-%d'), restrictionRange['Dly Mins'], 'ro')
    axes[0].set_title('Restricted Speed Delay Frequency') #set title 
    axes[0].set_xlabel('Date') #set x lable 
    axes[0].set_ylabel('Minutes') #set y label
    axes[1].plot(noRestrictionRange['JourneyDate'].dt.strftime('%m-%d'), noRestrictionRange['Dly Mins'], 'bo')
    axes[1].set_title('No Restrictions Speed Delay Frequency') #set title 
    axes[1].set_xlabel('Date') #set x lable 
    axes[1].set_ylabel('Minutes') #set y label
    fig.tight_layout()
    plt.show()

    # summarize
    #calculate and print the mean, standard deviation and median for journeys with speed restriction 
    print('Blanket Restrictions    : mean=%.3f stdv=%.3f median=%.3f' % (np.mean(restrictionRange['Dly Mins']), np.std(restrictionRange['Dly Mins']), np.median(restrictionRange['Dly Mins'])))
    #calculate and print the mean, standard deviation and median for journeys with no speed restriction 
    print('No Blanket Restrictions : mean=%.3f stdv=%.3f median=%.3f' % (np.mean(noRestrictionRange['Dly Mins']), np.std(noRestrictionRange['Dly Mins']), np.median(noRestrictionRange['Dly Mins'])))
    print('[Log]: End of compareTwoDateRanges()')


# Question 3: What was the benefit of emergency timetable work (27-30 10-2021) in response to blanket speed restrictions
# compareTwoDateRanges('ALL JOURNIES', 0, '2021-10-27', '2021-10-30', '2020-12-18', '2020-12-21')
#defining function to generate graphs to answer question 3
def compareTwoDateRanges(journeyName, lineToExamine, restrictedStartDate, restrictedEndDate, unrestrictedStartDate, unrestrictedEndDate):
    print('[Log]: Running compareTwoDateRanges()')
    print('[Log]: Parameters - ' + journeyName + ' ' + lineToExamine + ' ' + restrictedStartDate + ' ' + restrictedEndDate + ' ' + unrestrictedStartDate + ' ' + unrestrictedEndDate)
    
    #if line to examine is set to 0 then set to 'ALL LINES'
    if (lineToExamine == 0):
        lineToExamine = 'ALL LINES'

    firstDateRangeDf = blanketRestrictionDf[blanketRestrictionDf["JourneyDate"].isin(pd.date_range(restrictedStartDate, restrictedEndDate))]
    secondDateRangeDf = blanketRestrictionDf[blanketRestrictionDf["JourneyDate"].isin(pd.date_range(unrestrictedStartDate, unrestrictedEndDate))]
#plotting boxplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')')
    axes[0].boxplot(firstDateRangeDf['Dly Mins'])
    axes[0].set_title('Speed Delay (' + restrictedStartDate + '-' + restrictedEndDate + ')' )
    axes[0].set_ylabel('Minutes') #setting y label 
    axes[1].boxplot(secondDateRangeDf['Dly Mins'])
    axes[1].set_title('Speed Delay (' + unrestrictedStartDate + '-' + unrestrictedEndDate + ')' )
    axes[1].set_ylabel('Minutes') #setting y label 
    fig.tight_layout()
    plt.show()

#plotting histogram 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')')
    axes[0].hist(firstDateRangeDf['Dly Mins'])
    axes[0].set_title('Speed Delay (' + restrictedStartDate + '-' + restrictedEndDate + ')' )
    axes[0].set_xlabel('Frequency') #set x lable 
    axes[0].set_ylabel('Minutes') #setting y label 
    axes[1].hist(secondDateRangeDf['Dly Mins'])
    axes[1].set_title('Speed Delay (' + unrestrictedStartDate + '-' + unrestrictedEndDate + ')' )
    axes[1].set_xlabel('Frequency') #set x lable 
    axes[1].set_ylabel('Minutes') #setting y label 
    fig.tight_layout()
    plt.show()

#plotting scatterplot 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')')
    axes[0].plot(firstDateRangeDf['JourneyDate'].dt.strftime('%m-%d'), firstDateRangeDf['Dly Mins'], 'ro')
    axes[0].set_title('Speed Delay (' + restrictedStartDate + '-' + restrictedEndDate + ')' )
    axes[0].set_xlabel('Date') #set x lable 
    axes[0].set_ylabel('Minutes') #setting y label 
    axes[1].plot(secondDateRangeDf['JourneyDate'].dt.strftime('%m-%d'), secondDateRangeDf['Dly Mins'], 'bo')
    axes[1].set_title('Speed Delay (' + unrestrictedStartDate + '-' + unrestrictedEndDate + ')' )
    axes[1].set_xlabel('Date') #set x lable 
    axes[1].set_ylabel('Minutes') #setting y label 
    fig.tight_layout()
    plt.show()

 # summarize
    #calculate and print the mean, standard deviation and median for journeys with speed restriction 
    print('FIRST DATE RANGE  Df    : mean=%.3f stdv=%.3f median=%.3f' % (np.mean(firstDateRangeDf['Dly Mins']), np.std(firstDateRangeDf['Dly Mins']), np.median(firstDateRangeDf['Dly Mins'])))
    print('SECOND DATE RANGE Df    : mean=%.3f stdv=%.3f median=%.3f' % (np.mean(secondDateRangeDf['Dly Mins']), np.std(secondDateRangeDf['Dly Mins']), np.median(secondDateRangeDf['Dly Mins'])))
    print('[Log]: End of compareTwoDateRanges()')

# defining function to generate graphs to answer question 1 
def weatherPerformance(journeyName = 'All Journies', lineToExamine = 0, startDate = '', endDate = ''):
    print('[Log]: Running weatherPerformance()')
    badWeatherDf = delayDf[delayDf['Memo'].str.contains("FLOOD") | delayDf['Memo'].str.contains("WEATHER")] 
    stdDf = delayDf[~delayDf['Memo'].str.contains("FLOOD") & ~delayDf['Memo'].str.contains("WEATHER")] 
    
    if (lineToExamine != 0):
        journeyName = journeyName
        lineToExamine = str(lineToExamine)
        badWeatherDf = badWeatherDf[badWeatherDf['LineCode'] == lineToExamine]
        stdDf = stdDf[stdDf['LineCode'] == lineToExamine]


    if (startDate != '' and endDate != ''):
        titleStr = ' (between ' + str(startDate) + ' & ' + str(endDate) + ')'
        badWeatherDf = badWeatherDf[badWeatherDf["JourneyDate"].isin(pd.date_range(startDate, endDate))]
        stdDf = stdDf[stdDf["JourneyDate"].isin(pd.date_range(startDate, endDate))]
    else:
        titleStr = ' (all dates)'

#plotting boxplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')' + titleStr)
    axes[0].boxplot(badWeatherDf['Dly Mins'])
    axes[0].set_title('Bad Weather Delay (Mins)')
    #axes[0].set_xlabel('Blanket Restrictions')
    axes[0].set_ylabel('Minutes') #setting y label 
    axes[1].boxplot(stdDf['Dly Mins'])
    axes[1].set_title('Other Speed Delay (Mins)')
    #axes[1].set_xlabel('No Blanket Restrictions')
    axes[1].set_ylabel('Minutes') #setting y label 
    fig.tight_layout()
    plt.show()

#plotting histogram 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')' + titleStr)
    axes[0].hist(badWeatherDf['Dly Mins'])
    axes[0].set_title('Bad Weather Delay Frequency')
    axes[0].set_xlabel('Frequency') #set x lable 
    axes[0].set_ylabel('Minutes') #setting y label 
    axes[1].hist(stdDf['Dly Mins'])
    axes[1].set_title('Other Delay Frequency')
    axes[1].set_xlabel('Frequency') #set x lable 
    axes[1].set_ylabel('Minutes') #setting y label 
    fig.tight_layout()
    plt.show()

#plotting scatterplot 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) #setting size 
    fig.suptitle('Delays for ' + str(lineToExamine) + ' (' + journeyName + ')' + titleStr)
    axes[0].plot(badWeatherDf['Dly Mins'], 'ro')
    axes[0].set_title('Bad Weather Delay Frequency') #set title 
    #axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Minutes') #setting y label 
    axes[1].plot(stdDf['Dly Mins'], 'bo')
    axes[1].set_title('Other Delay Frequency') #set title 
    #axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Minutes') #setting y label 
    fig.tight_layout()
    plt.show()
 
 # summarize
    #calculate and print the mean, standard deviation and median for journeys with speed restriction 
    print('Bad Weather   : mean=%.3f stdv=%.3f median=%.3f' % (np.mean(badWeatherDf['Dly Mins']), np.std(badWeatherDf['Dly Mins']), np.median(badWeatherDf['Dly Mins'])))
    print('Other         : mean=%.3f stdv=%.3f median=%.3f' % (np.mean(stdDf['Dly Mins']), np.std(stdDf['Dly Mins']), np.median(stdDf['Dly Mins'])))
    print('[Log]: End of weatherPerformance()')

 # #### generating different graphs 
#    generateGraphData('Euston to Edinburgh & Glasgow', 22114000, '2021-10-01', '2021-10-31')
#    generateGraphData('Euston to Glasgow', 23121000, '2021-10-01', '2021-10-31')
#    generateGraphData('Euston to Edinburgh & Glasgow', 22114000, '', '')
#    generateGraphData('Euston to Glasgow', 23121000, '', '')
#    generateGraphData('All Journies', 0, '', '')
#    generateGraphData('All Journies', 0, '2021-10-01', '2021-10-31')

    # Benefit of the works vs other timeframes
#    compareTwoDateRanges('ALL JOURNIES', 0, '2021-10-27', '2021-10-30', '2020-12-20', '2020-12-23')
#    compareTwoDateRanges('Euston to Edinburgh & Glasgow', 22114000, '2021-10-27', '2021-10-30', '2020-12-20', '2020-12-23')
#    compareTwoDateRanges('Euston to Glasgow', 23121000, '2021-10-27', '2021-10-30', '2020-12-20', '2020-12-23')

    # Relationship between extreme weather and railway performance
        # Extreme weather tends to be worse than any other type of delay.
#    weatherPerformance(startDate='2021-10-01', endDate='2021-10-31')
#    weatherPerformance(startDate='2020-12-20', endDate='2020-12-23')

def weatherPrediction():
    weatherStationDf = weatherDf
    #turn string into date format 
    weatherStationDf[' Data Date Time'] = pd.to_datetime(weatherStationDf[' Data Date Time'])

    weatherStationDf = weatherStationDf.set_index(' Data Date Time')
    #resample the data to make times no restriction and restriction times comparable 
    weatherStationDfAvgDay = weatherStationDf.resample('H').mean()
    weatherStationDfAvgDay.index = weatherStationDfAvgDay.index.strftime('%Y-%m-%d')
    #turning string format into date format
    weatherStationDfAvgDay.index = pd.to_datetime(weatherStationDfAvgDay.index).date
    
    #print(min(weatherStationDfAvgDay.index))
    #print(max(weatherStationDfAvgDay.index))

    journeyDelayDf = delayDf

    
    journeyDelayDf['JourneyDate'] = journeyDelayDf['JourneyDate'].dt.strftime('%Y-%m-%d')
    #turn string format into date format
    journeyDelayDf['JourneyDate'] = pd.to_datetime(journeyDelayDf['JourneyDate']).dt.date


    
    listOfDates = weatherStationDfAvgDay.index.tolist()
# test if journey date is between the dates 
    journeyDelayDf = journeyDelayDf[np.in1d(journeyDelayDf['JourneyDate'], listOfDates)]
    journeyDelayDf['UniformDates'] = journeyDelayDf['JourneyDate']

#seperate the data by periods of speed restrictions and without speed restrictions 
    speedRestrictions = journeyDelayDf[journeyDelayDf['Memo'].str.contains("BLANKET") | journeyDelayDf['Memo'].str.contains("RESTRICTION") | journeyDelayDf['Memo'].str.contains("ESR")] 
    nospeedRestrictions = journeyDelayDf[~journeyDelayDf['Memo'].str.contains("BLANKET") & ~journeyDelayDf['Memo'].str.contains("RESTRICTION") & ~journeyDelayDf['Memo'].str.contains("ESR")]
#turnning restrictions into binray values 
#if restricted then 1
    speedRestrictions['Restricted'] = 1
#if not restricted then 0 
    nospeedRestrictions['Restricted'] = 0
#turn string format into date format
    speedRestrictions['JourneyDate'] = pd.to_datetime(speedRestrictions['JourneyDate'])
    speedRestrictions = speedRestrictions.set_index('JourneyDate')
    #resample the data 
    speedRestrictions = speedRestrictions.resample('H').mean()
    speedRestrictions = speedRestrictions['Restricted'].dropna()

    
    speedRestrictionsWeather = weatherStationDfAvgDay.dropna()
    speedRestrictionsWeather = speedRestrictionsWeather[np.in1d(speedRestrictionsWeather.index, speedRestrictions.index.tolist())]
    speedRestrictionsWeather['Restricted'] = 1
#turn string format into date format
    nospeedRestrictions['JourneyDate'] = pd.to_datetime(nospeedRestrictions['JourneyDate'])
    nospeedRestrictions = nospeedRestrictions.set_index('JourneyDate')
    #resample the data  
    nospeedRestrictions = nospeedRestrictions.resample('H').mean()
    nospeedRestrictions = nospeedRestrictions['Restricted'].dropna()

    
    nospeedRestrictionsWeather = weatherStationDfAvgDay.dropna()
    nospeedRestrictionsWeather = nospeedRestrictionsWeather[np.in1d(nospeedRestrictionsWeather.index, nospeedRestrictions.index.tolist())]
    nospeedRestrictionsWeather['Restricted'] = 0

    nospeedRestrictionsWeather = nospeedRestrictionsWeather[np.in1d(nospeedRestrictionsWeather.index, speedRestrictions.index.tolist()) == False]
#combine speed restrected and non restrected data as a training set 
    trainingData = pd.concat([nospeedRestrictionsWeather, speedRestrictionsWeather])
#creating data (y data) including class lables (restrected or not)
    y = trainingData['Restricted']
# creating data expluding class lables (x data )
    x = trainingData.loc[:, trainingData.columns != 'Restricted']
# splitting x and y data into train and test groups 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=29956597, shuffle=True)
#import logistic regression from sklearn 
    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression()
#training the model on x and y data 
    logisticRegr.fit(x_train, y_train)
#use the model to predict class on test data 
    predictions = logisticRegr.predict(x_test)
#compare how well the model predict 
    score = logisticRegr.score(x_test, y_test)
#print score 
    print(score)
#print the predictions 
    print('logr preds')
    print(predictions)
#print the actual values 
    print('actual')
    print(y_test)

    print('length of dataset ' + str(len(trainingData)))

#creating nerual network - Keras sequential model 
def kerasmodal():
    model = Sequential()
    #add layers by passing a list of layers to the sequential constructor
    #activation type is sigmoid sigmoid(x) = 1 / (1 + exp(-x)
    #Sigmoid logistic function outputs values in range (0,1)
    model.add(layers.Dense(128,activation='sigmoid'))
    #droput layer to help overfitting  
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16,activation='sigmoid'))
    model.add(layers.Dense(1,activation='sigmoid'))
#configurate model for training 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def kerasWeatherPredictions(resampleRate, epochs):
    weatherStationDf = weatherDf
    weatherStationDf[' Data Date Time'] = pd.to_datetime(weatherStationDf[' Data Date Time'])

    weatherStationDf = weatherStationDf.set_index(' Data Date Time')
    #resample the data
    weatherStationDfAvgDay = weatherStationDf.resample(resampleRate).mean()
    weatherStationDfAvgDay.index = weatherStationDfAvgDay.index.strftime('%Y-%m-%d')
    #turn string into date formate 
    weatherStationDfAvgDay.index = pd.to_datetime(weatherStationDfAvgDay.index).date

    journeyDelayDf = delayDf

    journeyDelayDf['JourneyDate'] = journeyDelayDf['JourneyDate'].dt.strftime('%Y-%m-%d')
    #turn string format into date format 
    journeyDelayDf['JourneyDate'] = pd.to_datetime(journeyDelayDf['JourneyDate']).dt.date


    
    listOfDates = weatherStationDfAvgDay.index.tolist()

    journeyDelayDf = journeyDelayDf[np.in1d(journeyDelayDf['JourneyDate'], listOfDates)]
    journeyDelayDf['UniformDates'] = journeyDelayDf['JourneyDate']
#seperating data based on the memo to speed restricted and non restricted 
    speedRestrictions = journeyDelayDf[journeyDelayDf['Memo'].str.contains("BLANKET") | journeyDelayDf['Memo'].str.contains("RESTRICTION") | journeyDelayDf['Memo'].str.contains("ESR")] 
    nospeedRestrictions = journeyDelayDf[~journeyDelayDf['Memo'].str.contains("BLANKET") & ~journeyDelayDf['Memo'].str.contains("RESTRICTION") & ~journeyDelayDf['Memo'].str.contains("ESR")]
#turnning restrictions into binray values 
#if restricted then 1
    speedRestrictions['Restricted'] = 1
#if not restricted then 0
    nospeedRestrictions['Restricted'] = 0
#turn string format into date format
    speedRestrictions['JourneyDate'] = pd.to_datetime(speedRestrictions['JourneyDate'])
    speedRestrictions = speedRestrictions.set_index('JourneyDate')
    #resample the data 
    speedRestrictions = speedRestrictions.resample(resampleRate).mean()
    speedRestrictions = speedRestrictions['Restricted'].dropna()

    
    speedRestrictionsWeather = weatherStationDfAvgDay.dropna()
    speedRestrictionsWeather = speedRestrictionsWeather[np.in1d(speedRestrictionsWeather.index, speedRestrictions.index.tolist())]
    speedRestrictionsWeather['Restricted'] = 1
#turn string format into date format
    nospeedRestrictions['JourneyDate'] = pd.to_datetime(nospeedRestrictions['JourneyDate'])
    nospeedRestrictions = nospeedRestrictions.set_index('JourneyDate')
    #resample the data 
    nospeedRestrictions = nospeedRestrictions.resample(resampleRate).mean()
    nospeedRestrictions = nospeedRestrictions['Restricted'].dropna()

    
    nospeedRestrictionsWeather = weatherStationDfAvgDay.dropna()
    nospeedRestrictionsWeather = nospeedRestrictionsWeather[np.in1d(nospeedRestrictionsWeather.index, nospeedRestrictions.index.tolist())]
    nospeedRestrictionsWeather['Restricted'] = 0

    nospeedRestrictionsWeather = nospeedRestrictionsWeather[np.in1d(nospeedRestrictionsWeather.index, speedRestrictions.index.tolist()) == False]
#combine speed restricted and non restircted data for training set 
    trainingData = pd.concat([nospeedRestrictionsWeather, speedRestrictionsWeather])

    trainingData = trainingData.drop('Unnamed: 0', 1)
#creating data (y data) including class lables (restrected or not)
    y = trainingData['Restricted']
# creating data expluding class lables (x data)
    x = trainingData.loc[:, trainingData.columns != 'Restricted']
# splitting x and y data into train and test groups 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29956597, shuffle=True)


    from sklearn.linear_model import LogisticRegression
    #logisticRegr = LogisticRegression()
    clf = KerasRegressor(build_fn=kerasmodal, 
                                 epochs=epochs, 
                                 batch_size=1, 
                                 verbose=0)

#import pipeline 

    from imblearn.pipeline import Pipeline
#import SMOTE
    from imblearn.over_sampling import SMOTE
#import Random Under Sampler
    from imblearn.under_sampling import RandomUnderSampler
#import Kfold
    from sklearn.model_selection import KFold
#import standard scaler 
    from sklearn.preprocessing import StandardScaler
#pipeline of transforms and resamples with final estimator 
#avoids leaking test set into train set 
    pipeline = Pipeline(steps=[('scale', StandardScaler()), ('oversample', SMOTE()), ('undersample', RandomUnderSampler()), ('model', clf)])
 #set number of splits of kfold 
    k_fold = KFold(n_splits=2)
    #fit the model 
    [pipeline.fit(x_train, y_train).score(x_test, y_test) for train, test in k_fold.split(x)]
    #Transform the data, and apply predict with the final estimator.
    predictions = pipeline.predict(x_test)
    
    #score = pipeline.score(x_test, y_test)
    #print(score)
    bounded_preds = [ round(elem) for elem in predictions ] # score based on threshold
  #print predictions 
    print('keras preds')
    print(bounded_preds)
#print actual values 
    print('actual')
    print(y_test)

    #expects
    # Wind Speed, Wind Gust, Air Temperature, Air Pressure, Relative Humidity, Track Temperature, Precipitation Intensity, Precipitation Quantity, Rainfall In Last Hour, Rainfall in Last 24 hours
    #[28,53,19,9,1000,84,10,0.22,0.03,0.87]
    samplePrediction = pd.DataFrame([[28,53,19,9,1000,84,10,0.22,0.03,0.87]])

    kerasPrediction = pipeline.predict(samplePrediction)
    # if samplePrediction has more than one sample, ensure all preds are bounded between 0 and 1... this way we treat continuous output as discrete values
    #boundedPrediction = [ round(elem) for elem in kerasPrediction ]
    print()
    print('Based on inputted data... \n \n (1 restrictions should be in place with this weather \n 0 restrictions should not be in place) \n')
    print(np.round(kerasPrediction))

# Examples...
def createExamples():
    print('[Log]: Running createExamples()')
   
#    weatherPerformance()

#    weatherPrediction()

    kerasWeatherPredictions('H', 25) # resample by 'H', 'D', 'M' ... hourly provides the most data. 

    print('[Log]: End of createExamples()')

createExamples()