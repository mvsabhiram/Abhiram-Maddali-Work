## Project title
Dengue Fever Prediction based on Time series and RNN analysis

## Motivation
This is method was to see how to implement RNN/LSTM on the time series data and get the predictions for instances of dengue fever. 
## Tech/framework used
<b>Built with</b>
- python 3.6
- juypter notebook

## Features
This program uses LSTM for the calculating the predition and gives the results based on the previous instances of the data. 

## Libraries Used
Libraries Used:
1. Pandas 
2. Numpy
3. Matlibplot
4. Scalar
5. MinMaxScalar
6. Keras 

## Installation
Provide step by step series of examples and explanations about how to get a development env running.
 - [ ] pip install keras 
 - [ ] Check the version of keras for 0.61
 - [ ] Check if keras is running with tensor flow backend
	 - [ ] If Keras not running and throws an error
		 - [ ] Install Tensor Flow 
        
		  
## Tests
In this program testing is done using the data for 2008 data for city San Juan and 2009 for Iquitos city . 

## How to use?
1. Load the dataset into the model using read_csv function. 
2. Select the required data for the analysis by see the count of data. I have selected 1991-2005 for San Juan city and 2002-2008 for Iquitor city 
3. Clean the data by removing the null values and adding the lable column 
4. Convert all the values to scalar using minmaxscalar() funtion
5. Convert the dataset into supervised learning using series_to_supervised function
6. reshape the dataset to feed into the LSTM model 
7. Define the LSTM classifier using the sequential and
8. train and validate the data to calculate the RMSE value.

**Author**
[Abhiram Maddali]

