## Project title
Sentiment Analysis on the legal corpus data
## Motivation
There were large amount of data available at the corpus database, hence I thought it would be fetch good results it sentiment prediction is performed on the data. 
## Tech/framework used
<b>Built with</b>
- python 3.6
- juypter notebook
- Ntlk
- textblob
- word2vec
## Features
This program will enable to calculate the sentiment of the sentences given to the model.  
## Code Example
Libraries Used:
1. Pandas 
2. Numpy
3. lXML
4. NLTP
5. Textblob
6. word2vec
## Installation
Provide step by step series of examples and explanations about how to get a development env running.
 - [ ] Install LXML using pip install
 - [ ] install NLTP library
	 - [ ] download all the nltp corpus using:
			 - [1]import nltp
			 - [2]nltp.download() 
- [ ] Install Textblob for sentiment analysis using pip install 
- [ ] Install word2vec using pip install gensim  
- [ ] Change the source file address in the read_csv command.  
 ## Tests
All the test data should be tokenised and should be free of stop words, puntuations, lower case and should be loaded in the program and then converted into the required statement. 

## How to use?
1.  We have to load the data from the XML into the data frame using the functino folderloader. 
2. Clean the data using 
3. Calcualate the sentiement value for the data
4. Vectorize the model using the word2vec 
5. Create the feature matrix using 
6. Train the random forest classifier and test for the external data 

*Author:* 
[Abhiram Maddali]