# CNN Sentiment Analysis
Final project for Data Analysis Methods class. Used web scraping on data from CNN archives to get sentiment on whether news is more positive, negative or neutral for Democrats and Republicans.

# Final Project Report
By Shan Santhakumar, Kenneth Salce, John-Paul McDonald, Deepanjali Halaharivi, Shaan Pakala
Project Proposal
For our project, our group would like to perform a sentiment analysis of political news articles related to the two major US parties, Democrat and Republican. To be specific, we will develop a general political article sentiment analysis model using linear regression, regardless of party. Then we will apply it to articles covering Democrats and articles covering Republicans separately in order to observe how the news source talks about either party separately. Then we will compare the Republican and Democrat results at the end. For the news site we will gather information from, we chose CNN as it is a popular news source with easily accessible data, which frequently covers political topics. We will perform a sentiment analysis on recent articles from CNN and see if the sentiment of certain political parties as of late has been more positive or negative. In order to do this, we will perform text mining on CNN's database of past news articles and collect data consisting of the first two paragraphs of all news articles from the months January and February of 2023. From our dataset of first and second paragraphs from all news CNN released covering the Democratic or Republican Party, we will analyze each paragraph and give it a rating of either positive, negative, or neutral based on the contents of each paragraph. Using a database of words associated with positive political articles and a database for words associated with negative political articles, we can find the frequency of these keywords in the articles. Using the frequency we will build a linear regression model to predict the sentiment of the article, using the number of positive and negative words as the variables. This sentiment should come as a negative number if the sentiment is negative, and a positive number if it’s positive, and a number close to 0 if it is neutral. The only issue is what number to start classifying the articles as neutral, so we will have to play around with some thresholds for that. In order to provide data to test our model on, we will manually identify the sentiment of articles by reading through them and use that data to train and test our model. 


# Our Questions
Does CNN’s articles talk positively or negatively about Democrats?  
Does CNN’s articles talk positively or negatively about Republicans?



# Data Collection

The first challenge was actually gathering data for CNN’s news articles, since we could not simply go online and find a CSV for CNN’s news articles. So we used BeautifulSoup to go onto CNN’s website and scrape all the articles we needed. More specifically, it will use the link for all the articles of the month and then go into each article in that month in order to gather the first two paragraphs for us to analyze. 
Since our program is only going to be analyzing the sentiment of Democratic and Republican articles, we needed to look through the headlines to determine what the article was related to and only scrape the ones that are pertinent to our analysis. To do this, we had to develop a word bank of Democratic and Republican Party keywords. This was pretty much the same format as web scraping CNN’s articles, except we went online and found data on the current Republican and Democratic office holders in the United States (Governors, Senators, and Representatives). We then compiled these into Democrat and Republican word banks. We also included some other keywords such as “Democrat” and "Republican” and “Biden” and “Trump.” After doing this, we had to manually go through each word bank and remove some politicians with generic names such as “Johnson” or “Ryan.” One politician that really threw us off at first was a politician named “Justice,” since the Department of Justice is talked about often in CNN’s articles. 
	After refining the word banks, we could put the web scraping application to use and gather all of the articles from CNN’s website. In this process, the first two paragraphs of each article related to either a Democrat or Republican is scraped and written into a txt file for later analysis. We also included if the article was talking about a Republican or Democrat in the txt file for convenience when we analyze. In order to alleviate future confusion in our sentiment analysis, we disregarded articles that were talking about both Republicans and Democrats. This was necessary as an article could be a positive article for Democrats but a negative article for Republicans, which would thoroughly confuse our sentiment analysis. 

# Tokenization and Data Cleaning

Now that we have all of our data stored in a txt file, it is time to start our text analysis. First thing we need to do is separate this large corpus into the individual articles we can analyze separately. This was rather simple because when writing the articles into the text file, we separated each article by an indent followed by a newline character, so a simple .split(‘     \n’) did the trick. Now we have an array of all of the articles. We navigated this array with a for loop to look at each article. We then split each article further into a temporary array of words (tokens) with .split(). We previously tagged each article “r” for Republican and “d” for Democratic article at the end, so now we can see which type of article this is and store it. Then we remove this character from the array of words so it doesn’t interfere with our analysis. We converted this array of words into a temporary dataframe and dropped the duplicates in order to resemble a one-hot encoding vector of the words. From here it was also easy to remove all the punctuation and make all the words lowercase because we do not want to be case sensitive. 
Now we have a temporary dataframe with all the unique words (cleaned) in the article we are currently looking at (and we are iterating through each article with a for loop).

# Positive and Negative Political Word Banks

In order to determine the number of “positive” and “negative” words in each article to base our sentiment analysis linear regression model on, we had to first determine what was a “positive” and “negative” political word. This was done by generating large word banks for each to compare each word to. This word bank was generated by several repetitions of asking ChatGPT for a list of 600 common “positive” or “negative” words used in political articles. After each repetition, we looked through the list ourselves and removed words we thought were too general or would not apply to our analysis. Then we added this list to a txt file along with the lists gathered from other repetitions. At the end we used a program to remove all the duplicates and make them all lowercase in the txt file. Now we have our word banks.

# Counting the Number of Positive and Negative Words

Now we have to work on building a linear regression model in order to try to predict the sentiment of the political articles based on the number of positive and negative political words. At this point we have our word banks, and we are iterating through every article and looking at a dataframe of all of the unique words in the article. From here we just count the number of positive words and negative words and store it in a dataframe alongside the article text and the political party. In order to count the number of positive and negative words, we just go through our temporary data frame of all the unique words in the article and add one to the positive word count if the word is in the dpositive political word bank, and same for the negative words. By the end we will have a data frame full of every single article, with the number of positive political words, the number of negative political words, and the political party. Since they are counting the unique words in the data frame, if the article says the same negative word twice, it will still only count as one. This is to ensure that no one word in our word bank will affect the sentiment too much if repeated in the same sentence. This way if any specific word in our word banks are used out of context or unusually frequently, it will not affect the results as much since it will only count once instead of all the times it is used.
 

# EDA

*Note: True Sentiment is 1 for positive overall sentiment, 0 for neutral, and -1 for negative.
The True Sentiments were filled out manually by us based on the observed sentiment of the article. The color represents the frequency of each observation.



When looking solely at the graphs of the number of positive and negative political words versus the true sentiment of the article, it is hard to observe a concrete linear pattern that our linear regression model would require. However, when we plot the difference between the number of positive words and negative words, we start to see something we like.


	Difference = # of positive words - # of negative words

While the linear trend is not perfect, it does seem to be there.


# Building the Linear Regression Model


Since our linear regression model should give a positive number if it is a positive sentiment, negative number if it is a negative sentiment, and a number close to 0 if it is neutral, we have to determine how close to 0 to consider neutral. This is what the thresholds on line 17 are for. The sentiment has to reach above 0.3 in order to be considered positive, and has to reach below -0.2 to be considered negative. After experimentation, these are the thresholds that gave us the most accurate results with our testing data (73% correct classifications of articles).
On line 19 we have the linear regression model coefficients and intercept. At first we wanted to build a linear model with no intercept since we thought an article with 0 positive and 0 negative words would be 0 (neutral), but this failed to give us higher than 55% correct classifications in our testing data. As a result, we decided to include the intercept and saw our results improve significantly to 73%. Then line 20 is just filling in our sentiment predictions for all of the articles we have using the linear regression model we just built.

# Conclusions

Using our finished linear regression sentiment analysis model, we can observe the results from CNN’s articles.

From the above stacked bar chart we can see that most news for Democrats and Republicans in CNN is more negative. The next most common type of news for both parties is neutral, and the least frequent type of news is positive news. We can observe that CNN’s articles of Republicans seem to be slightly more negative; however, it is pretty close and our sentiment analysis model is not the most accurate so we cannot really draw any conclusions there.

# Room for Improvement

Our linear regression model accuracy relies heavily on the quality of our positive and negative political word banks, so finding a more legitimate word bank compiled from more in depth analysis of news articles might make our results better. Our word bank does cover a lot of the more common words used in political articles, so it is usable.
This model also does mainly rely on the articles we pick and cannot accurately assess anything about CNN or other media output as a whole.
A linear regression model is not ideal for classifying the sentiment of articles into positive and negative; however, it did yield us a decent accuracy of 73%.


# Member Contributions


Shan Santhakumar- Made graphs for EDA and conclusion. 
Kenneth Salce- Worked on linear regression model training. 
John-Paul McDonald- Worked on report and slides, helped with web scraping. 
Deepanjali Halaharivi- Worked on report and slides, helped with EDA. 
Shaan Pakala- Gathered data by web scraping, helped with linear regression model. 


All members contributed equally.
