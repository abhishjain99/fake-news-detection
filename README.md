# fake-news-detection
<!-- #### Fake News--
Tired of online rumors masquerading as truth?</b>
We've all seen them - those outrageous headlines and social media posts that just reek of something fishy. That is the world of fake news.
This project tackles this problem head-on, aiming to be your weapon against misinformation.  It's packed with tools and resources to help you:
Spot fake news from a mile away! 📰
Don't be a pawn in the fake news game! ⚔️
Spread the truth like wildfire! 🔥
Once you've debunked a myth, share your knowledge and help others navigate the online information jungle. Be a beacon of truth!💡
-->

<!--
#### TfidfVectorizer--
TF : Term Frequency: The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.

IDF : Inverse Document Frequency: Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.

The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

#### PassiveAggressiveClassifier--
Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

### Detecting Fake News with Python--
To build a model to accurately classify a piece of news as REAL or FAKE.

This project deals with fake and real news. Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

The fake news Dataset
The dataset we’ll use for this python project- we’ll call it news.csv. This dataset has a shape of 7796×4. The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE. The dataset takes up 29.2MB of space and you can download it here.

Project Prerequisites
You’ll need to install the following libraries with pip:

pip install numpy pandas sklearn
You’ll need to install Jupyter Lab to run your code. Get to your command prompt and run the following command:


### Steps for detecting fake news with Python
Follow the below steps for detecting fake news and complete your first advanced Python Project –

Make necessary imports:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

2. Now, let’s read the data into a DataFrame, and get the labels from the DataFrame..
df=pd.read_csv('news.csv')
labels=df.label
labels.head()

4. Split the dataset into training and testing sets.
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

5. Let’s initialize a TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7 (terms with a higher document frequency will be discarded).
Stop words are the most common words in a language that are to be filtered out before processing the natural language data.
And a TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.

Now, fit and transform the vectorizer on the train set, and transform the vectorizer on the test set.

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

6. Next, we’ll initialize a PassiveAggressiveClassifier. This is. We’ll fit this on tfidf_train and y_train.
Then, we’ll predict on the test set from the TfidfVectorizer and calculate the accuracy with accuracy_score() from sklearn.metrics.

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

7. We got an accuracy of 93.162% with this model. Finally, let’s print out a confusion matrix to gain insight into the number of false and true negatives and positives.

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
Output Screenshot:

python projects - confusion matrix

So with this model, we have 866 true positives, 905 true negatives, 63 false positives, and 67 false negatives.

Summary
Today, we learned to detect fake news with Python. We took a political dataset, implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fit our model. We ended up obtaining an accuracy of 93.16% in magnitude.
-->