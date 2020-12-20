# Sentimental-Analysis
Sentimental analysis using svm(support vector machine)
##### Description
*Sentiment analysis for a dataset of comments which are classified as positive or negative using support vector machine.*
*The steps included are data preprocessing and cleaning , label encoder, featur extraction using TF-IDF and then training and testing of model using svm.Image of the dataset used is*

![Dataset](/dataset.png)

##### Steps followed in the algorithm
1.Loading the csv file to a pandas Dataframe.
2.Performing Data cleaning such as tokenization ,removing stopwards and unique characters.
3.Label encoding the sentiemnts for better classification.
4.Feature extraction done on the dataset. Tf-idf is used to find the weightage of the importance of words.
5. Performing SVM algorithm using scikit learn library and finding accuracy.

##### Output
![](/output1.png)
