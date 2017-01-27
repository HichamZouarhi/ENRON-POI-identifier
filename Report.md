**Detect Persons of Interest from ENRON Data**

**using machine learning in Python**


** 1- Summarize the Goal of the project :**

The goal of this project is to build an algorithm to identify a person of interest based on financial and email data made public as a result of the Enron scandal. The initial list of persons of interest in the fraud case, is made from individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

In a first time I will include all features and then select only the ten best features that will represent better the data

The total number of data points ( people in the training and testing dataset) = 146
among them there are 18 POI and 128 non-POI

In the initial dataset there are 21 available features, I picked manually the ones that seem to have more importance based on the mini-projects and identification of the dataset

On the features targetted there are some features missing values :

Data points with missing salary values : POI  1  / non-POI :  50
Data points with missing total payments values : POI  0  / non-POI :  21
Data points with missing bonus values : POI  2  / non-POI :  62
Data points with missing total stock values : POI  0  / non-POI :  20

By combining them with the email data we can compensate on the missing data

There are two outliers on the data one on that won't be needed in this investigation : 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'
they had been deleted from the dataset to avoid droping the accuracy of the identifier

These outliers are detected by ploting the data on several axisand reaching for the extreme values to check on them

for the rest of the outliers they seem to be valid data points for the top ranking official of ENRON

**1- features used in the identifier :**

by selecting the most important features on the data set I added an additional feature that I scaled to be as a proportion of the emails
exchanged from the person in the data entry with a Person Of Interest

ratio = ((emails from/to poi ) / total email from/to person)*100

the missing values from the email data received a 0 value for the ratio feature, since there doesn't seem to be a way to compute them

For the purpose of extracting the best features to identify the persons of interest I used pca, first with Minka’s MLE, and another algorithm using pca randomized with the selection of the K best features

After getting a suggestion to scale the features using a MinMaxScaler, the metrics used to evaluate the classifiers improved.

Also the 10 best features selected by the KBest algorithm are the following with their epective scores :

 ('exercised_stock_options', 24.815079733218194),
 ('total_stock_value', 24.182898678566872), 
 ('bonus', 20.792252047181538), 
 ('salary', 18.289684043404513), 
 ('deferred_income', 11.458476579280697), 
 ('long_term_incentive', 9.9221860131898385), 
 ('restricted_stock', 9.212810621977086), 
 ('total_payments', 8.7727777300916809), 
 ('loan_advances', 7.1840556582887247), 
 ('expenses', 6.0941733106389666)

the new feature added isn't among them which means that the classifiers that will use the 10 best features will nly be based on the 
financial data

** Pick and Tune an Algorithm :**

- by using a tuned decision tree classifier with randomized PCA and Kbest features selection I got the following metrics :
	Accuracy: 0.79707	Precision: 0.26033	Recall: 0.28350	F1: 0.27142	F2: 0.27854
	Total predictions: 15000	True positives:  567	False positives: 1611	False negatives: 1433	True negatives: 11389

which is not good enough to meet the specifications

- and by using a PCA on the dataset with the method MLE and an AdaBoost Classifier :

	Accuracy: 0.85613	Precision: 0.44626	Recall: 0.32800	F1: 0.37810	F2: 0.34636
	Total predictions: 15000	True positives:  656	False positives:  814	False negatives: 1344	True negatives: 12186

- for the Gaussian Naive Bayes classifier used with pca I get :

GaussianNB(priors=None)
	Accuracy: 0.81820	Precision: 0.31669	Recall: 0.31400	F1: 0.31534	F2: 0.31453
	Total predictions: 15000	True positives:  628	False positives: 1355	False negatives: 1372	True negatives: 11645

tuning the parameters of an algorithm is looking for the comination of parameters that will give the best performance for that given 
algorithm, in my case the algorithm I chose to be tuned was the feature selection and the parameters for the decision tree classifier,
Even though it has be tuned to get the best performance but it did not met the specifications of a precision and recall scores above .3

the DecisionTree classifier is tuned by the following parameters :
	- K best features : representing how many best features from 1 to 10
	- classification criterion : either gini for impurity or entropy for information gain
	- DecisionTree splitter : either the best splits or best random splits



** Validation **

In order to test a classification algorithm we'll have to keep a portion of the dataset for the tests, the most common mistake is 
training the classifier on the whole dataset, while this may give good results on the tests ( since the test data will be the one that 
it has been trained for) it won't be able to give good results on new data that would be added, this is the case where it is too good to 
be true.
In this investigation I started by leaving a 30% of the dataset for the testing since for both of the algorithms implemented the 
metrics are completly irrational when using a smaller portion of data for testing

** Metrics **

The metrics used to validate the algorithm are :
	- precision : the true positives / (true positives + false positives) is the ability of the classifier not to label as positive a
	sample that is negative
	- recall : true positives / (true positives + false negatives) is the ability of the classifier to find all the positive samples