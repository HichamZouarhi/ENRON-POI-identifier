#!/usr/bin/python

import sys
import pickle
import pandas as pd
from pprint import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'total_payments', 'total_stock_value', 'poi_email_ratio',\
 'deferral_payments','deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 'loan_advances',\
  'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred'] 
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "total number of data points in the data : ", len(data_dict)

#pprint(data_dict.itervalues().next())
count_POI = 0
for person in data_dict:
	if data_dict[person]['poi']:
		count_POI+=1
print "number of POI : ",count_POI

count_non_POI = 0
for person in data_dict:
	if not data_dict[person]['poi']:
		count_non_POI+=1
print "number of non POI : ",count_non_POI

import math
count_poi_nan_salary = 0
count_non_poi_nan_salary = 0
count_poi_nan_total_payments = 0
count_non_poi_nan_total_payments = 0
count_poi_nan_bonus = 0
count_non_poi_nan_bonus = 0
count_poi_nan_total_stock_value = 0
count_non_poi_nan_total_stock_value = 0
for person in data_dict:
	if data_dict[person]['poi']:
		if math.isnan(float(data_dict[person]['salary'])):
			count_poi_nan_salary+=1
		if math.isnan(float(data_dict[person]['total_payments'])):
			count_poi_nan_total_payments+=1
		if math.isnan(float(data_dict[person]['bonus'])):
			count_poi_nan_bonus+=1
		if math.isnan(float(data_dict[person]['total_stock_value'])):
			count_poi_nan_total_stock_value+=1
	else:
		if math.isnan(float(data_dict[person]['salary'])):
			count_non_poi_nan_salary+=1
		if math.isnan(float(data_dict[person]['total_payments'])):
			count_non_poi_nan_total_payments+=1
		if math.isnan(float(data_dict[person]['bonus'])):
			count_non_poi_nan_bonus+=1
		if math.isnan(float(data_dict[person]['total_stock_value'])):
			count_non_poi_nan_total_stock_value+=1
print " Data points with missing salary values : POI ",count_poi_nan_salary," / non-POI : ",count_non_poi_nan_salary
print " Data points with missing total payments values : POI ",count_poi_nan_total_payments," / non-POI : ",count_non_poi_nan_total_payments
print " Data points with missing bonus values : POI ",count_poi_nan_bonus," / non-POI : ",count_non_poi_nan_bonus
print " Data points with missing total stock values : POI ",count_poi_nan_total_stock_value," / non-POI : ",count_non_poi_nan_total_stock_value


### Task 2: Remove outliers
import matplotlib.pyplot

data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
data_dict.pop('TOTAL',0) # this outlier has to be removed since it has nothing to do in this data
# data_to_plot = featureFormat(data_dict, ['bonus','salary'])

# for point in data_to_plot:
#     salary = point[0]
#     bonus = point[1]
#     matplotlib.pyplot.scatter( salary, bonus )

# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()

# I'll check now for outliers in the from/to POI emails

# data_to_plot = featureFormat(data_dict, ['from_poi_to_this_person','from_this_person_to_poi'])

# for point in data_to_plot:
#     x = point[0]
#     y = point[1]
#     matplotlib.pyplot.scatter( x, y )

# matplotlib.pyplot.xlabel("emails from POI to this person")
# matplotlib.pyplot.ylabel("emails from this person to POI")
# matplotlib.pyplot.show()

# the outliers seems valid data points, they should stay

# for name,person in data_dict.iteritems():
# 	#if person['from_poi_to_this_person']>500:
# 	if person['from_this_person_to_poi']>400:
# 		print name


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# the new feature I will add is the poi email ratio which will be calculated by poi from/to emails
#divided by the total emails sent or received

my_dataset = data_dict
for person in my_dataset:
	try:
		ratio = (float(my_dataset[person]['from_poi_to_this_person']\
			+my_dataset[person]['from_this_person_to_poi'])\
			/float(my_dataset[person]['from_messages']\
			+my_dataset[person]['to_messages']))*100
		my_dataset[person]['poi_email_ratio']=ratio
	except ValueError:
		my_dataset[person]['poi_email_ratio']=0
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from  sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

k_features = 10
k_best = SelectKBest(k=k_features)
k_best.fit(features, labels)
scores = k_best.scores_
feature_and_score = sorted(zip(features_list[1:], scores), key = lambda l: l[1],\
     reverse = True)


best_10_features_list = ['poi'] + [feature for (feature, score) in \
    feature_and_score][:k_features]
print "Top 10 features and their scores:\n", feature_and_score[:k_features], "\n"


pl = make_pipeline(SelectKBest(), PCA(random_state = 42, svd_solver='randomized'), DecisionTreeClassifier(random_state = 42))
params = dict(
	selectkbest__k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	decisiontreeclassifier__criterion = ['gini', 'entropy'],
	decisiontreeclassifier__splitter = ['best', 'random']
)

pca = PCA(n_components='mle')

from time import time

grid = GridSearchCV(pl, param_grid = params, scoring = 'recall')


from sklearn.ensemble import AdaBoostClassifier
clf_AdaBoost = AdaBoostClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

pca.fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

grid.fit(features_train, labels_train)
clf_DT = grid.best_estimator_

t0 = time()
clf_DT.fit(features_train,labels_train)
print "Decision Tree - training time:", round(time()-t0, 3), "s"
t1 = time()
predictions_DT = clf_DT.predict(features_test)
print "Decision Tree - prediction time:", round(time()-t1, 3), "s"

t0 = time()
clf_AdaBoost.fit(features_train_pca,labels_train)
print "AdaBoost - training time:", round(time()-t0, 3), "s"
t1 = time()
predictions_AdaBoost = clf_AdaBoost.predict(features_test_pca)
print "AdaBoost - prediction time:", round(time()-t1, 3), "s"

### Stochastic Gradient Descent
from sklearn import linear_model
clf_SGD = linear_model.SGDClassifier(class_weight = "balanced")

### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()

### Random Forests
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier()


clf_SGD.fit(features_train_pca,labels_train)
predictions_SGD = clf_SGD.predict(features_test_pca)

clf_NB.fit(features_train_pca,labels_train)
predictions_NB = clf_NB.predict(features_test_pca)

clf_RF.fit(features_train_pca,labels_train)
predictions_RF = clf_RF.predict(features_test_pca)


from sklearn.metrics import precision_score, recall_score
print "precision score for the Gaussian Naive Bayes Classifier : ",precision_score(labels_test,predictions_NB)
print "recall score for the Gaussian Naive Bayes Classifier : ",recall_score(labels_test,predictions_NB)

print "precision score for the Decision tree Classifier : ",precision_score(labels_test,predictions_DT)
print "recall score for the Decision tree Classifier : ",recall_score(labels_test,predictions_DT)

print "precision score for the AdaBoost Classifier : ",precision_score(labels_test,predictions_AdaBoost)
print "recall score for the AdaBoost Classifier : ",recall_score(labels_test,predictions_AdaBoost)

print "precision score for the Random Forest Classifier : ",precision_score(labels_test,predictions_RF)
print "recall score for the Random Forest Classifier : ",recall_score(labels_test,predictions_RF)

print "precision score for the Stochastic Gradient Descent Classifier : ",precision_score(labels_test,predictions_SGD)
print "recall score for the Stochastic Gradient Descent Classifier : ",recall_score(labels_test,predictions_SGD)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = clf_AdaBoost
dump_classifier_and_data(clf, my_dataset, features_list)