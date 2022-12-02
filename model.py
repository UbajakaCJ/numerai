
import pandas as pd
import numpy as np
from sklearn import metrics, cross_validation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

def main():
	# Set seed for reproducibility
	np.random.seed(0)

	print("Loading data...")
	# Load the data from the CSV files
	training_data = pd.read_csv('numerai_training_data.csv', header=0)
	prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

	# Transform the loaded CSV data into numpy arrays
	Y = training_data['target']
	X = training_data.drop('target', axis=1)
	t_id = prediction_data['t_id']
	x_prediction = prediction_data.drop('t_id', axis=1)

	features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=0)


	# Spot check Algorithms
	models = []

	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))

	# Evaluate each model in turn

	# results = []
	# names = []

	# for name, model in models:
	# 	kfold = cross_validation.KFold(n=10, random_state=0)
	# 	cv_results = cross_validation.cross_val_score(model, features_train, labels_train, cv=kfold, scoring='accuracy')
	# 	results.append(cv_results)
	# 	names.append(name)
	# 	msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
	# 	print(msg)


	# 	# Compare Algorithms
	# 	fig = plt.figure()
	# 	fig.suptitle('AlGORITHM COMPARISON')
	# 	ax = fig.add_subplot(111)
	# 	plt.boxplot(results)
	# 	ax.set_xticklabels(names)
	# 	plt.show()


	# The model that will learn to predict
	clf = DecisionTreeClassifier(min_samples_split=90000)

	# Model is trained on the numerai_training_data
	clf.fit(features_train, labels_train)

	pred = clf.predict(features_test)
	pred_prob = clf.predict_proba(features_test)

	accuracy = accuracy_score(labels_test, pred)
	logloss = log_loss(labels_test, pred_prob)

	print('Accuracy: ', accuracy)
	print('Logloss: ', logloss)

	print('Predicting...')
	# The trained model is used to make predictions on the numerai_tournament_data
	# The interest is in the probability that the target is 1
	y_prediction = clf.predict_proba(x_prediction)
	value = y_prediction[:, 1]
	results_df = pd.DataFrame(data={'probability':value})
	joined = pd.DataFrame(t_id).join(results_df)

	print('Writing predictions to predictions.csv')
	# Save the predictions out to a CSV file
	joined.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
	main()