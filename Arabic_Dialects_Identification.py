import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('arabic_dialects_dataset.csv')

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.3, random_state=42)

# Convert the text data to feature vectors
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data['text'])
test_features = vectorizer.transform(test_data['text'])

# Train the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(train_features, train_labels)

# Predict the labels for the testing data using Multinomial NB classifier
nb_pred = nb_classifier.predict(test_features)

# Evaluate the performance of Multinomial NB classifier
nb_accuracy = accuracy_score(test_labels, nb_pred)
nb_precision = precision_score(test_labels, nb_pred, average='macro')
nb_recall = recall_score(test_labels, nb_pred, average='macro')
nb_f1 = f1_score(test_labels, nb_pred, average='macro')
nb_cm = confusion_matrix(test_labels, nb_pred)

print("Multinomial NB Classifier Results:")
print("Accuracy: {:.2f}%".format(nb_accuracy*100))
print("Precision: {:.2f}%".format(nb_precision*100))
print("Recall: {:.2f}%".format(nb_recall*100))
print("F1 score: {:.2f}%".format(nb_f1*100))
print("Confusion matrix:")
print(nb_cm)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_features, train_labels)

# Predict the labels for the testing data using Random Forest classifier
rf_pred = rf_classifier.predict(test_features)

# Evaluate the performance of Random Forest classifier
rf_accuracy = accuracy_score(test_labels, rf_pred)
rf_precision = precision_score(test_labels, rf_pred, average='macro')
rf_recall = recall_score(test_labels, rf_pred, average='macro')
rf_f1 = f1_score(test_labels, rf_pred, average='macro')
rf_cm = confusion_matrix(test_labels, rf_pred)

print("Random Forest Classifier Results:")
print("Accuracy: {:.2f}%".format(rf_accuracy*100))
print("Precision: {:.2f}%".format(rf_precision*100))
print("Recall: {:.2f}%".format(rf_recall*100))
print("F1 score: {:.2f}%".format(rf_f1*100))
print("Confusion matrix:")
print(rf_cm)

# Save the results to an Excel file
results_df = pd.DataFrame({
    'Classifier': ['Multinomial NB', 'Random Forest'],
    'Accuracy': [nb_accuracy, rf_accuracy],
    'Precision': [nb_precision, rf_precision],
    'Recall': [nb_recall, rf_recall],
    'F1 score': [nb_f1, rf_f1]
})

results_df.to_excel('arabic_dialects_results.xlsx', index=False)

# Create a dataframe with actual and predicted labels for comparison
comparison_df = pd.DataFrame({
    'Actual': test_labels,
    'Multinomial NB': nb_pred,
    'Random Forest': rf_pred
})

# Save the comparison dataframe to an Excel file
comparison_df.to_excel('arabic_dialects_comparison.xlsx', index=False)

