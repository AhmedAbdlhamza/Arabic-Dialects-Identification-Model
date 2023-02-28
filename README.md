# Arabic-Dialects-Identification-Model
here's a complete Python code to perform the Arabic dialects identification using MultinomialNB and Random Forest classifiers and evaluate the model using various performance metrics:

This creates a dataframe 'comparison_df' that has three columns: 'Actual', 'Multinomial NB', and Random ,Forest'. The Actual column contains the true labels for the testing data, while the Multinomial NB and Random Forest columns contain the predicted labels for the testing data using the corresponding classifier.

Finally, the comparison dataframe is saved to an Excel file called arabic_dialects_comparison.xlsx using the to_excel() method with index=False argument to exclude the row index from the Excel file.
