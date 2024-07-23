## Reddit Text Classification with Machine Learning
***There will be a portuguese and a english version - under review of code and documentation***

This script utilizes machine learning techniques to classify Reddit posts into various subjects. It employs a comprehensive pipeline that encompasses data loading, pre-processing, model training, evaluation, and visualization. The script aims to demonstrate the effectiveness of different machine learning algorithms for text classification tasks.

### Data Loading

The `load_data` function retrieves posts from specified Reddit subreddits and extracts their text content. It filters posts based on character count to ensure a minimum level of text length. The function returns two lists:

* `data`: Contains the post text.
* `labels`: Contains the corresponding subject labels.

### Data Splitting

The `split_data` function divides the data into training and testing sets using `train_test_split` from scikit-learn. This ensures the model is trained on a representative subset of the data and evaluated on unseen data. The function returns four sets:

* `X_treino`: Training data.
* `X_teste`: Testing data.
* `y_treino`: Training labels.
* `y_teste`: Testing labels.

### Preprocessing Pipeline

The `preprocessing_pipeline` function defines a text preprocessing pipeline, including the following steps:

1. **Character Removal:** Removes non-alphanumeric characters, URLs, and HTML tags using regular expressions.
2. **TF-IDF Vectorization:** Converts text into a numerical representation using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which considers the importance of words within a document and across the entire corpus.
3. **Dimensionality Reduction:** Reduces the dimensionality of the TF-IDF matrix using Truncated Singular Value Decomposition (TruncatedSVD) to improve computational efficiency.

### Model Creation

The `cria_modelos` function creates three machine learning models:

1. **K-Nearest Neighbors (KNN):** Classifies new data points based on their similarity to the nearest neighbors in the training set.
2. **Random Forest:** Ensembles multiple decision trees to make predictions, reducing overfitting and improving generalization.
3. **Logistic Regression:** Employs logistic regression to predict the probability of each subject label.

### Training and Evaluation

The `treina_avalia` function trains and evaluates the created models:

1. **Pipeline Construction:** Combines the preprocessing pipeline with each model into a single pipeline.
2. **Model Training:** Fits each pipeline to the training data using the `fit` method.
3. **Predictions:** Generates predictions for the testing data using the `predict` method.
4. **Evaluation:** Calculates classification metrics like precision, recall, and F1-score using the `classification_report` function to assess model performance.
5. **Results Storage:** Stores the results for each model in a list.

### Visualization

* **Distribution Plot:** The `plot_distribution` function creates a bar chart to visualize the distribution of posts across different subjects.
* **Confusion Matrix & Heatmap:** The `plot_confusion` function generates a confusion matrix and a heatmap to visualize the model's performance for each subject, providing insights into potential misclassifications.

### Script Execution

The `if __name__ == "__main__":` block ensures the script's code is executed only when run directly, not when imported as a module. Here's a breakdown of the execution steps:

1. **Data Loading:** Calls the `load_data` function to load the data.
2. **Data Splitting:** Calls the `split_data` function to split the data into training and testing sets.
3. **Preprocessing Pipeline:** Creates the preprocessing pipeline using the `preprocessing_pipeline` function.
4. **Model Creation:** Creates the machine learning models using the `cria_modelos` function.
5. **Training and Evaluation:** Trains and evaluates the models using the `treina_avalia` function.
6. **Visualization:** Generates the distribution plot and confusion matrices for each model.

This comprehensive documentation provides a clear understanding of the script's functionality, making it a valuable resource for anyone interested in utilizing machine learning for text classification tasks.
