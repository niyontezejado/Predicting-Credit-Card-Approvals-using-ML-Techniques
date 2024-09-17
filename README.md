# Predicting Credit Card Approvals Using Neural Networks and Machine Learning Techniques

## Project Overview

This project aims to predict whether a credit card application will be approved based on applicant data using neural networks and other machine learning algorithms. The dataset consists of various attributes related to applicants such as their income, credit history, and personal information. By applying machine learning techniques, the project seeks to develop a model that can accurately classify whether a credit card application should be approved or denied.

## Project Structure

- **Data Preparation**: Loading the dataset, cleaning, and preprocessing (handling missing values, encoding categorical variables, and normalizing the data).
- **Model Building**: Implementing different machine learning algorithms, including:
  - Neural Networks
  - Random Forest
  - K-Nearest Neighbors
  - Logistic Regression
- **Model Evaluation**: Assessing the performance of the models using evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- **Hyperparameter Tuning**: Applying GridSearchCV to optimize the hyperparameters of the models.

## Dataset

The dataset used in this project consists of 690 records and 16 features. The features include both categorical and numerical variables, and the target variable indicates whether a credit card application is approved (`+`) or denied (`-`).

- **A1**: Categorical
- **A2**: Numerical
- **A3**: Numerical
- **A4**: Categorical
- **A5**: Categorical
- **A6**: Categorical
- **A7**: Categorical
- **A8**: Numerical
- **A9**: Categorical
- **A10**: Categorical
- **A11**: Numerical
- **A12**: Categorical
- **A13**: Categorical
- **A14**: Numerical
- **A15**: Numerical
- **A16**: Target (Approved `+` or Denied `-`)

## Technologies Used

- **Python**: Core programming language
- **Jupyter Notebook**: Environment for running and documenting code
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning library for model training and evaluation
- **Matplotlib/Seaborn**: Data visualization libraries
- **GridSearchCV**: Hyperparameter optimization

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/niyontezejado/Predicting-Credit-Card-Approvals-using-ML-Techniques.git
    ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
   
3. Run the Jupyter notebook to execute the project:
    ```bash
    jupyter notebook
    ```

## Steps to Run the Project

1. **Load the Dataset**: The dataset is loaded and inspected for any missing values or inconsistencies.
2. **Preprocessing**: Categorical variables are encoded, missing values are handled, and numerical features are normalized.
3. **Model Training**: Train the machine learning models using the preprocessed data.
4. **Model Evaluation**: Evaluate each model's performance using metrics such as accuracy, precision, recall, and confusion matrix.
5. **Hyperparameter Tuning**: Use GridSearchCV to find the best hyperparameters for the models.
6. **Conclusion**: Select the best model based on its performance and finalize the model for deployment.

## Results

- **Accuracy**: The neural network model achieved an accuracy of `77%` on the test dataset.
- **Other Metrics**: Additional evaluation metrics such as precision, recall, and F1-score were used to assess the overall performance of the models.

## Conclusion and Future Work

The neural network model was able to predict credit card approvals with good accuracy. In future iterations, the model can be improved by:
- Including additional features or external datasets
- Trying more advanced algorithms such as Gradient Boosting or XGBoost
- Applying feature engineering techniques to better capture relationships between features

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.


