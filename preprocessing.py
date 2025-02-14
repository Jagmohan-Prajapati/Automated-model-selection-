import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from skopt.space import Integer, Real, Categorical
import requests

def load_data(url):
    """Load the dataset from the given URL."""
    response = requests.get(url)
    data = pd.read_csv(pd.compat.StringIO(response.text))
    return data

def clean_data(data):
    """Clean the dataset by handling missing values and encoding."""
    # Replace '?' with NaN for easier handling of missing values
    data = data.replace('?', np.nan)
    
    # Impute missing values with mode
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    print(f"Missing values after imputation: {data.isnull().sum().sum()}")
    
    # Encode all columns
    label_encoder = LabelEncoder()
    for column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])
    
    return data

def perform_eda(data):
    """Perform Exploratory Data Analysis."""
    # Plot the distribution of the target variable (party)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=data['party'], palette='pastel')
    plt.title('Distribution of Party Labels')
    plt.xlabel('Party (0=Democrat, 1=Republican)')
    plt.ylabel('Count')
    plt.savefig('party_distribution.png')
    plt.close()
    
    # Plot the correlation heatmap of features
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def split_data(data):
    """Split the dataset into training, validation, and testing sets."""
    X = data.drop('party', axis=1)
    y = data['party']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def define_search_space():
    """Define the search space for hyperparameters."""
    search_space = [
        Integer(10, 500, name="n_estimators"),
        Integer(1, 20, name="max_depth"),
        Real(0.01, 1.0, name="learning_rate", prior="log-uniform"),
        Real(0.1, 1.0, name="subsample"),
        Categorical(["gini", "entropy"], name="criterion"),
        Real(1e-5, 1e-2, name="min_impurity_decrease", prior="log-uniform")
    ]
    
    print("Search Space Defined:")
    for param in search_space:
        print(param)
    
    return search_space

def main():
    # Load data
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/house-votes-84-ODrtmcJiV6YWI7b35yMQ8Hetc0mKPo.csv"
    data = load_data(url)
    
    # Clean and encode data
    data_cleaned = clean_data(data)
    
    # Perform EDA
    perform_eda(data_cleaned)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_cleaned)
    
    # Define search space
    search_space = define_search_space()
    
    # Save preprocessed data
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)

if __name__ == "__main__":
    main()

