import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.decomposition import TruncatedSVD
import re
import pickle
import nltk
import warnings
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load cleaned data from database into dataframe
    Args:
        database_filepath: String. Contains cleaned data table
        table_name: String. Contains cleaned data
    Returns:
       X: numpy.ndarray. Disaster messages
       Y: numpy.ndarray. Disaster categories for each messages
       category_name: list. Disaster category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM MessagesETL", engine)
    category_names = df.columns[4:]
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text (a disaster message).
    Args:
        text: String. A disaster message
        lemmatizer: nltk.stem.Lemmatizer
    Returns:
        tokens: list contains tokens
    """
    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)    
    
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")] 
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]    
    
    return tokens


def build_model():
    """
    Build model
    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    # Set pipeline
    new_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Set parameters for gird search
    parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [10, 20], 
              'clf__estimator__min_samples_split': [2, 4]} 

    # Set grid search
    cv = GridSearchCV(new_pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model
    Args:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator
        X_test: numpy.ndarray. Disaster messages
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names
    Returns:
        None
    """
    # Predict categories of messages.
    Y_predictionCV_test = model.predict(X_test)
    
    # Print classification report on test data
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_predictionCV_test[:, i]))

def save_model(model, model_filepath):
    """
    Save model
    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_filepath: String. Trained model is saved as pickel into this file.
    Returns:
        None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()