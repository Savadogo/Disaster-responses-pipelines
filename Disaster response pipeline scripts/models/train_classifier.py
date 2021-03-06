import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

import argparse
parser=argparse.ArgumentParser(description='Training and saving a model')
parser.add_argument('--model_filepath',type=str,default='models/classifier.pk',help='Path of the model to save')
parser.add_argument('--database_filepath',type=str,default="data/DisasterResponse.db",help='Path of the cleaned database to use')


def load_data(database_filepath):
    """
    Loading the cleaned data resulting from the process_data.py script.
    - param database_filepath: the path of the cleaned data to use.
    return:
       - X: the features data;
       - Y: the target 
       - category_names: the category labels
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('cleanData',engine)
    df['related']=df['related'].replace(2,0)
    X = df[['message']].values
    Y   = df.loc[:,'related':'direct_report'].values
    category_names=df.loc[:,'related':'direct_report'].columns
    return X.ravel(),Y,category_names


def tokenize(text):
    """
    Tokenizing a text.
    - param text: the text to tokenize.
    return:
       - the clean_token;
    """
    regex_express='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(regex_express, text)
    for url in detected_urls:
        text =text.replace(url,'url_link')
        
    regex_express='[^a-zA-Z0-9]'
    detected_spc = re.findall(regex_express, text)
    for spc in detected_spc:
        text =text.replace(spc,' ')
        
    tokens = nltk.tokenize.word_tokenize(text)
    lemmatizer =  nltk.stem.WordNetLemmatizer() 
    
    clean_tokens = []
    for tok in tokens:
        clean_tok =lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Building the model to use on the clean data to predict the categories
    return:
       - A gridsearch object of the pipeline;
    """

    parameters = {
    'clf__estimator__learning_rate':[0.1,0.3,0.5,0.7,1],
    'clf__estimator__n_estimators':[10,25,50,80,100]}
    pipe=Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return GridSearchCV(pipe, parameters)

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating our model with the precision, recall and  f1-score metrics.
    - param model: the model to evalute.
    - param X_test: the features data to use for the test;
    - param Y_test: the targets data to use for the test;
    - category_names: the labels of the disaster categories;
    """
    y_pred=model.predict(X_test)
    ypred=pd.DataFrame(y_pred,columns=category_names)
    yreal=pd.DataFrame(Y_test,columns=category_names)
    for coll in category_names:
        print('-----'+coll+'-----')
        print(classification_report(yreal[coll], ypred[coll]))


def save_model(model, model_filepath):
    """
   Saving the trained model.
    - param model: the trained model to save.
    - param model_filepath: the path where to save the trained model.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
Processing all the steps to train the model.
    """
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