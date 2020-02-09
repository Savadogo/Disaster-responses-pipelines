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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

import argparse
parser=argparse.ArgumentParser(description='Training and saving a model')
parser.add_argument('--model_filepath',type=str,default='models/classifier.pk',help='Path of the model to save')
parser.add_argument('--database_filepath',type=str,default="data/DisasterResponse.db",help='Path of the cleaned database to use')


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('cleanData',engine)
    df['related']=df['related'].replace(2,0)
    X = df[['message']].values
    Y   = df.loc[:,'related':'direct_report'].values
    category_names=df.loc[:,'related':'direct_report'].columns
    return X.ravel(),Y,category_names


def tokenize(text):
    regex_express='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(regex_express, text)
    for url in detected_urls:
        text =text.replace(url,'url_link')
        
    regex_express='^[a-zA-Z0-9]'
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
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    ypred=pd.DataFrame(y_pred,columns=category_names)
    yreal=pd.DataFrame(Y_test,columns=category_names)
    for coll in category_names:
        print('-----'+coll+'-----')
        print(classification_report(yreal[coll], ypred[coll]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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