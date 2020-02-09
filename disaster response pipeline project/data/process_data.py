import sys
import pandas as pd
from sqlalchemy import create_engine
import argparse
parser=argparse.ArgumentParser(description='Importing, processing and saving data')
parser.add_argument('--messages_filepath',type=str,default='data/disaster_messages.csv ',help='Path of the messages data file to use')
parser.add_argument('--categories_filepath',default='data/disaster_categories.csv' ,help='Path of the categories data file to use')
parser.add_argument('--database_filepath',type=str,default="data/DisasterResponse.db",help='Path of the cleaned database to save')


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, left_on='id', right_on='id')


def clean_data(df):
    categories = df['categories'].str.split(";",expand=True)
    row = categories.loc[0]
    category_colnames = [nam.split("-")[0] for nam in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.replace(column+"-","")
        categories[column] = categories[column].astype("int")
    df=df.drop('categories',axis=1)
    df = pd.concat([df, categories], axis=1, join ='inner')
    df=df.drop_duplicates()
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('cleanData', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()