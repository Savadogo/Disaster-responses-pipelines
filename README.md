# Disaster-responses-pipelines
Repository for Udacity Data Science project which consist of building a Machine learning model to predict from a message on twitter the category of disaster the message is about.

## Packages requirements
- json
- plotly
- pandas
- nltk
- sklearn verion >=0.20.3
- sqlalchemy

# Instructions

## Running the ETL pipeline
Objectives : to clean the data and store it in a database.

Files used : a disater messages file and a disaster categories file.

command : python data/process_data.py messages_filepath categories_filepath clean_data_filepath

## Running the ML pipeline
Objectives : to train a classifier and saved it.

Files used : a clean database resulting from the ETL pipeline.

command : python models/train_classifier.py clean_database_filepath model_filepath

## Deploying the web app
Objectives : Running a web app which display statistics on the disaster database. The web app also allows the user to input a new message and gets from this message a prediction of the category of disaster the message is about.

   ### how to deploy:


# Licensing
"The code in this project is licensed under MIT license."
