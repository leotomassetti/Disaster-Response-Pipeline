# Disaster-Response-Pipeline
The Disaster Response Pipelineproject is a part of Udacity Data Scientist Nanodegree. 

## Project Overview
- In this project, data engineering is applied to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
- During a disaster, the web app helps the user to get a classification result of messages in several categories and visualize the data in a display, reducing the potential reaction time of the responding organizations.

## Files
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- workspace
		- \app
			- run.py: flask file to run the app
			- \templates
				- master.html: main page of the web application 
				- go.html: result web page
		- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- DisasterResponse.db: disaster response database
			- process_data.py: ETL process
		- \models
			- train_classifier.py: classification code

## Required libraries
- nltk 3.3.0
- numpy 1.15.2
- pandas 0.23.4
- scikit-learn 0.20.0
- sqlalchemy 1.2.12

## Instructions

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
- [Figure Eight](https://www.figure-eight.com/) for dataset
- [Udacity](https://www.udacity.com/) for advice and review.
