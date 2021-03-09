# Disaster-Response-Pipeline
The Disaster Response Pipelineproject is a part of Udacity Data Scientist Nanodegree. 

## Project Overview
- In this project, data engineering is applied to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

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

## Acknowledgements
- [Figure Eight](https://www.figure-eight.com/) for dataset
- [Udacity](https://www.udacity.com/) for advice and review.
