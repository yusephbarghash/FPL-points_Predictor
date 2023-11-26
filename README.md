# Fantasy Premier League Player Points Prediction 

## Overview

This simple project is aimed at predicting the points that Fantasy Premier League (FPL) players will score in the next gameweek. To achieve this, I conducted an analysis of historical FPL data and developed a machine learning model that uses various features to make predictions. 

## Data Source

- I utilized the FPL dataset from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
- Specifically, I worked with the player data from the 2023-24 season, focusing on the `players_raw.csv` file.

## Feature Selection

- I've selected a set of relevant features to be used as input variables for my machine learning model. These features were chosen based on their potential impact on a player's performance.
- Feature selection was carried out using various techniques, such as feature importance analysis and domain knowledge.

## Data Analysis

- I performed some simple data Analysis to git some helpful insights. This included sorting, graphs, and other data analysis techniques.
- Details of data analysis steps are available here in this file `FPL_analysis_and_preprocessing.ipynb` notebook.


## Data Preprocessing

- I performed data preprocessing to clean and prepare the dataset for analysis. This included handling missing values, feature engineering, and splitting the data into training and testing sets.
- Details of data preprocessing steps are available here in the `FPL_preprocessing_and_MLmodel.ipynb` notebook.


## Machine Learning Model

- I developed a machine learning model to predict player points in the next gameweek. The model is based on a regression approach.
- I used the scikit-learn library in Python and employed algorithms like Linear Regression, Random Forest, Gradient Boosting and others.
- Model hyperparameter tuning was carried out to optimize performance.
- More information about the model building process is available in the `FPL_preprocessing_and_MLmodel.ipynb` notebook.

## Model Evaluation

- I evaluated the model's performance using appropriate metrics, such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) and R2 accuracy metric.
- Cross-validation was used to assess the model's robustness.

## Model Deployment with Streamlit

#### Streamlit Integration
- The machine learning model developed for predicting player points has been deployed using Streamlit.
- Streamlit, a Python library, enables the creation of interactive web applications for showcasing and utilizing machine learning models.

#### Deployment Details
- The deployed application allows users to input player-specific parameters and get predictions for the upcoming gameweek.

#### Accessing the Deployed Model
- The deployed application can be accessed at[Streamlit App URL](link_to_your_streamlit_app).


## Conclusion

- In conclusion, I have successfully developed a machine learning model that predicts FPL player points for the next gameweek.
- The model's accuracy is `0.9121008860382093`, indicating its ability to provide accurate predictions.


## Acknowledgments

- I would like to express my gratitude to the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) project for providing the data used in this analysis.
- Special thanks to the scikit-learn and other open-source libraries that supported my machine learning efforts.

---

