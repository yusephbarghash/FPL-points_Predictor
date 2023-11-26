
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder , StandardScaler , MinMaxScaler
from category_encoders import BinaryEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV , cross_validate
import joblib

Columns = joblib.load('inputs.pkl')
Model = joblib.load('model.pkl')

players_df = pd.read_csv('data/sheet.csv')
teams = list(players_df['team'].unique())
positions = list(players_df['position'].unique())

org_df = pd.read_csv('data/Data_for_the_ML_model.csv')

def prediction(player, position):
    df = pd.DataFrame(columns= Columns)
    
    df.at[0,'position'] = position
    df.at[0,'form'] = float(org_df[org_df['web_name'] == player]['form'].iloc[0])
    df.at[0,'now_cost'] = float(org_df[org_df['web_name'] == player]['now_cost'].iloc[0])
    df.at[0,'selected_by_percent'] = float(org_df[org_df['web_name'] == player]['selected_by_percent'].iloc[0])
    df.at[0,'event_points'] = int(org_df[org_df['web_name'] == player]['event_points'].iloc[0])
    df.at[0,'total_points'] = int(org_df[org_df['web_name'] == player]['total_points'].iloc[0])
    df.at[0,'ict_index'] = float(org_df[org_df['web_name'] == player]['ict_index'].iloc[0])
    df.at[0,'influence'] = float(org_df[org_df['web_name'] == player]['influence'].iloc[0])
    df.at[0,'creativity'] = float(org_df[org_df['web_name'] == player]['creativity'].iloc[0])
    df.at[0,'threat'] = float(org_df[org_df['web_name'] == player]['threat'].iloc[0])
    df.at[0,'transfers_in_event'] = int(org_df[org_df['web_name'] == player]['transfers_in_event'].iloc[0])
    df.at[0,'transfers_out_event'] = int(org_df[org_df['web_name'] == player]['transfers_out_event'].iloc[0])
    df.at[0,'bonus'] = int(org_df[org_df['web_name'] == player]['bonus'].iloc[0])
    df.at[0,'goals_scored'] = int(org_df[org_df['web_name'] == player]['goals_scored'].iloc[0])
    df.at[0,'assists'] = int(org_df[org_df['web_name'] == player]['assists'].iloc[0])
    df.at[0,'goals_conceded'] = int(org_df[org_df['web_name'] == player]['goals_conceded'].iloc[0])
    df.at[0,'clean_sheets'] = int(org_df[org_df['web_name'] == player]['clean_sheets'].iloc[0])
    df.at[0,'saves'] = int(org_df[org_df['web_name'] == player]['saves'].iloc[0])
    df.at[0,'own_goals'] = int(org_df[org_df['web_name'] == player]['own_goals'].iloc[0])
    df.at[0,'penalties_missed'] = int(org_df[org_df['web_name'] == player]['penalties_missed'].iloc[0])
    df.at[0,'penalties_saved'] = int(org_df[org_df['web_name'] == player]['penalties_saved'].iloc[0])
    df.at[0,'penalties_order'] = float(org_df[org_df['web_name'] == player]['penalties_order'].iloc[0])
    df.at[0,'yellow_cards'] = int(org_df[org_df['web_name'] == player]['yellow_cards'].iloc[0])
    df.at[0,'red_cards'] = int(org_df[org_df['web_name'] == player]['red_cards'].iloc[0])
    df.at[0,'starts'] = int(org_df[org_df['web_name'] == player]['starts'].iloc[0])
    df.at[0,'minutes'] = int(org_df[org_df['web_name'] == player]['minutes'].iloc[0])
    df.at[0,'expected_goals'] = float(org_df[org_df['web_name'] == player]['expected_goals'].iloc[0])
    df.at[0,'expected_assists'] = float(org_df[org_df['web_name'] == player]['expected_assists'].iloc[0])
    df.at[0,'expected_goal_involvements'] = float(org_df[org_df['web_name'] == player]['expected_goal_involvements'].iloc[0])
    df.at[0,'expected_goals_conceded'] = float(org_df[org_df['web_name'] == player]['expected_goals_conceded'].iloc[0])
    df.at[0,'chance_of_playing_next_round'] = float(org_df[org_df['web_name'] == player]['chance_of_playing_next_round'].iloc[0])
    df.at[0,'chance_of_playing_this_round'] = float(org_df[org_df['web_name'] == player]['chance_of_playing_this_round'].iloc[0])
    df.at[0,'status'] = str(org_df[org_df['web_name'] == player]['status'].iloc[0])
    df.at[0,'in_dreamteam'] = int(org_df[org_df['web_name'] == player]['in_dreamteam'].iloc[0])
    df.at[0,'dreamteam_count'] = int(org_df[org_df['web_name'] == player]['dreamteam_count'].iloc[0])
    df.at[0,'ep_this'] = float(org_df[org_df['web_name'] == player]['ep_this'].iloc[0])
    df.at[0,'strength'] = int(org_df[org_df['web_name'] == player]['strength'].iloc[0])
    df.at[0,'is_GKP'] = int(org_df[org_df['web_name'] == player]['is_GKP'].iloc[0])
    df.at[0,'goal_involvement'] = int(org_df[org_df['web_name'] == player]['goal_involvement'].iloc[0])
    
    result = Model.predict(df)
    return result[0]

def main():
    st.title('FPL points predictor')
    team = st.selectbox('Team', teams)
    position = st.selectbox('Position', positions)
    players_list = list(players_df[(players_df['team'] == team) & (players_df['position'] == position)]['web_name'])
    player = st.selectbox('Player', players_list)
    
    if st.button("Predict"):
        result = prediction(player, position)
        st.text(f"The expected poits this player could score is : {result}")
    
main()
