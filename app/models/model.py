import pickle
import pandas as pd
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_model-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{BASE_DIR}/scaler-{__version__}.pkl", "rb") as file:
    scaler=pickle.load(file)
with open(f"{BASE_DIR}/order_input_model-{__version__}.txt", "r") as doc:
    lines=doc.readlines()

orden_variables=[]
for line in lines:
    orden_variables.append(line.replace("\n",''))

df_completa_vacia=pd.DataFrame(columns=orden_variables)

response={"store_type":"Gourmet Supermarket","store_state":"Zacatecas","grocery_sqft":"15012","frozen_sqft":"4193","coffee_bar":"1","video_store":"0","prepared_food":"1","florist":"0","media_type":"Daily Paper","marital_status":"M","gender":"F","total_children":"5","education":"Partial College","member_card":"Bronze","occupation":"Professional","houseowner":"N","avg_cars_at_home_approx":"4","avg_yearly_income":"$90K - $110K","num_children_at_home":"4","SRP":"2.8","net_weight":"17.8","recyclable_package":"1","low_fat":"0","units_per_case":"32","food_family":"Food","unit_sales_in_millions":"1","promotion_name":"Price Smashers","sales_country":"USA"}



def predict_cost(response):

    df_response=pd.DataFrame(response,index=[0])
    df_response[['unit_sales_in_millions','total_children','avg_cars_at_home_approx','num_children_at_home','SRP','net_weight','recyclable_package','low_fat','units_per_case','grocery_sqft','frozen_sqft','coffee_bar','video_store','prepared_food','florist']]=df_response[['unit_sales_in_millions','total_children','avg_cars_at_home_approx','num_children_at_home','SRP','net_weight','recyclable_package','low_fat','units_per_case','grocery_sqft','frozen_sqft','coffee_bar','video_store','prepared_food','florist']].astype('float64')
    df_response[['unit_sales_in_millions','total_children','avg_cars_at_home_approx','num_children_at_home','SRP','net_weight','recyclable_package','low_fat','units_per_case','grocery_sqft','frozen_sqft','coffee_bar','video_store','prepared_food','florist']]=scaler.transform(df_response[['unit_sales_in_millions','total_children','avg_cars_at_home_approx','num_children_at_home','SRP','net_weight','recyclable_package','low_fat','units_per_case','grocery_sqft','frozen_sqft','coffee_bar','video_store','prepared_food','florist']])
    df_response_dummies=pd.get_dummies(df_response)
    for column in df_response_dummies.columns:
        if column not in df_completa_vacia.columns: 
            print(column)
        df_completa_vacia.loc[0,column]=df_response_dummies.loc[0,column]
    df_completa=df_completa_vacia.fillna(0)
    
    return model.predict(df_completa)[0]
