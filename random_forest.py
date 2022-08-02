from asyncio.base_futures import _FINISHED
from operator import contains
from tkinter import E
from unicodedata import name
import requests
from msal import PublicClientApplication, SerializableTokenCache
import pandas as pd
import atexit
import os
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
from sklearn.model_selection import train_test_split

cache = SerializableTokenCache()
if os.path.exists("my_cache.bin"):
    cache.deserialize(open("my_cache.bin", "r").read())
atexit.register(lambda: open("my_cache.bin", "w").write(cache.serialize())
                # Hint: The following optional line persists only when state changed
                if cache.has_state_changed else None)
app = PublicClientApplication(client_id="59975f73-1443-4474-a4c5-f05c23f21980", authority="https://login.microsoftonline.com/organizations", token_cache=cache)
scope = "api://res-group.com/support-services-digital-platform/user_impersonation"
account = app.get_accounts()
if len(account) > 0:
    account = account[0]
result = app.acquire_token_silent([scope], account=account)
if result is None:
    result = app.acquire_token_interactive([scope])
if "access_token" not in result:
    print(result.get("error"))
    print(result.get("error_description"))
    exit(1)
access_token = result['access_token']

with open('CF_ODH_OAquery.sql') as sql:
    query = sql.read()


driver = 'ODBC Driver 17 for SQL Server'
server_name = 'smart-data-warehouse-production-global.database.windows.net'
database_name = 'SMART'
username = 'sam.fitzgerald@res-group.com'
authentication = 'ActiveDirectoryInteractive'

def Nice_data():
    url = "https://app.uno-res.com/api/assets?include=equipment"    
    response = requests.get(url=url, headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"})

    asset_data = response.json()

    data = pd.json_normalize(asset_data).set_index('id')

    equipment = data.explode('equipment')

    equipment = equipment['equipment'].apply(pd.Series)
    equipment['smart_key'] = equipment['metadata'].apply(pd.Series)['smart_key']
    equipment['hub_height'] = equipment['turbine'].apply(pd.Series)['hubHeight']
    data = data.join(equipment, rsuffix="_equipment", lsuffix="_asset")
    data.fillna(data.median())
    data.to_clipboard()
    return data

def create_df(data_type,query,driver,server_name,database_name,username,authentication):
    if data_type == 'csv':
        df = pd.read_csv(r"C:\Users\SaFitzgerald\OneDrive - RES Group\Documents\projects\fleet_intelligence\sql.csv",encoding="UTF-8")
    elif data_type == 'sql':
        df = sql_to_df(query,driver,server_name,database_name,username,authentication)
        df.to_csv('sql.csv')
    return df

def sql_to_df(query,driver,server_name,database_name,username,authentication):


    connection_string = f"DRIVER={{{driver}}};SERVER={server_name};PORT=1433;DATABASE={database_name};UID={username};AUTHENTICATION={authentication}"
    con = pyodbc.connect(connection_string)
    df = pd.read_sql_query(query, con)
    con.close()
    return df

def encode(data):
    df_numeric = data.select_dtypes(exclude=['object'])
    df_obj = data.select_dtypes(include=['object']).copy()

    cols = []
    for c in df_obj:
        dummies = pd.get_dummies(df_obj[c])
        dummies.columns = [c + "_" + str(x) for x in dummies.columns]
        cols.append(dummies)
    df_obj = pd.concat(cols, axis=1)

    data = pd.concat([df_numeric, df_obj], axis=1)
    data.reset_index(inplace=True, drop=True)
    
    return data 

def merge(): 
    sql_df = create_df('csv',query,driver,server_name,database_name,username,authentication)
    print(1)
    api_df = Nice_data()
    print(2)
    api_df[api_df['smart_key'].notna()]
    print(3)
    api_df['subtechnology'] = api_df['subtechnology'].fillna('Onshore')
    print(4) 
    api_df.rename(columns = {'name_asset':'Site', 'smart_key':'TurbineApiKey'}, inplace = True)
    print(5)
    combined_df = pd.merge(sql_df, api_df, how='inner', on=['Site', 'TurbineApiKey'])
    print(6)
    #combined_df.to_csv('test.csv')
    print('finished')
    return combined_df
    


#create_df('csv',query,driver,server_name,database_name,username,authentication)
#Nice_data()

def random_forest(metric):
    rf_df = merge()
    useful_cols = [metric,
       'WindSpeedMean', 'WindSpeedStandardDeviation',
       'country.id', 'manufacturer','location.lat',
       'location.lng','hub_height','ratedPowerMW',
       'TemperatureMean','TurbulenceMean',
       'TemperatureStandardDeviation', 'location.elevation']
    target_col = 'CapacityFactor'
    
    rf_df =rf_df[useful_cols].fillna(rf_df.median())
    rf_df.to_clipboard()
    rf_df = encode(rf_df)
    
    
    X = rf_df
    y = X.pop(metric)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    model = RandomForestRegressor(n_estimators = 10, random_state=12345)
    
    model.fit(X_train, y_train)
    print(pd.Series(model.feature_importances_,index = X_train.columns))
    importance = pd.Series(model.feature_importances_, index = X_train.columns).sort_values()
    importance.plot.bar()
    plt.show(block=True)
    y_pred = model.predict(X_test)
    compare_df = pd.DataFrame({"actual": y_test, "pred": y_pred})
    compare_df.plot.scatter(x="actual", y="pred")
    print(compare_df.corr())
    plt.show(block=True)

random_forest('OperationalAvailability')