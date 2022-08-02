from operator import contains
from tkinter import E
from unicodedata import name
import requests
from msal import PublicClientApplication, SerializableTokenCache
import pandas as pd
import atexit
import os
from sklearn.ensemble import RandomForestRegressor
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pyodbc
from sklearn.model_selection import train_test_split
import numpy as np

with open('CF_ODH_OAquery.sql') as sql:
    query = sql.read()


driver = 'ODBC Driver 17 for SQL Server'
server_name = 'smart-data-warehouse-production-global.database.windows.net'
database_name = 'SMART'
username = 'sam.fitzgerald@res-group.com'
authentication = 'ActiveDirectoryInteractive'

def sql_to_df(query,driver,server_name,database_name,username,authentication):


    connection_string = f"DRIVER={{{driver}}};SERVER={server_name};PORT=1433;DATABASE={database_name};UID={username};AUTHENTICATION={authentication}"
    con = pyodbc.connect(connection_string)
    df = pd.read_sql_query(query, con)
    con.close()
    return df


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

    return data

def filter_equip(filter_dict,Range= True):
    assets_df = Nice_data()
    dict_list = list(filter_dict.keys())
    column_list = list(assets_df.columns)
    absent_list =[]
    for x in dict_list:
        if x not in column_list:
            absent_list.append(x)
    dict_list = list(filter_dict.keys())
    filterable_dict = {key:val for key, val in filter_dict.items() if key not in absent_list}
    assets_df = assets_df[list(filterable_dict.keys())]
    assets_df[assets_df['smart_key'].notna()]
    assets_df['subtechnology'] = assets_df['subtechnology'].fillna('Onshore')
        
    for Filter_col, Filter in filterable_dict.items():
        if len(Filter) >=1: 
            if len(Filter) == 2 and Range == True:
                if type(Filter[0]) == int or type(Filter[0]) == float:
                    assets_df= assets_df[assets_df[str(Filter_col)].between(Filter[0],Filter[1])]
                else:
                    assets_df = assets_df.loc[assets_df[str(Filter_col)].isin(Filter)]
            else:
                assets_df = assets_df.loc[assets_df[str(Filter_col)].isin(Filter)]
    assets_df.loc[assets_df.astype(str).drop_duplicates().index]
    return assets_df


def haversine(lat1, long1, lat2, long2):
    # convert decimal degrees to radians 
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

    # haversine formula 
    dlong = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    distance = c*r
    return distance


Filter_equip_dict1 = {'name_asset':[],
'country.id':[], 'country.subdivision':[],
'technology':['Wind'],
'subtechnology':[],
'location.lat':[],
'location.lng':[],
'location.elevation':[],
'gridConnectionCapacity.exportMW':[],
'name_equipment':[],
'ratedPowerMW':[],
'model':[],
'manufacturer':[],
'gridConnectionPointId':[],
'hub_height':[],
'smart_key':[],
'month':[],
'WindSpeedMean':[],
'TemperatureMean':[],
'TurbulenceMean':[],
'TemperatureStandardDeviation':[]
    }

Filter_equip_dict2 = {'name_asset':[],
'country.id':[], 'country.subdivision':[],
'technology':['Wind'],
'subtechnology':[],
'location.lat':[],
'location.lng':[],
'location.elevation':[],
'gridConnectionCapacity.exportMW':[],
'name_equipment':[],
'ratedPowerMW':[],
'model':[],
'manufacturer':[],
'gridConnectionPointId':[],
'hub_height':[],
'smart_key':[],
'month':[],
'WindSpeedMean':[]
}

Filter_equip_dict3 = {'name_asset':[],
'country.id':[], 'country.subdivision':[],
'technology':['Wind'],
'subtechnology':[],
'location.lat':[],
'location.lng':[],
'gridConnectionCapacity.exportMW':[],
'name_equipment':[],
'ratedPowerMW':[],
'model':[],
'manufacturer':['Senvion'],
'gridConnectionPointId':[],
'hub_height':[],
'smart_key':[],
'month':[],
'WindSpeedMean':[]

    }

Filter_equip_dict4 = {'name_asset':[],
'country.id':[], 'country.subdivision':[],
'technology':['Wind'],
'subtechnology':[],
'location.lat':[],
'location.lng':[],
'gridConnectionCapacity.exportMW':[],
'name_equipment':[],
'ratedPowerMW':[],
'model':[],
'manufacturer':['Nordex'],
'gridConnectionPointId':[],
'hub_height':[],
'smart_key':[],
'month':[],
'WindSpeedMean':[]

    }

Filter_equip_dict5 = {'name_asset':[],
'country.id':[], 'country.subdivision':[],
'technology':['Wind'],
'subtechnology':[],
'location.lat':[],
'location.lng':[],
'gridConnectionCapacity.exportMW':[],
'name_equipment':[],
'ratedPowerMW':[],
'model':[],
'manufacturer':['Gamesa'],
'gridConnectionPointId':[],
'hub_height':[],
'smart_key':[],
'month':[],
'WindSpeedMean':[]

    }
def merge_data(filter_dict,centre,rng,numerical_range,df):
    months = filter_dict['month']
    month_list = month_to_num(months)
    equip = filter_equip(filter_dict,numerical_range)
    equip.rename(columns = {'name_asset':'Site', 'smart_key':'TurbineApiKey'}, inplace = True)
    filtered_equip = pd.merge(df, equip, how='inner', on=['Site', 'TurbineApiKey'])
    if len(month_list) >= 1:
        filtered_equip = filtered_equip[filtered_equip['StartDate'].dt.month.isin(month_list)]
    Filter1 = filter_dict.get('WindSpeedMean')
    Filter2 = filter_dict.get('TemperatureMean')
    Filter3 = filter_dict.get('TemperatureStandardDeviation')
    Filter4 = filter_dict.get('TurbulenceMean')
    if len(Filter1) == 2:
        filtered_equip= filtered_equip[filtered_equip['WindSpeedMean'].between(Filter1[0],Filter1[1])]
    label = [x for x in list(filter_dict.values()) if x]
    if len(Filter2) == 2:
        filtered_equip= filtered_equip[filtered_equip['TemperatureMean'].between(Filter2[0],Filter2[1])]
    if len(Filter3) == 2:
        filtered_equip= filtered_equip[filtered_equip['TemperatureStandardDeviation'].between(Filter3[0],Filter3[1])]
    if len(Filter4) == 2:
        filtered_equip= filtered_equip[filtered_equip['TurbulenceMean'].between(Filter4[0],Filter4[1])]
    if rng and centre != False:
        
        result = [(x,y) for x, y in zip(filtered_equip['location.lat'], filtered_equip['location.lng'],)]
        result2 = [haversine(centre[0], centre[1], x[0], x[1]) for x in result]
        print(result2)
        filtered_equip['Distance'] = result2
        filtered_equip = filtered_equip[filtered_equip['Distance'] <= rng]
        filtered_equip["Comparison"] = f'{label} and within {rng} from {centre}'
    else:
        filtered_equip["Comparison"] = str(label)
    no_equip = filtered_equip.drop_duplicates(subset=['Site','TurbineName'], keep='first')
    no_sites = filtered_equip.drop_duplicates(subset='Site', keep='first')
 
    print(f'Number of sites satisfied by dict is', len(no_sites))
    print(f'Equipment numbers satisfied by dict is', len(no_equip))
    
    return filtered_equip

    
def create_df(data_type,query,driver,server_name,database_name,username,authentication):
    if data_type == 'csv':
        df = pd.read_csv(r"C:\Users\SaFitzgerald\OneDrive - RES Group\Documents\projects\fleet_intelligence\sql.csv",encoding="UTF-8")
    elif data_type == 'sql':
        df = sql_to_df(query,driver,server_name,database_name,username,authentication)
        df.to_csv('sql.csv')
    return df

def plot_Factor(Filter_equip_dict1,Filter_equip_dict2,Filter_equip_dict3,Filter_equip_dict4,Filter_equip_dict5,metric,Start_Date,End_Date,coord,rng,numerical_range,data_type,query,driver,server_name,database_name,username,authentication):#not yet got data for equip
    df = create_df(data_type,query,driver,server_name,database_name,username,authentication)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df.to_parquet("raw_data.pq")
    Start_Date = pd.to_datetime(Start_Date)
    End_Date = pd.to_datetime(End_Date)
    df = df.loc[(df['StartDate'] >= Start_Date) & (df['StartDate'] <= End_Date)]
    filtered_equip1 = merge_data(Filter_equip_dict1,coord[0],rng[0],numerical_range,df)
    filtered_equip1.to_clipboard()
    equip_list = [filtered_equip1]
    if Filter_equip_dict2 != False:
        filtered_equip2 = merge_data(Filter_equip_dict2,coord[1],rng[1],numerical_range,df)
        equip_list.append(filtered_equip2)
        if Filter_equip_dict3 != False:
            filtered_equip3 = merge_data(Filter_equip_dict3,coord[2],rng[2],numerical_range,df)
            equip_list.append(filtered_equip3)
            if Filter_equip_dict4 != False:
                filtered_equip4 = merge_data(Filter_equip_dict4,coord[3],rng[3],numerical_range,df)
                equip_list.append(filtered_equip4)
                if Filter_equip_dict5 != False:
                    filtered_equip5 = merge_data(Filter_equip_dict5,coord[4],rng[4],numerical_range,df)
                    equip_list.append(filtered_equip5)
    df = pd.concat(equip_list, ignore_index=True, sort=False)
    df = df[df[metric].notna()]
    df.sort_values(metric)
    Q1=df[metric].quantile(0.25)
    Q3=df[metric].quantile(0.75)
    IQR=Q3-Q1
    df = df.loc[(df[metric] >= (Q1 - 1.5 * IQR)) & (df[metric] <= (Q3 + 1.5 * IQR))]
    df = df.sort_values('StartDate')
    df['month_year'] = df['StartDate'].apply(lambda x: x.strftime('%b-%Y'))
     

    '''sns.set_theme(style="whitegrid")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.histplot(df, x=metric, hue="Comparison", fill = True,kde=True,element='poly', ax=ax_hist)
    sns.boxplot(filtered_equip1[metric], ax=ax_box)
    ax_box.set(xlabel='Your Site')
    #ax = sns.displot(df, x=metric, hue="Comparison", kind="kde", bw_adjust=.5, fill = True, cut = 0)
    #sns.boxplot(df[metric],ax=ax)
    plt.show(block=True)
    f, axs = plt.subplots(2,1,
     #                 figsize=(8,6),
                      sharex=True,
                      gridspec_kw=dict(height_ratios=[0.5,3]))
    # makde density plot along x-axis without legend
    sns.boxplot(df[metric],ax=axs[0])
    # make scatterplot with legends
    sns.displot(df, x=metric, hue="Comparison", kind="kde", bw_adjust=.5, fill = True, cut = 0,ax=axs[1])
    f.tight_layout()
    #f, (ax_box, ax_dis) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    
    #ax = sns.violinplot(x=filtered_equip1[metric])
    #ax.set_xticklabels(ax.get_xticklabels(),rotation = 25)
    #ax_box.set(xlabel='')
    #ax = sns.violinplot(x='Comparison', y=metric, data=df, cut = 0, bw=.2)
    #ax.set_xticklabels(ax.get_xticklabels(),rotation = 25)
    plt.show(block=True)
    #ax.set_xticklabels(ax.get_xticklabels(),rotation = 25)
    df.to_csv('merged_df_for_plot.csv')'''
    return df

def month_to_num(months):
    comp_months = []
    for x in months:
        if x == 'Summer' or x == 'summer':
            months.extend(['Jun','Jul','Aug'])
        elif x == 'Autumn' or x == 'autumn':
            months.extend(['Sep','Oct','Nov'])
        elif x == 'Winter' or x == 'winter':
            months.extend(['Dec','Jan','Feb'])
        elif x == 'Spring' or x == 'spring':
            months.extend(['Mar','Apr','May'])
        elif len(x) == 3:
            datetime_object = datetime.datetime.strptime(x, "%b")
            comp_months.append(datetime_object.month)
        else:
            datetime_object = datetime.datetime.strptime(x, "%B")
            comp_months.append(datetime_object.month)
    return comp_months


coord = [(46.917, 1.719),(52.366, -0.814),(0,0),(0,0),(0,0)]
rng = [False,False,False,False,False]

#plot_Factor(Filter_equip_dict1,Filter_equip_dict2,False,Filter_equip_dict4,Filter_equip_dict5,'CapacityFactor',"2011-01-01","2021-12-12",coord,rng,True,'csv',query,driver,server_name,database_name,username,authentication)#leave coords, and set range to false if distance filter not in use)

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


def rf(metric = 'OperationalAvailability'):
    rf_df = plot_Factor(Filter_equip_dict1,False,Filter_equip_dict3,Filter_equip_dict4,Filter_equip_dict5,'OperationalAvailability',"2011-01-01","2021-12-12",coord,rng,True,'csv',query,driver,server_name,database_name,username,authentication)
    
    useful_cols = [metric,
       'WindSpeedMean', 'WindSpeedStandardDeviation',
       'country.id', 'manufacturer','location.lat',
       'location.lng','hub_height','ratedPowerMW',
       'TemperatureMean','TurbulenceMean',
       'TemperatureStandardDeviation', 'location.elevation']
    target_col = metric
    
    #rf_df =rf_df[useful_cols].fillna(rf_df.median())
    rf_df = rf_df[useful_cols].dropna()
    rf_df['randNumCol'] = np.random.randint(0, 1000, rf_df.shape[0])
    rf_df = encode(rf_df)
    
    
    X = rf_df
    y = X.pop(target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    model = RandomForestRegressor(n_estimators = 25, random_state=123)
    
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

rf()