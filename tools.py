import pandas as pd
import numpy as np
import folium
import json
from matplotlib import colors
import branca
from IPython.display import HTML
from matplotlib import pyplot as plt
import statsmodels.api as sm

def get_geojson_grid(upper_right, lower_left, n=6):
    all_boxes = []

    lat_steps = np.linspace(lower_left[0], upper_right[0], n+1)
    lon_steps = np.linspace(lower_left[1], upper_right[1], n+1)

    lat_stride = lat_steps[1] - lat_steps[0]
    lon_stride = lon_steps[1] - lon_steps[0]

    for lat in lat_steps[:-1]:
        for lon in lon_steps[:-1]:
            # Define dimensions of box in grid
            upper_left = [lon, lat + lat_stride]
            upper_right = [lon + lon_stride, lat + lat_stride]
            lower_right = [lon + lon_stride, lat]
            lower_left = [lon, lat]

            # Define json coordinates for polygon
            coordinates = [
                upper_left,
                upper_right,
                lower_right,
                lower_left,
                upper_left
            ]

            geo_json = {"type": "FeatureCollection",
                        "properties":{
                            "lower_left": lower_left,
                            "upper_right": upper_right
                        },
                        "features":[]}

            grid_feature = {
                "type":"Feature",
                "geometry":{
                    "type":"Polygon",
                    "coordinates": [coordinates],
                }
            }

            geo_json["features"].append(grid_feature)
            all_boxes.append(geo_json)

    return all_boxes

def create_map(agg_data):
    #agg_data = data.loc[datetime]
    indxs = np.reshape(np.reshape(np.arange(1,2501),(50,50)).T,(1,2500))[0]
    lower_left =  [40.49612, -74.25559]
    upper_right = [40.91553, -73.70000999999999]
    m = folium.Map(zoom_start = 11, location=[40.78,-73.96])
    grid = get_geojson_grid(upper_right, lower_left, n=50)
    
    for i, box in zip(indxs,grid):
        geo_json = json.dumps(box)
        if str(i) in agg_data.index:
            color = plt.cm.Reds(np.log(agg_data[str(i)]) / np.log(max(agg_data)))
            color = colors.to_hex(color)
            gj = folium.GeoJson(geo_json, style_function=lambda feature, color=color: {'fillColor': color,'color':"black",'weight': 2,'dashArray': '5, 5','fillOpacity': 0.55})
            popup = folium.Popup("{} taxi trips in area {}".format(int(np.round(agg_data[str(i)])),i))
            gj.add_child(popup)
            m.add_child(gj)
    return m

def embed_map(m):
    from IPython.display import HTML

    m.save('index.html')
    with open('index.html') as f:
        html = f.read()

    iframe = '<iframe srcdoc="{srcdoc}" style="width: 100%; height: 800px; border: none"></iframe>'
    srcdoc = html.replace('"', '&quot;')
    return HTML(iframe.format(srcdoc=srcdoc))

def get_Xy(data, zone, time):
    y = data[zone].shift(periods=-time)    
    X = pd.DataFrame(index=data.index)
    
    X['monday'] = (X.index.dayofweek.values == 0).astype(int)
    X['tuesday'] = (X.index.dayofweek.values == 1).astype(int)
    X['wednesday'] = (X.index.dayofweek.values == 2).astype(int)
    X['thursday'] = (X.index.dayofweek.values == 3).astype(int)
    X['friday'] = (X.index.dayofweek.values == 4).astype(int)
    X['saturday'] = (X.index.dayofweek.values == 5).astype(int)
    X['sunday'] = (X.index.dayofweek.values == 6).astype(int)
    
    k = 7
    N = len(data)
    T = np.array(range(1,N+1))
    for i in np.arange(1,k+1):
        s_name = 's_' + str(i)
        c_name = 'c_' + str(i)
        s = np.sin(T*2*np.pi*i/168.0)
        c = np.cos(T*2*np.pi*i/168.0)
        X[s_name] = s
        X[c_name] = c
    
    for i in range(0,24):
        X['hour_' + str(i)] = (X.index.hour.values == i).astype(int)
    for i in range(1,2):
        X['prev_'+str(i)] = y.shift(periods=i)
    X['prev_day_1'] = y.shift(periods=24)
    for i in range(1,5):
        X['prev_week_'+str(i)] = y.shift(periods=168*i)
    for i in range(1,2):
        X[str(i)+'_day_sum'] = y.rolling(window=24*i).sum()
    for i in [1,4]:
        X[str(i)+'_week_sum'] = y.rolling(window=168*i).sum()
    
    indx = pd.merge(X.dropna(),y.dropna(),left_index=True,right_index=True).index
    X = sm.add_constant(X.loc[indx])
    y = y.loc[indx]
    return X, y

def multiprediction(data, datetime):
    daterange = pd.date_range(start='2015-12-01 00:00:00',end=datetime,freq='H')
    predicted = pd.Series(index=data.columns)
    for zone in data.columns:
        X, y = get_Xy(data, zone,1)
        X = X.loc[daterange]
        y = y.loc[daterange]
        indxs = X.index[:-2]
        model = sm.OLS(y.loc[indxs],X.loc[indxs],missing='drop').fit()
        predicted[zone] = round(model.predict(np.array(X.loc[datetime]))[0])
    predicted[predicted < 0] = 0
    return predicted

def one_area_prediction(data, area, start, stop):
    X, y = get_Xy(data, area,0)
    indxs = pd.date_range(start=start,end=stop,freq='H')
    daterange = pd.date_range(start='2015-12-01 00:00:00',end=stop,freq='H')
    model = sm.OLS(y.loc[daterange],X.loc[daterange],missing='drop').fit()
    real_values = y[indxs]
    predicted_values = model.predict(X.loc[indxs]).round()
    predicted_values[predicted_values < 0] = 0
    return predicted_values, real_values, real_values - predicted_values

def plot_prediction(prediction):
    predicted, real, errors = prediction
    print 'First plot: real numbers of taxi drives (blue) and predicted (red)'
    print 'Second plot: errors'
    plt.figure(figsize=(25,14))
    plt.subplot(2,1,1)
    plt.plot(real,color='b')
    plt.plot(predicted,color='r')
    plt.subplot(2,1,2)
    plt.plot(errors)
    plt.show()
    print 'MAE: ', abs(errors).mean()
    print 'MSE: ', (errors**2).mean()
    print 'Mean Error: ', errors.mean()