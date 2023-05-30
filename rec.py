#imports
import pandas as pd
import numpy as np
import warnings
import string
import json


from flask import Flask, request
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from geopy.geocoders import Nominatim 
from flask_cors import CORS

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

df = pd.read_csv('.\locations.csv', sep=',')
f = open('touristplaces.json')

data = json.load(f)


country = "India"
city_names = df["location"]

longitude =[]
latitude =[]
geolocator = Nominatim(user_agent="Trips")

for c in city_names.values:
    location = geolocator.geocode(c+','+ country)
    latitude. append(location. latitude)
    longitude. append(location. longitude)

df["Latitude"] = latitude
df["Longitude"] = longitude

l2 = df.iloc[:,-1:-3:-1]

kmeans = KMeans(5)
kmeans.fit(l2)

identified_clusters = kmeans.fit_predict(l2)
identified_clusters = list(identified_clusters)

df["loc_clusters"] = identified_clusters


@app.route("/getTouristPlaces", methods=["POST"])
def get_tourist_places():
    try:
        city = request.json
        print('1------------', city["city"])
        print(data[city["city"]])
        print('2------------')
        return data[city["city"]]
    except Exception as error:
        print(error)
        return ["test"]


@app.route("/getrecommendation/<place>", methods=['GET'])
def hello_world(place):
    try:
        formattedPlace = string.capwords(place, sep = None)
        cluster = df.loc[df['location'] == formattedPlace, 'loc_clusters']
        cluster = cluster.iloc[0]
        cities = df.loc[df['loc_clusters'] == cluster, 'location']
        city_lst = []
        for c in range(len(cities)):
            if cities.iloc[c] == formattedPlace:
                continue
            else:
                city_lst.append(cities.iloc[c])
        city_lst = set(city_lst)
        print(city_lst)

        return list(city_lst)
    except:
        return list()
    
app.run(host='0.0.0.0', port=8000)


# # #D:\Bhupendra\recomendation_system
# input_city = input("Enter a city name: ")

# #cluster


# #print(type(cities))
#         #a = cities.iloc[c]
#         #print(type(cities.iloc[c]))
#     # print(city_lst)


