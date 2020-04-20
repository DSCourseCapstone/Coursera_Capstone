#!/usr/bin/env python
# coding: utf-8

# # Segmenting and Clustering Neighborhoods in Toronto

# ## This program will explore, segment and cluster neighborhoods in the city of Toronto

# ## Part 1: Build dataframe of Postal Codes by Borough and Neighborhood

# ### Import libraries; Scrape table

# In[1]:


import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


website_url = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup = BeautifulSoup(website_url,'lxml')
scrape_table = soup.find('table',{'class':'wikitable sortable'})
# scrape_table


# In[2]:


df_scraped = pd.read_html(str(scrape_table))
df_scraped = df_scraped[0].dropna(axis=0) #drop rows with missing values
df_scraped = df_scraped[~df_scraped['Borough'].isin(['Not assigned'])]
print(df_scraped.head())


# In[3]:


df_postal = df_scraped.groupby(['Postal code','Borough'], sort=False)['Neighborhood'].apply(', '.join).reset_index() #reset the index on the table
df_postal['Neighborhood'] = np.where(df_postal.Neighborhood == 'Not assigned', df_postal.Borough, df_postal.Neighborhood) #if Borough is not assigned, Borough is the same as Neighborhood
df_postal.head()


# In[4]:


df_postal.shape


# ## Part 2: Add Geographical Coordinates

# In[5]:


# Use a .csv file to get geographical coordinates
get_ipython().system('wget -O GeoCord.csv http://cocl.us/Geospatial_data/')


# In[6]:


df_coord = pd.read_csv('GeoCord.csv') # Read the csv data
df_coord.head()


# In[15]:


# Create Latitude and Longitude columns on the Postal table
df_postal['Latitude'] = np.nan
df_postal['Longitude'] = np.nan

# For each postal code in df_postal, find corresponding coordinates in df_coord and assign it to df_postal
for idx in df_postal.index:
  coord_idx = df_coord['Postal Code'] == df_postal.loc[idx, 'Postal code']
  df_postal.at[idx, 'Latitude'] = df_coord.loc[coord_idx, 'Latitude'].values
  df_postal.at[idx, 'Longitude'] = df_coord.loc[coord_idx, 'Longitude'].values

# Display the results
df_postal.head(20)


# In[7]:


get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes ')
import folium # map rendering library


# In[14]:


from geopy.geocoders import Nominatim


# In[13]:


address = 'Toronto'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# ## Create a map of Toronto

# In[16]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_postal['Latitude'], df_postal['Longitude'], df_postal['Borough'], df_postal['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# ## Part 3 - Neighborhood Clustering

# ## Focus on Scarborough

# In[17]:


scarborough_data = df_postal[df_postal['Borough'] == 'Scarborough'].reset_index(drop=True)
scarborough_data.head()


# In[18]:


address = 'Scarborough, Toronto'

geolocator = Nominatim(user_agent="scarborough_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Scarborough are {}, {}.'.format(latitude, longitude))


# In[19]:


# create map of Scarborough using latitude and longitude values
map_scarborough = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(scarborough_data['Latitude'], scarborough_data['Longitude'], scarborough_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_scarborough)  
    
map_scarborough


# In[20]:


CLIENT_ID = 'BEAPWRG0JOWTWTNU4CZIMAJ02AAM5MBOVSLA1OMMY4X2T5CT' # your Foursquare ID
CLIENT_SECRET = 'WP0TC2MBBCNKCBWXMACGMDZCLTPPCJDDBK3W5FVKQ0PLIYHS' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ## Let's explore the first neighborhood in our dataframe.

# In[21]:


scarborough_data.loc[0, 'Neighborhood']


# In[22]:


neighborhood_latitude = scarborough_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = scarborough_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = scarborough_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# ## Now, let's get the top 100 venues that are in Malvern/Rouge within a radius of 500 meters.

# In[23]:


# type your answer here
LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius

# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[35]:


results = requests.get(url).json()
results 


# In[36]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[37]:


import json # library to handle JSON files
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe


# In[38]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON


# In[39]:


# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[40]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## Let's create a function to repeat the same process to all the neighborhoods in Scarborough

# In[41]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
          # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    return(nearby_venues)


# In[53]:


scarborough_venues = getNearbyVenues(names=scarborough_data['Neighborhood'],
                                   latitudes=scarborough_data['Latitude'],
                                   longitudes=scarborough_data['Longitude']
                                  )


# In[54]:


print(scarborough_venues.shape)
scarborough_venues.head()


# In[55]:


scarborough_venues.groupby('Neighborhood').count()


# ## Let's find out how many unique categories can be curated from all the returned venues

# In[58]:


print('There are {} unique categories.'.format(len(scarborough_venues['Venue Category'].unique())))


# ## Analyze Each Neighborhood

# In[59]:


# one hot encoding
scarborough_onehot = pd.get_dummies(scarborough_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
scarborough_onehot['Neighborhood'] = scarborough_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [scarborough_onehot.columns[-1]] + list(scarborough_onehot.columns[:-1])
scarborough_onehot = scarborough_onehot[fixed_columns]

scarborough_onehot.head()


# In[60]:


scarborough_onehot.shape


# In[61]:


scarborough_grouped = scarborough_onehot.groupby('Neighborhood').mean().reset_index()
scarborough_grouped


# ## Let's confirm the new size

# In[62]:


scarborough_grouped.shape


# ## Let's print each neighborhood along with the top 5 most common venues

# In[63]:


num_top_venues = 5

for hood in scarborough_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = scarborough_grouped[scarborough_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ## Let's put that into a *pandas* dataframe

# ## First, let's write a function to sort the venues in descending order.

# In[102]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# ## Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[202]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = scarborough_grouped['Neighborhood']

for ind in np.arange(scarborough_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(scarborough_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Cluster Neighborhoods

# ## Run *k*-means to cluster the neighborhood into 3 clusters.

# In[203]:


# import k-means from clustering stage
from sklearn.cluster import KMeans


# In[204]:


# set number of clusters
kclusters = 3

scarborough_grouped_clustering = scarborough_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(scarborough_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# ## Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[205]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

scarborough_merged = scarborough_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
scarborough_merged = scarborough_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

scarborough_merged.head() # check the last columns!


# In[192]:


# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[ ]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(scarborough_merged['Latitude'], scarborough_merged['Longitude'], scarborough_merged['Neighborhood'], scarborough_merged['Cluster Labels']):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster],
        fill=True,
        fill_color=rainbow[cluster],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Examine Clusters

# In[206]:


scarborough_merged.loc[scarborough_merged['Cluster Labels'] == 0, scarborough_merged.columns[[1] + list(range(5, scarborough_merged.shape[1]))]]


# In[207]:


scarborough_merged.loc[scarborough_merged['Cluster Labels'] == 1, scarborough_merged.columns[[1] + list(range(5, scarborough_merged.shape[1]))]]


# In[208]:


scarborough_merged.loc[scarborough_merged['Cluster Labels'] == 2, scarborough_merged.columns[[1] + list(range(5, scarborough_merged.shape[1]))]]


# 
