import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS

get_ipython().run_line_magic('matplotlib', 'inline')

raw_df=pd.read_csv('../input/zomato-restaurants-data/zomato.csv',encoding = "ISO-8859-1", engine='python')
country=pd.read_excel('../input/zomato-restaurants-data/Country-Code.xlsx')


df=pd.merge(raw_df,country,on='Country Code',how='left')



df.head(2)


# ## EDA


## Check the statistical measurements of numerical columns
df.describe()


# 1. Max value of  Average Cost for two is 80000 which is possible because the data has values in different currencies.
# 2. The variance in  lat and long is low for 75% of the data signifying that most of the restaurants are in one location. The country code which has ~75% of the values is 1 which is most probably India.

## Check the number of rows, type of columns and if there is any null value in the dataset
df.info()


# ##### The dataset has 9551 rows and 21 columns. Only Cuisines column seems to have null values



## Another way of finding null values and column wise density of null values
plt.figure(figsize=(5,5))
sns.heatmap(df.isnull(),cbar=False,yticklabels=False)


# ##### Based on the heatmap there appears to null value in the cuisines column (white bars at the top).


## Cuisines column deep dive
sum(df['Cuisines'].isnull()==True)


# ##### There are 9 null values in the dataset.



## pair plot to perform the univariate/bivariate analysis
sns.pairplot(df)


# 1. Data is sparse as it contains a lot of 0 values
# 2. Aggregate rating is 0 for most of the restaurants. however >0 rating distribution is similar to normal distribution.
# 3. Aggregate rating increases with the number of votes.
# 4. Price range column has only 4 discreet values



plt.scatter(df['Rating color'],df['Rating text'])


# ##### The scatter plot shows which color corresponds to which rating


sns.distplot(df['Aggregate rating'])



import geopandas as gpd
from shapely.geometry import Point



india_shp=gpd.read_file('../input/world-map-shape-files/TM_WORLD_BORDERS-0.3.shp')


# ##### Restaurants plotting on World Map



geometry=[Point(xy) for xy in zip(df['Longitude'],df['Latitude'])]

crs = {'init': 'epsg:27700'}

gdf=gpd.GeoDataFrame(df['Restaurant ID'],crs=crs,geometry=geometry)

fig,ax=plt.subplots(figsize=(20,10))
india_shp.plot(ax=ax,cmap='Pastel2')
gdf.plot(ax=ax,color='red',markersize=12)


# ##### Based on the map, few latitude nad longitude does not appear to be correct

# #####  Country specific Data



df.groupby('Country').agg({'Restaurant ID':np.count_nonzero,'Votes':np.sum,'Aggregate rating':np.mean,'Average Cost for two':np.mean,'Currency':np.max,'Has Online delivery':np.max,}).reset_index().sort_values('Restaurant ID',ascending=False).reset_index(drop=True).rename(columns={'Restaurant ID':'No. of restaurants','Votes':'Total Votes'})


# 1. Online Delivery is present in two countries: India and UAE
# 2. Zomato operates in a total of 15 countries

# #### Top 5 Highest Rated Cuisines
df.groupby(['Cuisines'])['Aggregate rating'].mean().sort_values(ascending=False).head()


# #### Top Cuisine in each City (First 5 Examples)
df.groupby(['Country','City','Cuisines'])['Aggregate rating'].mean().reset_index().sort_values(['Country','City','Aggregate rating'],ascending=False).reset_index(drop=True).groupby(['Country','City']).head(1).reset_index(drop=True).head()


# #### Top Restaurant in each city (First 5 examples)
df.groupby(['Country','City','Restaurant Name'])['Aggregate rating'].mean().reset_index().sort_values(['Country','City','Aggregate rating'],ascending=False).reset_index(drop=True).groupby(['Country','City']).head(1).reset_index(drop=True).head()


# #### Top 10 Restaurants with maximum branches on Zomato
df.groupby('Restaurant Name')['Restaurant Name'].count().sort_values(ascending=False).head(10)




expanded_df=df.groupby(['Currency','Price range'])['Average Cost for two'].mean().reset_index()
expanded_df['Average Cost for two']=expanded_df['Average Cost for two'].apply(lambda x: np.round(x,0))
expanded_df.pivot(index='Currency',columns='Price range',values='Average Cost for two').fillna(0)


# The table shows the average cost for two at different price ranges for different currencies. There is a positive correlation between average cost for two and price range. We can also see Dollar and Pound as the strongest currencies and Indonesian Rupiah to be the weakest.

# ## Top 10 Most popular restaurants (Max no of Votes) **
plt.figure(figsize=(20,10))
df.groupby(['Country','City','Restaurant Name'])['Votes'].count().reset_index().sort_values(['Country','City','Votes'],ascending=False).groupby(['Country','City']).head(1).sort_values('Votes',ascending=False).reset_index(drop=True).head(10).plot('Restaurant Name','Votes',kind='bar')


# ## Top 10 most popular cities (Max number of votes)
# 1. The metric can be used as a proxy to sort the cities by popularity on Zomato platform
df.groupby(['Country','City'])['Votes'].count().reset_index().sort_values('Votes',ascending=False).reset_index(drop=True).head(10).plot('City','Votes',kind='bar')


# New Delhi is clearly the most popular city out of 141 city Zomato operates in.

# #### %age of Restaurants with Online Delivery by City in India
a=pd.merge(df[(df['Country']=='India')].groupby(['City']).agg({'Restaurant ID':np.count_nonzero}).reset_index(),df[(df['Has Online delivery']=='Yes')&(df['Country']=='India')].groupby(['City']).agg({'Restaurant ID':np.count_nonzero}).reset_index(),on='City',how='inner')                                                    
a.rename(columns={'Restaurant ID_x':'Total_restaurants','Restaurant ID_y':'Online_restaurants'},inplace=True)
a['%age']=a['Online_restaurants']*100/a['Total_restaurants']
a['%age']=a['%age'].apply(lambda x: np.round(x,2))
a.sort_values('Online_restaurants',ascending=False).reset_index(drop=True)


#  #### WordCloud for most popular Cuisines In India
stopwords=list(STOPWORDS)
word_text=WordCloud(background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=5,
        random_state=1,
                    relative_scaling=0.5,
                   height=250,
                   width=300).generate(str(df[df['Country Code']==1]['Cuisines']))




plt.figure(figsize=(10,10))
plt.imshow(word_text,interpolation='bilinear')
plt.axis("off")


# #### WordCloud for most popular Restaurants in India
word_res=WordCloud(stopwords=stopwords,
                  height=250,
                  width=250,
                  random_state=2,
                  background_color='white',
                  max_words=200,
                  max_font_size=40, 
                   relative_scaling=0.5,
                  scale=5).generate(str(df[df['Country Code']==1]['Restaurant Name']))




plt.figure(figsize=(10,10))
plt.imshow(word_res)
plt.axis('off')



