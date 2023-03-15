import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import matplotlib as mpl
import matplotlib.pyplot as plt

df_train =pd.read_csv("C:/Users/kavimathi/Documents/S8-PROJECT/DATASET/Test.csv")
df_train.isna().sum(axis=0)
df_train.fillna(df_train.mean(), inplace = True)
df_train.dropna(axis=0, inplace=True)
df_train.isna().sum(axis=0)
df_train = df_train.drop(columns=["ID"])

fig2 = px.histogram(df_train,x='Age',color='Age',template='plotly_dark')
fig2 = px.histogram(df_train,x='Gender',color='Gender',template='plotly_dark')
fig2 = px.histogram(df_train,x='Ever_Married',color='Ever_Married',template='plotly_dark')
fig2 = px.histogram(df_train,x='Graduated',color='Graduated',template='plotly_dark')
fig2 = px.histogram(df_train,x='Profession',color='Profession',template='plotly_dark')
fig2 = px.histogram(df_train,x='Work_Experience',color='Work_Experience',template='plotly_dark')
fig2 = px.histogram(df_train,x='Spending_Score',color='Spending_Score',template='plotly_dark')
fig2 = px.histogram(df_train,x='Family_Size',color='Family_Size',template='plotly_dark')
fig3 = px.histogram(df_train,x='Var_1',color='Var_1',template='plotly_dark')
fig3.show()

mk = LabelEncoder()
df_train['Gender'] = mk.fit_transform(df_train['Gender'])
df_train['Ever_Married'] = mk.fit_transform(df_train['Ever_Married'])
df_train['Graduated'] = mk.fit_transform(df_train['Graduated'])
df_train['Spending_Score'] = mk.fit_transform(df_train['Spending_Score'])
df_train['Var_1'] = mk.fit_transform(df_train['Var_1'])
df_train['Profession'] = mk.fit_transform(df_train['Profession'])
df_train['Family_Size'] = mk.fit_transform(df_train['Family_Size'])
df_train['Work_Experience'] = mk.fit_transform(df_train['Work_Experience'])
df_train['Segmentation'] = mk.fit_transform(df_train['Segmentation'])

scaler = StandardScaler()                                         
df_scaled = scaler.fit_transform(df_train)

wcss_1 = []                                     
range_values = range(1, 10)                    
for i in range_values:                        
  kmeans = KMeans(n_clusters=i)                 
  kmeans.fit(df_scaled)            
  wcss_1.append(kmeans.inertia_)



kmeans = KMeans(n_clusters=5)           
kmeans.fit(df_scaled)         
labels = kmeans.labels_

labels, len(labels)

np.unique(labels, return_counts=True)

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [df_train.columns])
cluster_centers

cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [df_train.columns])
cluster_centers

df_mk_cluster = pd.concat([df_train, pd.DataFrame({'cluster': labels})], axis = 1) 
df_mk_cluster.head()

clusters_count = df_mk_cluster['cluster'].value_counts()                       
clusters_count = clusters_count.to_frame().reset_index()                      
clusters_count.columns = ['clusters', 'count']                               
clusters_count = clusters_count.sort_values('clusters', ascending = True)     

labels = [
        "B", 
        "A", 
        "D", 
        "E",
        "C"
        ]

plt.figure(figsize=(15,9))
mpl.rcParams['font.size'] = 17
colors = sns.color_palette('Set2')[0:5]
plt.pie(clusters_count['count'], 
        explode=(0.05, 0.05, 0.05, 0.05, 0.05), 
        labels = labels,
        colors= colors,
        autopct='%1.1f%%',
        textprops = dict(color ="white", fontsize=19),
        counterclock = False,
        startangle=180,
        wedgeprops={"edgecolor":"gray",'linewidth':1}
        )

plt.axis('equal')
plt.text(-0.8, 1.2, "Clusters", size=30, color="#FFC300", fontweight="bold")
plt.text(-1.8, 1.2, "Distribution", size=30, color="white")
plt.show();



