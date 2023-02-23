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
#from plotly.offline import init_notebook_mode,iplot
from sklearn.model_selection import train_test_split

from flask import Flask,request
from flask_cors import CORS

import matplotlib as mpl
import matplotlib.pyplot as plt
app=Flask(__name__)
CORS(app)

df_train =pd.read_csv("C:/Users/kavimathi/Documents/S8-PROJECT/DATASET/Test.csv")
df_train.head()
df_train.isna().sum(axis=0)
sns.heatmap(df_train.isnull());
df_train.fillna(df_train.mean(), inplace = False)
df_train.dropna(axis=0, inplace=True)
df_train.isna().sum(axis=0)
df_train = df_train.drop(columns=["ID"])
temp = df_train.describe()
temp.style.background_gradient(cmap='Oranges')
g1 = [go.Box(y=df_train.Work_Experience,name="Work_Experience",marker=dict(color="rgba(51,0,0,0.9)"),hoverinfo="name+y")]
g2 = [go.Box(y=df_train.Family_Size,name="Family_Size",marker=dict(color="rgba(0,102,102,0.9)"),hoverinfo="name+y")]
layout2 = go.Layout(title="Work Experience | Family Size",yaxis=dict(range=[0,13])) 
fig2 = go.Figure(data=g1+g2,layout=layout2)
#iplot(fig2)
grafico = px.box(df_train, y='Age')
grafico.show()
fig2 = px.histogram(df_train,x='Age',color='Age',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Gender',color='Gender',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Ever_Married',color='Ever_Married',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Graduated',color='Graduated',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Profession',color='Profession',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Work_Experience',color='Work_Experience',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Spending_Score',color='Spending_Score',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Family_Size',color='Family_Size',template='plotly_dark')
fig2.show()
fig2 = px.histogram(df_train,x='Var_1',color='Var_1',template='plotly_dark')
fig2.show()

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
df_train
scaler = StandardScaler()                                         
df_scaled = scaler.fit_transform(df_train)
type(df_scaled)
min(df_scaled[0]), max(df_scaled[0])
df_scaled
wcss_1 = []                                     
range_values = range(1, 10)                    
for i in range_values:                        
        kmeans = KMeans(n_clusters=i)                 
        kmeans.fit(df_scaled)            
        wcss_1.append(kmeans.inertia_)
        print(wcss_1)
        grafico = px.line(x = range(1,10), y = wcss_1)
plt.plot(wcss_1, '-o',)  
grafico.show()
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
sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.grid' : False, 'font.family': 'Ubuntu'})

for i in df_mk_cluster:
        g = sns.FacetGrid(df_mk_cluster, col = "cluster", hue = "cluster", palette = "Set2")
        g.map(plt.hist, i, bins=10, ec="k") 
        g.set_xticklabels(rotation=30, color = 'white')
        g.set_yticklabels(color = 'white')
        g.set_xlabels(size=15, color = 'white')
        g.set_titles(size=15, color = '#FFC300', fontweight="bold")
        g.fig.set_figheight(5);
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
df_scaled
from scipy.cluster.hierarchy import dendrogram, linkage
dendrograma = dendrogram(linkage(df_scaled, method='ward'))
plt.title('Dendrograma')
plt.xlabel('X')
plt.ylabel('Y');
from sklearn.cluster import AgglomerativeClustering
hc_g = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage = 'ward')
rotulos = hc_g.fit_predict(df_scaled)
rotulos
grafico = px.scatter(x = df_scaled[:,0], y = df_scaled[:,1], color = rotulos)
grafico.show()
df_scaled
from sklearn.cluster import DBSCAN
dbscan_g = DBSCAN(eps = 0.95, min_samples=2)
dbscan_g.fit(df_train)
rotulos = dbscan_g.labels_
rotulos
grafico = px.scatter(x = df_scaled[:,0], y = df_scaled[:,1], color = rotulos)
grafico.show()
from sklearn import datasets
X_random, y_random = datasets.make_moons(n_samples=1500, noise = 0.09)
X_random
grafico = px.scatter(x = X_random[:,0], y = X_random[:,1])
grafico.show()
kmeans = KMeans(n_clusters=2)
rotulos = kmeans.fit_predict(X_random)
grafico = px.scatter(x = X_random[:,0], y = X_random[:, 1], color = rotulos)
grafico.show()
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
rotulos = hc.fit_predict(X_random)
grafico = px.scatter(x = X_random[:,0], y = X_random[:, 1], color = rotulos)
grafico.show()
dbscan = DBSCAN(eps=0.1)
rotulos = dbscan.fit_predict(X_random)
grafico = px.scatter(x = X_random[:,0], y = X_random[:, 1], color = rotulos)
grafico.show()


