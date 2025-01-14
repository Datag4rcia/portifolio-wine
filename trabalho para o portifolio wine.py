#!/usr/bin/env python
# coding: utf-8

# In[258]:


#baixando as bibliotecas necessárias
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install matplotlib')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install ploty.express')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scipy')
get_ipython().system('pip install plotly')
#importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as mtp
import seaborn as sns
import numpy as np 
import plotly.express as px
import scipy as sc
import plotly as px


# In[259]:


#carregando o dataset
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()

wine


# In[260]:


#transformando o dataset em tabela
wine= pd.DataFrame(wine.data, columns=wine.feature_names)
print(wine)


# In[261]:


#informações da tabela
wine.info()
#estatistica básica
wine.describe()


# In[262]:


#renomeação das colunas
wine=wine.rename(columns={'alcalinity_of_ash':'ph','nonflavanoid_phenols':'nonflavanoid','od280/od315_of_diluted_wines':'id_of_wine'})
wine.head()


# In[263]:


#visualização de informações gerais
import warnings
warnings.filterwarnings('ignore')
sns.pairplot(wine)


# In[264]:


#visualização das distribuição das variáveis

mtp.figure(figsize=(10, 6))

sns.boxplot(data=wine)

mtp.title('Distribuição das Variáveis - Gráfico de Caixa')


mtp.show()


# In[266]:


#Separando as variáveis de interesse
wine_sub = wine[['alcohol', 'color_intensity', 'id_of_wine', 'ph']]
wine_sub.columns = ['alcohol', 'color_intensity', 'id_of_wine', 'ph']
wine_sub.head()


# In[267]:


#correlação
wine_sub.corr()


# In[269]:


from scipy import stats
# Média
mean = np.mean(wine_sub, axis=0)  # Média de cada variável (coluna)
print(f"Média: \n{mean}")

# Variância
variance = np.var(wine_sub, axis=0)  # Variância de cada variável
print(f"Variância: \n{variance}")

# Desvio padrão
std_dev = np.std(wine_sub, axis=0)  # Desvio padrão de cada variável
print(f"Desvio Padrão: \n{std_dev}")

#Moda
mode = stats.mode(wine_sub)

print(f"A moda é: {mode.mode[0]} com uma frequência de {mode.count[0]}")

#normalização
from scipy.stats import zscore
normalized_df = wine_sub.apply(zscore)

print("Tabela Normalizada (Z-Score):")
print(normalized_df)

# Teste t de Student
t_stat, p_value = stats.ttest_ind(wine_sub['id_of_wine'], wine_sub['ph'])

print(f"Estatística t: {t_stat}")
print(f"Valor p: {p_value}")


# In[271]:


#visualização da dispersão entre os dados da tabela de interesse
import matplotlib.pyplot as mtp
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# Criando a figura 3D
fig = mtp.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Gráfico de dispersão
scatter = ax.scatter(
    wine_sub['alcohol'],            # Eixo X
    wine_sub['color_intensity'],    # Eixo Y
    wine_sub['ph'],                # Eixo Z
    c=wine_sub['id_of_wine'],       # Cor baseada na 4ª variável
    cmap='viridis',                 # Paleta de cores
    s=100,                          # Tamanho dos pontos
    alpha=0.8                       # Opacidade
)

# Adicionando rótulos
ax.set_title("Gráfico de Dispersão 3D com Matplotlib", fontsize=16)
ax.set_xlabel("Álcool", fontsize=12)
ax.set_ylabel("Intensidade de Cor", fontsize=12)
ax.set_zlabel("Ph", fontsize=12)

# Adicionando a barra de cores
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('ID do Vinho', fontsize=12)

# Exibir o gráfico
mtp.show()


# In[272]:


#visualizando dos dados de correlação
import seaborn as sns
import matplotlib.pyplot as plt

correlacao=wine_sub.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")

plt.title('Matriz de Correlação das Variáveis')

plt.show()
fig.show()


# In[273]:


#preparação dos dados
def definirnomes (target):
    if(target== 0):
        return 'tipo de uva 1'
    elif (target== 1):
        return 'tipo de uva 2'
    elif (target == 2):
        return 'tipo de uva 3'


# In[274]:


#clusterização
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances


#clusterização para 3

kmeans3 = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans3.labels_
metrics.silhouette_score(X, labels, metric='euclidean')


#clusterização para 4

kmeans4 = KMeans(n_clusters=4, random_state=1).fit(X)
labels = kmeans4.labels_
metrics.silhouette_score(X, labels, metric='euclidean')

#clusterização para 5

kmeans5 = KMeans(n_clusters=5, random_state=1).fit(X)
labels = kmeans5.labels_
metrics.silhouette_score(X, labels, metric='euclidean')

#clusterização para 6

kmeans6 = KMeans(n_clusters=6, random_state=1).fit(X)
labels = kmeans6.labels_
metrics.silhouette_score(X, labels, metric='euclidean')


# In[275]:


#visualização grafica
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

# Clusterização para k=3
# Ensure X is a NumPy array
if isinstance(X, pd.DataFrame):
    X = X.to_numpy()

# Plotting clusters
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]  # Assuming `labels` is a NumPy array
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

# Plotting centroids
centroids = kmeans3.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

plt.legend()
plt.title('Clustering Visualization')

plt.show()



# In[276]:


kmeans4 = KMeans(n_clusters=4, random_state=1).fit(X)
labels = kmeans4.labels_
metrics.silhouette_score(X, labels, metric='euclidean')
#para k=4
if isinstance(X, pd.DataFrame):
    X = X.to_numpy()

# Visualizando os clusters
plt.figure(figsize=(8, 6))
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

# Plotando os centróides
centroids = kmeans4.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

# Configurando o gráfico
plt.legend()
plt.title('Clustering Visualization')
plt.show()


# In[277]:


kmeans5 = KMeans(n_clusters=5, random_state=1).fit(X)
labels = kmeans5.labels_
metrics.silhouette_score(X, labels, metric='euclidean')

#para k=5

if isinstance(X, pd.DataFrame):
    X = X.to_numpy()

# Visualizando os clusters
plt.figure(figsize=(8, 6))
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

# Plotando os centróides
centroids = kmeans4.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

# Configurando o gráfico
plt.legend()
plt.title('Clustering Visualization')
plt.show()
plt.show()


# In[278]:


kmeans6 = KMeans(n_clusters=6, random_state=1).fit(X)
labels = kmeans6.labels_
metrics.silhouette_score(X, labels, metric='euclidean')

#para k=6

if isinstance(X, pd.DataFrame):
    X = X.to_numpy()

# Visualizando os clusters
plt.figure(figsize=(8, 6))
for cluster in np.unique(labels):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

# Plotando os centróides
centroids = kmeans4.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

# Configurando o gráfico
plt.legend()
plt.title('Clustering Visualization')
plt.show()


# In[279]:


#aplicação do metódo Elbow method
valores_k= []
inercia= []

for i in range (1,10):
    kmeansi= KMeans(n_clusters=i, random_state=0)
    kmeansi.fit(X)
    labels = kmeansi.labels_
    centroids = kmeansi.cluster_centers_
    valores_k.append(i)
    inercia.append(kmeansi.inertia_)
    print(kmeansi.inertia_)


# In[280]:


#visualização da relação entre a inercia e k
fig, ax= plt.subplots()

ax.plot(valores_k,inercia)

plt.show()


# In[281]:


#visualizando k=3 e k=4 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Clusterização para 4 clusters
kmeans4 = KMeans(n_clusters=4, random_state=1).fit(X)
labels4 = kmeans4.labels_
centroids4 = kmeans4.cluster_centers_

# Clusterização para 3 clusters
kmeans3 = KMeans(n_clusters=3, random_state=1).fit(X)
labels3 = kmeans3.labels_
centroids3 = kmeans3.cluster_centers_

# Cálculo do Silhouette Score
silhouette_avg_4 = silhouette_score(X, labels4, metric='euclidean')
silhouette_avg_3 = silhouette_score(X, labels3, metric='euclidean')

# Criando subgráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico para 4 clusters
for cluster in np.unique(labels4):
    cluster_points = X[labels4 == cluster]
    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
ax1.scatter(centroids4[:, 0], centroids4[:, 1], s=200, c='red', marker='X', label='Centroids')
ax1.set_title(f'Clusterização com KMeans (4 clusters)\nSilhouette Score: {silhouette_avg_4:.2f}')
ax1.legend()

# Gráfico para 3 clusters
for cluster in np.unique(labels3):
    cluster_points = X[labels3 == cluster]
    ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
ax2.scatter(centroids3[:, 0], centroids3[:, 1], s=200, c='red', marker='X', label='Centroids')
ax2.set_title(f'Clusterização com KMeans (3 clusters)\nSilhouette Score: {silhouette_avg_3:.2f}')
ax2.legend()

# Exibindo os gráficos
plt.tight_layout()
plt.show()


# In[282]:


#definindo qual é a melhor opção 

from sklearn.metrics import davies_bouldin_score
#para k=3
silhouette_avg = silhouette_score(X, kmeans3.labels_)
print(f"Silhouette Score: {silhouette_avg}")
#para k=4
silhouette_avg = silhouette_score(X, kmeans4.labels_)
print(f"Silhouette Score: {silhouette_avg}")
#para k=5
silhouette_avg = silhouette_score(X, kmeans5.labels_)
print(f"Silhouette Score: {silhouette_avg}")
#para k=6
silhouette_avg = silhouette_score(X, kmeans6.labels_)
print(f"Silhouette Score: {silhouette_avg}")


# In[283]:


valores_k= []
s= []

for i in range (2,10):
    kmeansi= KMeans(n_clusters=i, random_state=0)
    kmeansi.fit(X)
    labels = kmeansi.labels_
    centroids = kmeansi.cluster_centers_
    valores_k.append(i)
    s.append(silhouette_score(X, kmeansi.labels_))

#visualização da relação entre a s e k
fig, ax= plt.subplots()

ax.plot(valores_k,s)

plt.show()    


# In[286]:


import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px

# Carregar dados
data =wine_sub

# Ajustar colunas para as especificadas
# Supondo que "id_of_wine" seja um identificador único
if 'id_of_wine' not in data.columns:
    data['id_of_wine'] = range(1, len(data) + 1)

# Renomear colunas para garantir compatibilidade
if 'color intensity' not in data.columns:
    data.rename(columns={'color_intensity': 'color intensity'}, inplace=True)

# Selecionar colunas
data = data[['alcohol', 'color intensity', 'id_of_wine', 'ph']]  # Usar apenas as variáveis especificadas

# Inicializar o app
app = dash.Dash(__name__)
app.title = "Wine Dashboard"

# Estilo padrão com cor vinho
wine_color = '#722f37'
app.layout = html.Div([
    html.H1("Wine Dashboard", style={'textAlign': 'center', 'color': wine_color}),

    # Elemento 1: Gráfico de dispersão
    html.Div([
        dcc.Graph(id='scatter-plot'),
        html.Label("Escolha a variável X para o gráfico de dispersão:", style={'color': wine_color}),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in data.columns if col != 'id_of_wine'],
            value='alcohol',
            clearable=False,
            style={'backgroundColor': wine_color, 'color': 'white'}
        ),
        html.Label("Escolha a variável Y para o gráfico de dispersão:", style={'color': wine_color}),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in data.columns if col != 'id_of_wine'],
            value='color intensity',
            clearable=False,
            style={'backgroundColor': wine_color, 'color': 'white'}
        )
    ], style={'margin': '20px'}),

    # Elemento 2: Gráfico de barras
    html.Div([
        dcc.Graph(id='bar-chart'),
        html.Label("Distribuição da intensidade da cor do vinho", style={'color': wine_color})
    ], style={'margin': '20px'}),

    # Elemento 3: Tabela interativa
    html.Div([
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": col, "id": col} for col in data.columns],
            data=data.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': wine_color, 'color': 'white'},
            style_data={'backgroundColor': '#f7f7f7', 'color': wine_color}
        ),
    ], style={'margin': '20px'}),

    # Elemento 4: Controle deslizante
    html.Div([
        html.Label("Filtrar vinhos por intensidade de cor:", style={'color': wine_color}),
        dcc.Slider(
            id='color-intensity-slider',
            min=data['color intensity'].min(),
            max=data['color intensity'].max(),
            step=0.1,
            value=data['color intensity'].mean(),
            marks={round(ci, 1): str(round(ci, 1)) for ci in data['color intensity'].unique()[:10]},
            tooltip={'placement': 'bottom', 'always_visible': True}
        )
    ], style={'margin': '20px'}),

    # Elemento 5: Gráfico de pizza
    html.Div([
        dcc.Graph(id='pie-chart'),
        html.Label("Distribuição do Álcool", style={'color': wine_color})
    ], style={'margin': '20px'})
], style={'backgroundColor': '#f9f6f6'})

# Callback para o gráfico de dispersão
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'), Input('y-axis', 'value')]
)
def update_scatter(x_axis, y_axis):
    fig = px.scatter(data, x=x_axis, y=y_axis, color='ph',
                     title=f'{x_axis} vs {y_axis}',
                     color_continuous_scale='reds')
    fig.update_layout(plot_bgcolor='#f9f6f6', paper_bgcolor='#f9f6f6', font_color=wine_color)
    return fig

# Callback para o gráfico de barras
@app.callback(
    Output('bar-chart', 'figure'),
    Input('color-intensity-slider', 'value')
)
def update_bar_chart(color_intensity_value):
    filtered_data = data[data['color intensity'] >= color_intensity_value]
    fig = px.bar(
        filtered_data, x='id_of_wine', y='color intensity',
        title=f'Intensidade da cor para valores >= {color_intensity_value:.1f}',
        labels={"id_of_wine": "ID do Vinho", "color intensity": "Intensidade da Cor"},
        color='color intensity',
        color_continuous_scale='reds'
    )
    fig.update_layout(plot_bgcolor='#f9f6f6', paper_bgcolor='#f9f6f6', font_color=wine_color)
    return fig

# Callback para o gráfico de pizza
@app.callback(
    Output('pie-chart', 'figure'),
    Input('color-intensity-slider', 'value')
)
def update_pie_chart(color_intensity_value):
    filtered_data = data[data['color intensity'] >= color_intensity_value]
    fig = px.pie(
        filtered_data, values='alcohol', names='id_of_wine',
        title='Distribuição do Álcool por Vinho',
        color_discrete_sequence=px.colors.sequential.Reds
    )
    fig.update_layout(plot_bgcolor='#f9f6f6', paper_bgcolor='#f9f6f6', font_color=wine_color)
    return fig

# Rodar o app
if __name__ == '__main__':
    app.run_server(debug=True)



# In[ ]:




