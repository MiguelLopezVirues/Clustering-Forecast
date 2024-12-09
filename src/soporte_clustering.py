# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otras utilidades
# -----------------------------------------------------------------------
import math

# Para las visualizaciones
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesado y modelado
# -----------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler, MinMaxScaler


# Sacar número de clusters y métricas
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Modelos de clustering
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Para visualizar los dendrogramas
# -----------------------------------------------------------------------
import scipy.cluster.hierarchy as sch

from typing import Optional


def plot_3D_clusters(dataframe: pd.DataFrame) -> None:
    """
    Creates a 3D scatter plot for PCA-transformed clusters.

    Args:
        dataframe (pd.DataFrame): DataFrame containing three columns representing PCA-transformed dimensions.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the PCA-transformed data
    ax.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], dataframe.iloc[:, 2], c='b', s=50, alpha=0.6, edgecolor='k')
    
    # Labels and title
    ax.set_title('3D PCA Clusters')
    ax.set_xlabel(f"{dataframe.columns[0]}")
    ax.set_ylabel(f"{dataframe.columns[1]}")
    ax.set_zlabel(f"{dataframe.columns[2]}")

    plt.show()

def calculate_discount_expense_correlation(grupo: pd.DataFrame) -> float:
    """
    Calculates the Spearman correlation between discounted price and sales if discounts vary.

    Args:
        grupo (pd.DataFrame): DataFrame containing columns 'discount', 'discounted_price', and 'sales'.

    Returns:
        float: Spearman correlation value if discounts vary; otherwise 0.
    """
    if len(grupo['discount'].unique()) > 1:  # if there's variation in discounts
        return grupo['discount'].corr(grupo['sales'], method="spearman")
    return 0

def choose_kmeans_k(dataframe: pd.DataFrame, random_state: int = 99) -> None:
    """
    Determines optimal number of clusters for KMeans using various metrics.

    Args:
        dataframe (pd.DataFrame): DataFrame containing features for clustering.
        random_state (int): Random state for KMeans reproducibility. Defaults to 99.

    Returns:
        None
    """
    k_range = range(2, 16)
    inertia = []
    silhouette = []
    calinski_harabasz = []
    davies_bouldin = []

    X = dataframe

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X, kmeans.labels_))
        calinski_harabasz.append(calinski_harabasz_score(X, kmeans.labels_))
        davies_bouldin.append(davies_bouldin_score(X, kmeans.labels_))

    plt.figure(figsize=(15, 10))

    # Inertia Plot
    plt.subplot(2, 2, 1)
    plt.plot(k_range, inertia, marker='o')
    plt.title('Inertia')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    # Silhouette Score
    plt.subplot(2, 2, 2)
    plt.plot(k_range[1:], silhouette[1:], marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    # Calinski-Harabasz Index
    plt.subplot(2, 2, 3)
    plt.plot(k_range[1:], calinski_harabasz[1:], marker='o')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')

    # Davies-Bouldin Index
    plt.subplot(2, 2, 4)
    plt.plot(k_range[1:], davies_bouldin[1:], marker='o')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()

def perform_PCA(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Performs PCA on a DataFrame and plots explained variance.

    Args:
        dataframe (pd.DataFrame): DataFrame containing features to perform PCA on.

    Returns:
        pd.DataFrame: Transformed DataFrame after applying PCA.
    """
    pca = PCA()
    X_pca = pd.DataFrame(pca.fit_transform(dataframe))

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='Individual Variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance', color='orange')
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    return X_pca

def t_sne(dataframe: pd.DataFrame, dim: int = 2, perplexity: int = 5) -> Optional[np.ndarray]:
    """
    Performs t-SNE dimensionality reduction and visualizes the results.

    Args:
        dataframe (pd.DataFrame): DataFrame containing features for t-SNE.
        dim (int): Target dimensionality for t-SNE (2 or 3). Defaults to 2.
        perplexity (int): Perplexity parameter for t-SNE. Defaults to 5.

    Returns:
        Optional[np.ndarray]: t-SNE transformed array if dim is specified.
    """
    tsne = TSNE(n_components=dim, random_state=42, perplexity=perplexity, max_iter=1000)
    X_tsne = tsne.fit_transform(dataframe)

    if dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='b', alpha=0.6, edgecolor='k', s=50)
        plt.title(f't-SNE Visualization, perplexity {perplexity}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid()
        plt.show()
    elif dim == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c='b', alpha=0.6, edgecolor='k', s=50)
        ax.set_title(f't-SNE Visualization in 3D, perplexity {perplexity}')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        plt.show()

    return X_tsne



class Exploracion:
    """
    Clase para realizar la exploración y visualización de datos en un DataFrame.

    Atributos:
    dataframe : pd.DataFrame
        El conjunto de datos a ser explorado y visualizado.
    """

    def __init__(self, dataframe):
        """
        Inicializa la clase Exploracion con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a ser explorados.
        """
        self.dataframe = dataframe
    
    def explorar_datos(self):
        """
        Realiza un análisis exploratorio de un DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        print("5 registros aleatorios:")
        display(self.dataframe.sample(5))
        print("\n")

        print("Información general del DataFrame:")
        print(self.dataframe.info())
        print("\n")

        print("Duplicados en el DataFrame:")
        print(self.dataframe.duplicated().sum())
        print("\n")

        print("Estadísticas descriptivas de las columnas numéricas:")
        display(self.dataframe.describe().T)
        print("\n")

        print("Estadísticas descriptivas de las columnas categóricas:")
        categorical_columns = self.dataframe.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            display(self.dataframe[categorical_columns].describe().T)
        else:
            print("No hay columnas categóricas en el DataFrame.")
        print("\n")
        
        print("Número de valores nulos por columna:")
        print(self.dataframe.isnull().sum())
        print("\n")
        
        if len(categorical_columns) > 0:
            print("Distribución de valores categóricos:")
            for col in categorical_columns:
                print(f"\nColumna: {col}")
                print(self.dataframe[col].value_counts())
        
        print("Matriz de correlación entre variables numéricas:")
        display(self.dataframe.corr(numeric_only=True))
        print("\n")

    def visualizar_numericas(self):
        """
        Genera histogramas, boxplots y gráficos de dispersión para las variables numéricas del DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        columns = self.dataframe.select_dtypes(include=np.number).columns

        # Histogramas
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/2), ncols=2, figsize=(21, 13))
        axes = axes.flat
        plt.suptitle("Distribución de las variables numéricas", fontsize=24)
        for indice, columna in enumerate(columns):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], kde=True, color="#F2C349")

        if len(columns) % 2 != 0:
            fig.delaxes(axes[-1])

        plt.tight_layout()

        # Boxplots
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/2), ncols=2, figsize=(19, 11))
        axes = axes.flat
        plt.suptitle("Boxplots de las variables numéricas", fontsize=24)
        for indice, columna in enumerate(columns):
            sns.boxplot(x=columna, data=self.dataframe, ax=axes[indice], color="#F2C349", flierprops={'markersize': 4, 'markerfacecolor': 'cyan'})
        if len(columns) % 2 != 0:
            fig.delaxes(axes[-1])
        plt.tight_layout()
    
    def visualizar_categoricas(self):
        """
        Genera gráficos de barras (count plots) para las variables categóricas del DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        categorical_columns = self.dataframe.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) > 0:
            try:
                _, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(15, 5 * len(categorical_columns)))
                axes = axes.flat
                plt.suptitle("Distribución de las variables categóricas", fontsize=24)
                for indice, columna in enumerate(categorical_columns):
                    sns.countplot(data=self.dataframe, x=columna, ax=axes[indice])
                    axes[indice].set_title(f'Distribución de {columna}', fontsize=20)
                    axes[indice].set_xlabel(columna, fontsize=16)
                    axes[indice].set_ylabel('Conteo', fontsize=16)
                plt.tight_layout()
            except: 
                sns.countplot(data=self.dataframe, x=categorical_columns[0])
                plt.title(f'Distribución de {categorical_columns[0]}', fontsize=20)
                plt.xlabel(categorical_columns[0], fontsize=16)
                plt.ylabel('Conteo', fontsize=16)
        else:
            print("No hay columnas categóricas en el DataFrame.")

    def visualizar_categoricas_numericas(self, chart_type="scatter"):
        """
        Genera gráficos de dispersión para las variables numéricas vs todas las variables categóricas.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        categorical_columns = self.dataframe.select_dtypes(include=['object', 'category']).columns
        numerical_columns = self.dataframe.select_dtypes(include=np.number).columns
        if len(categorical_columns) > 0:
            for num_col in numerical_columns:
                try:
                    _, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 5 * len(categorical_columns)))
                    axes = axes.flat
                    plt.suptitle(f'Dispersión {num_col} vs variables categóricas', fontsize=24)
                    for indice, cat_col in enumerate(categorical_columns):
                        axes[indice].set_title(f"{num_col} vs {cat_col}")
                        if chart_type == "scatter":
                            sns.scatterplot(x=num_col, y=self.dataframe.index, hue=cat_col, data=self.dataframe, ax=axes[indice])
                            axes[indice].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
                        else:
                            sns.barplot(data=self.dataframe,
                                        x=cat_col,
                                        y=num_col,
                                        hue=cat_col,ax=axes[indice])
                        axes[indice].set_xlabel(num_col, fontsize=16)
                        axes[indice].set_ylabel(cat_col, fontsize=16)

                    plt.tight_layout()
                except: 
                    axes[indice].set_title(f"{num_col} vs {categorical_columns[0]}")
                    if chart_type == "scatter":
                        sns.scatterplot(x=num_col, y=self.dataframe.index, hue=categorical_columns[0], data=self.dataframe)
                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=10)
                    else:
                        sns.barplot(data=self.dataframe,
                            x=categorical_columns[0],
                            y=num_col,
                            hue=categorical_columns[0],ax=axes[indice])
                        

                    plt.xlabel(num_col, fontsize=16)
                    plt.ylabel('Índice', fontsize=16)
        else:
            print("No hay columnas categóricas en el DataFrame.")

    def correlacion(self, metodo="pearson", tamanio=(14, 8)):
        """
        Genera un heatmap de la matriz de correlación de las variables numéricas del DataFrame.

        Params:
            - metodo : str, optional, default: "pearson". Método para calcular la correlación.
            - tamanio : tuple of int, optional, default: (14, 8). Tamaño de la figura del heatmap.

        Returns:
            - None.
        """
        plt.figure(figsize=tamanio)
        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype=np.bool_))
        sns.heatmap(self.dataframe.corr(numeric_only=True, method=metodo), annot=True, cmap='viridis', vmax=1, vmin=-1, mask=mask)
        plt.title("Correlación de las variables numéricas", fontsize=24)



class Preprocesado:
    """
    Clase para realizar preprocesamiento de datos en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos a ser preprocesado.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa la clase Preprocesado con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a ser preprocesados.
        """
        self.dataframe = dataframe

    def estandarizar(self):
        """
        Estandariza las columnas numéricas del DataFrame.

        Este método ajusta y transforma las columnas numéricas del DataFrame utilizando `StandardScaler` para que
        tengan media 0 y desviación estándar 1.

        Returns:
            - pd.DataFrame. El DataFrame con las columnas numéricas estandarizadas.
        """
        # Sacamos el nombre de las columnas numéricas
        col_numericas = self.dataframe.select_dtypes(include=np.number).columns

        # Inicializamos el escalador para estandarizar los datos
        scaler = StandardScaler()

        # Ajustamos los datos y los transformamos
        X_scaled = scaler.fit_transform(self.dataframe[col_numericas])

        # Sobreescribimos los valores de las columnas en el DataFrame
        self.dataframe[col_numericas] = X_scaled

        return self.dataframe
    
    def codificar(self):
        """
        Codifica las columnas categóricas del DataFrame.

        Este método reemplaza los valores de las columnas categóricas por sus frecuencias relativas dentro de cada
        columna.

        Returns:
            - pd.DataFrame. El DataFrame con las columnas categóricas codificadas.
        """
        # Sacamos el nombre de las columnas categóricas
        col_categoricas = self.dataframe.select_dtypes(include=["category", "object"]).columns

        # Iteramos por cada una de las columnas categóricas para aplicar el encoding
        for categoria in col_categoricas:
            # Calculamos las frecuencias de cada una de las categorías
            frecuencia = self.dataframe[categoria].value_counts(normalize=True)

            # Mapeamos los valores obtenidos en el paso anterior, sobreescribiendo la columna original
            self.dataframe[categoria] = self.dataframe[categoria].map(frecuencia)

        return self.dataframe


class Clustering:
    """
    Clase para realizar varios métodos de clustering en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos sobre el cual se aplicarán los métodos de clustering.
    """
    
    def __init__(self, dataframe, scaled=False):
        """
        Inicializa la clase Clustering con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a los que se les aplicarán los métodos de clustering.
        """
        self.dataframe = dataframe.copy()
        self.dataframe_encoded = dataframe.copy()
        self.dataframe_escalado = None
        self.labels = {
            "kmeans": None,
            "agglomerative": None,
            "divisive": None,
            "spectral": None,
            "DBSCAN": None
        }

    def encodear_dataframe(self):
        encoder = OrdinalEncoder()
        categoricas = self.dataframe.select_dtypes(["category","object"]).columns.to_list()
        self.dataframe_encoded[categoricas] = encoder.fit_transform(self.dataframe[categoricas])
    
    def escalar_dataframe(self, scaler="standard"):
        scalers = {
            "standard":StandardScaler(),
            "minmax":MinMaxScaler(),
            "robust": RobustScaler()
        }

        self.dataframe_escalado = pd.DataFrame(scalers[scaler].fit_transform(self.dataframe_encoded), 
                                               columns=self.dataframe_encoded.columns.to_list())

    def preparar_dataframe(self, scaler="standard"):
        self.encodear_dataframe()
        self.escalar_dataframe(scaler)

    
    def sacar_clusters_kmeans(self, n_clusters=(2, 15), metric="calinski_harabasz"):
        """
        Utiliza KMeans y KElbowVisualizer para determinar el número óptimo de clusters basado en la métrica de silhouette.

        Params:
            - n_clusters : tuple of int, optional, default: (2, 15). Rango de número de clusters a probar.
        
        Returns:
            None
        """
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=n_clusters, metric=metric)
        visualizer.fit(self.dataframe_escalado)
        visualizer.show()
    
    def modelo_kmeans(self, dataframe_original, num_clusters, random_state=42):
        """
        Aplica KMeans al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - num_clusters : int. Número de clusters a formar.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        km_fit = kmeans.fit(self.dataframe_escalado)
        self.labels["kmeans"] = km_fit.labels_
        dataframe_original["clusters_kmeans"] = self.labels["kmeans"].astype(str)
        return dataframe_original
    
    def visualizar_dendrogramas(self, lista_metodos=["average", "complete", "ward"]):
        """
        Genera y visualiza dendrogramas para el conjunto de datos utilizando diferentes métodos de distancias.

        Params:
            - lista_metodos : list of str, optional, default: ["average", "complete", "ward"]. Lista de métodos para calcular las distancias entre los clusters. Cada método generará un dendrograma
                en un subplot diferente.

        Returns:
            None
        """
        _, axes = plt.subplots(nrows=1, ncols=len(lista_metodos), figsize=(20, 8))
        axes = axes.flat

        for indice, metodo in enumerate(lista_metodos):
            sch.dendrogram(sch.linkage(self.dataframe_escalado, method=metodo),
                           labels=self.dataframe_escalado.index, 
                           leaf_rotation=90, leaf_font_size=4,
                           ax=axes[indice])
            axes[indice].set_title(f'Dendrograma usando {metodo}')
            axes[indice].set_xlabel('Muestras')
            axes[indice].set_ylabel('Distancias')

    def modelo_aglomerativo_tabla(self, linkage_methods, distance_metrics, silhouette_distance="euclidean"):

        # Crear un DataFrame para almacenar los resultados
        results = []

        # Suponiendo que tienes un DataFrame llamado df_copia
        # Aquí df_copia debería ser tu conjunto de datos
        # Asegúrate de que esté preprocesado adecuadamente (normalizado si es necesario)

        for linkage_method in linkage_methods:
            for metric in distance_metrics:
                for cluster in range(2,6):
                    try:
                        # Configurar el modelo de AgglomerativeClustering
                        modelo = AgglomerativeClustering(
                            linkage=linkage_method,
                            metric=metric,  
                            distance_threshold=None,  # Para buscar n_clusters
                            n_clusters=cluster, # Cambia esto según tu análisis. Si tienen valor, el distance threshold puede ser none, y viceversa.
                        )
                        
                        # Ajustar el modelo
                        labels = modelo.fit_predict(self.dataframe_escalado) #etiquetas del cluster al que pertenece

                        # Calcular métricas si hay más de un cluster
                        if len(np.unique(labels)) > 1:
                            # Silhouette Score
                            silhouette_avg = silhouette_score(self.dataframe_escalado, labels, metric=silhouette_distance)

                            # Davies-Bouldin Index
                            db_score = davies_bouldin_score(self.dataframe_escalado, labels)

                            
                            # Cardinalidad (tamaño de cada cluster)
                            cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}
                        else:
                            cluster_cardinality = {'Cluster único': len(self.dataframe_escalado)}

                        # Almacenar resultados
                        results.append({
                            'linkage': linkage_method,
                            'metric': metric,
                            'silhouette_score': silhouette_avg,
                            'davies_bouldin_index': db_score,
                            'cluster_cardinality': cluster_cardinality,
                            'n_cluster': cluster
                        })

                    except Exception as e:
                        print(f"Error con linkage={linkage_method}, metric={metric}: {e}")
                        # Ward SÓLO acepta medir con distancia euclidiana en scikitlearn.

        # Crear DataFrame de resultados
        results_df = pd.DataFrame(results)

        # Mostrar resultados ordenados por silhouette_score
        results_df = results_df.sort_values(by='silhouette_score', ascending=False)

        # Mostrar el DataFrame
        return results_df
    
    def plot_clusters(self, columna_cluster, dataframe=None):
        # columnas_plot = df_cluster.columns.drop(columna_cluster)
        if dataframe is None:
            df_cluster = self.dataframe.select_dtypes(np.number).copy()
        else:
            df_cluster = dataframe.select_dtypes(np.number).copy()
        columnas_plot = df_cluster.columns

 
        df_cluster[columna_cluster] = self.labels[columna_cluster]

        

        ncols = math.ceil(12 / df_cluster[columna_cluster].nunique())
        nrows = math.ceil(len(columnas_plot) / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 8))
        axes = axes.flat


        for indice, columna in enumerate(columnas_plot):
            df_group = df_cluster.groupby(columna_cluster)[columna].mean().reset_index()
            sns.barplot(x=columna_cluster, y=columna, data=df_group, ax=axes[indice], palette="coolwarm")
            axes[indice].set_title(columna)  

        if len(columnas_plot) % 2 == 1: 
            fig.delaxes(axes[-1]) 

        plt.tight_layout()
        plt.show() 

    def radar_plot(self, metodo="kmeans"):
        variables = self.dataframe.select_dtypes(np.number).columns

        radar = self.dataframe_escalado
        radar[metodo] = self.labels[metodo]

        # Agrupar por cluster y calcular la media
        cluster_means = radar.groupby(metodo)[variables].mean()

        # Repetir la primera columna al final para cerrar el radar
        cluster_means = pd.concat([cluster_means, cluster_means.iloc[:, 0:1]], axis=1)

        # Crear los ángulos para el radar plot
        num_vars = len(variables)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el gráfico

        # Crear el radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Dibujar un gráfico para cada cluster
        for i, row in cluster_means.iterrows():
            ax.plot(angles, row, label=f'Cluster {i}')
            ax.fill(angles, row, alpha=0.25)

        # Configurar etiquetas de los ejes
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables)

        # Añadir leyenda y título
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Radar Plot de los Clusters', size=16)
        plt.show()
    
    def modelo_aglomerativo(self, num_clusters, metodo_distancias, dataframe_original):
        """
        Aplica clustering aglomerativo al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - num_clusters : int. Número de clusters a formar.
            - metodo_distancias : str. Método para calcular las distancias entre los clusters.
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        modelo = AgglomerativeClustering(
            linkage=metodo_distancias,
            distance_threshold=None,
            n_clusters=num_clusters
        )
        aglo_fit = modelo.fit(self.dataframe_escalado)
        self.labels["agglomerative"] = aglo_fit.labels_
        dataframe_original["clusters_agglomerative"] = self.labels["agglomerative"].astype(str)
        return dataframe_original
    
    
    def modelo_divisivo(self, dataframe_original, threshold=0.5, max_clusters=5):
        """
        Implementa el clustering jerárquico divisivo.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - threshold : float, optional, default: 0.5. Umbral para decidir cuándo dividir un cluster.
            - max_clusters : int, optional, default: 5. Número máximo de clusters deseados.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de los clusters.
        """
        def divisive_clustering(data, current_cluster, cluster_labels):
            # Si el número de clusters actuales es mayor o igual al máximo permitido, detener la división
            if len(set(current_cluster)) >= max_clusters:
                return current_cluster

            # Aplicar KMeans con 2 clusters
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(data)
            labels = kmeans.labels_

            # Calcular la métrica de silueta para evaluar la calidad del clustering
            silhouette_avg = silhouette_score(data, labels)

            # Si la calidad del clustering es menor que el umbral o si el número de clusters excede el máximo, detener la división
            if silhouette_avg < threshold or len(set(current_cluster)) + 1 > max_clusters:
                return current_cluster

            # Crear nuevas etiquetas de clusters
            new_cluster_labels = current_cluster.copy()
            max_label = max(current_cluster)

            # Asignar nuevas etiquetas incrementadas para cada subcluster
            for label in set(labels):
                cluster_indices = np.where(labels == label)[0]
                new_label = max_label + 1 + label
                new_cluster_labels[cluster_indices] = new_label

            # Aplicar recursión para seguir dividiendo los subclusters
            for new_label in set(new_cluster_labels):
                cluster_indices = np.where(new_cluster_labels == new_label)[0]
                new_cluster_labels = divisive_clustering(data[cluster_indices], new_cluster_labels, new_cluster_labels)

            return new_cluster_labels

        # Inicializar las etiquetas de clusters con ceros
        initial_labels = np.zeros(len(self.dataframe))

        # Llamar a la función recursiva para iniciar el clustering divisivo
        final_labels = divisive_clustering(self.dataframe.values, initial_labels, initial_labels)

        # Añadir las etiquetas de clusters al DataFrame original
        dataframe_original["clusters_divisive"] = final_labels.astype(int).astype(str)

        return dataframe_original

    def modelo_espectral(self, dataframe_original, n_clusters=3, assign_labels='kmeans'):
        """
        Aplica clustering espectral al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - n_clusters : int, optional, default: 3. Número de clusters a formar.
            - assign_labels : str, optional, default: 'kmeans'. Método para asignar etiquetas a los puntos. Puede ser 'kmeans' o 'discretize'.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        spectral = SpectralClustering(n_clusters=n_clusters, assign_labels=assign_labels, random_state=0)
        self.labels["spectral"] = spectral.fit_predict(self.dataframe_escalado)
        dataframe_original["clusters_spectral"] = self.labels["spectral"].astype(str)
        return dataframe_original
    
    def modelo_dbscan(self, dataframe_original, eps_values=[0.5, 1.0, 1.5], min_samples_values=[3, 2, 1]):
        """
        Aplica DBSCAN al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - eps_values : list of float, optional, default: [0.5, 1.0, 1.5]. Lista de valores para el parámetro eps de DBSCAN.
            - min_samples_values : list of int, optional, default: [3, 2, 1]. Lista de valores para el parámetro min_samples de DBSCAN.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        best_eps = None
        best_min_samples = None
        best_silhouette = -1  # Usamos -1 porque la métrica de silueta varía entre -1 y 1

        # Iterar sobre diferentes combinaciones de eps y min_samples
        for eps in eps_values:
            for min_samples in min_samples_values:
                # Aplicar DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.dataframe_escalado)

                # Calcular la métrica de silueta, ignorando etiquetas -1 (ruido)
                if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                    silhouette = silhouette_score(self.dataframe_escalado, labels)
                else:
                    silhouette = -1

                # Mostrar resultados (opcional)
                print(f"eps: {eps}, min_samples: {min_samples}, silhouette: {silhouette}")

                # Actualizar el mejor resultado si la métrica de silueta es mejor
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_eps = eps
                    best_min_samples = min_samples

        # Aplicar DBSCAN con los mejores parámetros encontrados
        best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        self.labels["DBSCAN"] = best_dbscan.fit_predict(self.dataframe_escalado)

        # Añadir los labels al DataFrame original
        dataframe_original["clusters_dbscan"] = self.labels["DBSCAN"]

        return dataframe_original

    def calcular_metricas(self, labels: str):
        """
        Calcula métricas de evaluación del clustering.
        """
        if len(set(self.labels[labels])) <= 1:
            raise ValueError("El clustering debe tener al menos 2 clusters para calcular las métricas.")

        silhouette = silhouette_score(self.dataframe_escalado, self.labels[labels])
        davies_bouldin = davies_bouldin_score(self.dataframe_escalado, self.labels[labels])

        unique, counts = np.unique(self.labels[labels], return_counts=True)
        cardinalidad = dict(zip(unique, counts))

        return pd.DataFrame({
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "cardinalidad": cardinalidad
        }, index = [0])