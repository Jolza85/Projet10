import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Définir une palette de couleurs
color_map = {
    '0': '#636EFA',  # Couleur pour le Cluster 0
    '1': '#EF553B',  # Couleur pour le Cluster 1
    '2': '#00CC96',  # Couleur pour le Cluster 2
    '3': '#AB63FA',  # Couleur pour le Cluster 3
    '4': '#FFA15A',  # Couleur pour le Cluster 4
}

# Fonction pour charger les données en fonction du modèle sélectionné
def load_data(model_name):
    if model_name == "SIFT":
        return pd.read_csv("df_tsne_sift.csv")
    elif model_name == "CNN":
        return pd.read_csv("df_tsne_cnn.csv")
    elif model_name == "CLIP":
        return pd.read_csv("df_tsne_clip.csv")
    else:
        return pd.DataFrame(columns=["tsne1", "tsne2", "tsne3", "class"])

# Titre de l'application
st.title("Dashboard classification photos restaurants avec CLIP d'OpenAI")

# Afficher les images des classes
classes = ["drink", "food", "inside", "menu", "outside"]
images = ["drink.jpg", "food.jpg", "inside.jpg", "menu.jpg", "outside.jpg"] 

st.subheader('Représentation des classes')
cols = st.columns(5)
for i, col in enumerate(cols):
    col.image(images[i], caption=classes[i], use_column_width=True)

# Sélection du modèle
model_name = st.sidebar.selectbox(
    'Choisissez un modèle',
    ('SIFT', 'CNN', 'CLIP')
)

# Charger les données correspondantes
df_embeddings = load_data(model_name)

# Vérifier si les données sont disponibles
if df_embeddings.empty:
    st.write("Aucune donnée disponible pour le modèle sélectionné.")
else:
    # Encodage des labels de classes en numériques
    le = LabelEncoder()
    df_embeddings['num_class'] = le.fit_transform(df_embeddings['class'])
    
    # Créer un graphique interactif en 3D avec couleurs par classe
    fig_3d = px.scatter_3d(
        df_embeddings, x='tsne1', y='tsne2', z='tsne3',
        color='class',
        labels={'tsne1': 'TSNE1', 'tsne2': 'TSNE2', 'tsne3': 'TSNE3'}
    )

    # Créer un graphique interactif en 2D avec couleurs par classe
    fig_2d = px.scatter(
        df_embeddings, x='tsne1', y='tsne2',
        color='class',
        labels={'tsne1': 'TSNE1', 'tsne2': 'TSNE2'}
    )

    st.subheader(f"Visualisation 3D t-SNE des Embeddings - {model_name}")
    st.plotly_chart(fig_3d)

    st.subheader(f"Visualisation 2D t-SNE des Embeddings - {model_name}")
    st.plotly_chart(fig_2d)
    
    # Calcul des clusters (KMeans)
    X_tsne = df_embeddings[['tsne1', 'tsne2', 'tsne3']].values
    cls_model = KMeans(n_clusters=5, random_state=6)
    df_embeddings['cluster'] = cls_model.fit_predict(X_tsne)

    st.subheader(f"Visualisation 2D t-SNE des Clusters Prédits - {model_name}")
    fig_2d_clusters = px.scatter(
        df_embeddings, x='tsne1', y='tsne2',
        color=df_embeddings['cluster'].astype(str),  # Convertir les clusters en string pour le mapping des couleurs
        color_discrete_map=color_map,  # Appliquer la palette de couleurs
        labels={'tsne1': 'TSNE1', 'tsne2': 'TSNE2'}
    )
    st.plotly_chart(fig_2d_clusters)

    # Afficher le DataFrame
    st.subheader("Données T-SNE avec la class initiale et le cluster prédit")
    st.dataframe(df_embeddings)
    
    # Calculer le score ARI
    ari_score = adjusted_rand_score(df_embeddings['class'], df_embeddings['cluster'])
    
    st.subheader(f"Score ARI : {ari_score:.3f}")

    # Calcul de la matrice de confusion
    conf_mat = metrics.confusion_matrix(df_embeddings['num_class'], df_embeddings['cluster'])

    # Création du DataFrame pour la heatmap
    df_cm = pd.DataFrame(conf_mat, index=[label for label in classes], columns=[i for i in range(5)])

    # Affichage de la heatmap dans Streamlit
    st.subheader("Heatmap entre les Clusters et les Catégories des Images")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='d')
    st.pyplot(plt)
    
    # Fonction pour prédire les distances aux centres des clusters
    def predict_cluster_distances(data):
        columns = ['tsne1', 'tsne2', 'tsne3']
        data_df = pd.DataFrame(data, columns=columns)
        return cls_model.transform(data_df)

    # Utiliser SHAP pour expliquer les distances aux clusters
    explainer = shap.KernelExplainer(predict_cluster_distances, X_tsne)
    shap_values = explainer.shap_values(X_tsne, silent=True)

    # Visualiser l'importance des features avec SHAP
    st.subheader("Interprétabilité avec SHAP")
    shap.summary_plot(shap_values, X_tsne, show=False, class_names={0:'Cluster 0', 1:'Cluster 1', 2:'Cluster 2', 3:'Cluster 3', 4:'Cluster 4'},class_inds='original')
    plt.gcf().set_size_inches(12, 8)
    st.pyplot(plt)

# URL du dashboard en ligne : https://projet10.streamlit.app