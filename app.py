import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

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
images = ["drink.jpg", "food.jpg", "inside.jpg", "menu.jpg", "outside.jpg"]  # Remplacez par les chemins réels de vos images

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

    # Afficher le DataFrame
    st.subheader("Données t-SNE")
    st.dataframe(df_embeddings)
    
    # Calcul des clusters (KMeans)
    X_tsne = df_embeddings[['tsne1', 'tsne2', 'tsne3']].values
    cls_model = KMeans(n_clusters=5, random_state=6)
    df_embeddings['cluster'] = cls_model.fit_predict(X_tsne)

    # Calculer le score ARI
    ari_score = adjusted_rand_score(df_embeddings['class'], df_embeddings['cluster'])
    
    st.subheader(f"Score ARI : {ari_score:.3f}")
    
    # Fonction pour prédire les distances aux centres des clusters
    def predict_cluster_distances(data):
        columns = ['tsne1', 'tsne2', 'tsne3']
        data_df = pd.DataFrame(data, columns=columns)
        return cls_model.transform(data_df)

    # Utiliser SHAP pour expliquer les distances aux clusters
    explainer = shap.KernelExplainer(predict_cluster_distances, X_tsne)
    shap_values = explainer.shap_values(X_tsne)

    # Visualiser l'importance des features avec SHAP
    st.subheader("Interprétabilité avec SHAP")
    shap.summary_plot(shap_values, X_tsne, show=False, class_names={0:'Cluster 0', 1:'Cluster 1', 2:'Cluster 2', 3:'Cluster 3', 4:'Cluster 4'},class_inds='original')
    plt.gcf().set_size_inches(12, 8)
    st.pyplot(plt)