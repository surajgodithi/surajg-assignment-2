import streamlit as st
import numpy as np
import plotly.express as px
from kmeans import KMeans, generate_data

def plot_data_interactive(placeholder, X, centroids=None, labels=None):
    df = {
        'x': X[:, 0],
        'y': X[:, 1],
        'label': labels if labels is not None else ['Unlabeled'] * X.shape[0]
    }

    fig = px.scatter(df, x='x', y='y', color='label', color_discrete_sequence=px.colors.qualitative.G10)

    if centroids is not None:
        centroid_df = {
            'x': centroids[:, 0],
            'y': centroids[:, 1]
        }
        fig.add_scatter(x=centroid_df['x'], y=centroid_df['y'], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Centroids')

    placeholder.plotly_chart(fig)

st.title("Interactive KMeans Clustering")

init_method = st.selectbox("Select Initialization Method", ['random', 'kmeans++', 'farthest_first', 'manual'])
n_clusters = st.slider("Number of Clusters", 2, 5, 3)
n_samples = st.slider("Number of Samples", 100, 500, 300)

if 'data' not in st.session_state:
    st.session_state['data'] = generate_data(n_samples=n_samples)

if 'kmeans_history' not in st.session_state:
    st.session_state['kmeans_history'] = None

if 'manual_centroids' not in st.session_state:
    st.session_state['manual_centroids'] = []

if 'step' not in st.session_state:
    st.session_state['step'] = 0

if 'kmeans_model' not in st.session_state:
    st.session_state['kmeans_model'] = None

if st.button("Generate New Data"):
    X = generate_data(n_samples=n_samples)
    st.session_state['data'] = X
    st.session_state['manual_centroids'] = []
    st.session_state['step'] = 0  
    st.session_state['kmeans_history'] = None  
    st.session_state['kmeans_model'] = None  
else:
    X = st.session_state.get('data', generate_data(n_samples=n_samples))

graph_placeholder = st.empty()

plot_data_interactive(graph_placeholder, X)

if st.session_state['kmeans_model'] is None:
    kmeans = KMeans(n_clusters=n_clusters, init_method=init_method)
    st.session_state['kmeans_model'] = kmeans
else:
    kmeans = st.session_state['kmeans_model']

labels = None
centroids = None

if init_method == "manual":
    st.write("Manually enter centroid coordinates:")

    centroids = []
    for i in range(n_clusters):
        x_val = st.number_input(f"Centroid {i+1} X", value=0.0)
        y_val = st.number_input(f"Centroid {i+1} Y", value=0.0)
        centroids.append([x_val, y_val])

    if st.button("Set Manual Centroids"):
        centroids = np.array(centroids)
        st.session_state['manual_centroids'] = centroids
        kmeans.centroids = centroids  
        st.session_state['kmeans_model'] = kmeans  
        plot_data_interactive(graph_placeholder, X, centroids=centroids)  

if st.session_state['kmeans_model'] is not None:
    if st.button("Step through KMeans"):
        if st.session_state['kmeans_history'] is None:
            kmeans.fit(X) 
            st.session_state['kmeans_history'] = kmeans.history 
            st.session_state['kmeans_model'] = kmeans 

        if st.session_state['step'] < len(st.session_state['kmeans_history']):
            centroids = st.session_state['kmeans_history'][st.session_state['step']]
            kmeans.centroids = centroids 
            labels = kmeans._assign_clusters()  
            st.session_state['step'] += 1  
            plot_data_interactive(graph_placeholder, X, centroids=centroids, labels=labels)
        else:
            st.session_state['step'] = len(st.session_state['kmeans_history'])  # Mark as converged
            labels = kmeans._assign_clusters() 
            plot_data_interactive(graph_placeholder, X, centroids=kmeans.centroids, labels=labels)
            st.write("KMeans has converged.")
            
    if st.button("Straight to Convergence"):
        labels = kmeans.fit(X)
        centroids = kmeans.centroids  
        labels = kmeans._assign_clusters() 
        plot_data_interactive(graph_placeholder, X, centroids=centroids, labels=labels)

# Reset button
if st.button("Reset"):
    st.session_state['step'] = 0
    st.session_state['manual_centroids'] = []
    st.session_state['kmeans_history'] = None 
    st.session_state['kmeans_model'] = None  
    st.experimental_rerun()
