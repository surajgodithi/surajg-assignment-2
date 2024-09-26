import streamlit as st
import numpy as np
import plotly.express as px
from kmeans import KMeans, generate_data

# Function to plot data interactively using Plotly
def plot_data_interactive(X, centroids=None, labels=None):
    df = {
        'x': X[:, 0],
        'y': X[:, 1],
        'label': labels if labels is not None else ['Unlabeled'] * X.shape[0]
    }

    fig = px.scatter(df, x='x', y='y', color='label')

    # If centroids are provided, plot them as red 'X'
    if centroids is not None:
        centroid_df = {
            'x': centroids[:, 0],
            'y': centroids[:, 1]
        }
        fig.add_scatter(x=centroid_df['x'], y=centroid_df['y'], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Centroids')

    st.plotly_chart(fig)

# Streamlit application
st.title("Interactive KMeans Clustering")

# Sidebar controls
init_method = st.selectbox("Select Initialization Method", ['random', 'kmeans++', 'farthest_first', 'manual'])
n_clusters = st.slider("Number of Clusters", 2, 5, 3)
n_samples = st.slider("Number of Samples", 100, 500, 300)

# Button to generate new data
if st.button("Generate New Data"):
    X = generate_data(n_samples=n_samples)
    st.session_state['data'] = X
    st.session_state['manual_centroids'] = []
else:
    X = st.session_state.get('data', generate_data(n_samples=n_samples))

# Initialize KMeans
kmeans = KMeans(n_clusters=n_clusters, init_method=init_method)
labels = None
centroids = None

# For manual initialization, let the user enter centroids manually
if init_method == "manual":
    st.write("Manually enter centroid coordinates:")

    if 'manual_centroids' not in st.session_state:
        st.session_state['manual_centroids'] = []

    # Create input fields for manual centroid entry
    centroids = []
    for i in range(n_clusters):
        x_val = st.number_input(f"Centroid {i+1} X", value=0.0)
        y_val = st.number_input(f"Centroid {i+1} Y", value=0.0)
        centroids.append([x_val, y_val])

    # Once the centroids are entered, run KMeans
    if st.button("Run KMeans with Manual Centroids"):
        centroids = np.array(centroids)
        labels = kmeans.fit(X, manual_centroids=centroids)
        plot_data_interactive(X, centroids=centroids, labels=labels)

else:
    # Step-through option
    if 'step' not in st.session_state:
        st.session_state['step'] = 0

    if st.button("Step through KMeans"):
        labels = kmeans.fit(X)
        st.session_state['step'] += 1
        centroids = kmeans.history[st.session_state['step'] % len(kmeans.history)]
        labels = kmeans._assign_clusters()
        plot_data_interactive(X, centroids=centroids, labels=labels)
    else:
        plot_data_interactive(X, labels=labels)

# Reset button
if st.button("Reset"):
    st.session_state['step'] = 0
    st.session_state['manual_centroids'] = []
    st.experimental_rerun()
