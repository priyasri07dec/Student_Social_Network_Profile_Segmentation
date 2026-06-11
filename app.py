import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load files

model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(
    page_title="Student Segmentation",
    layout="wide"
)

st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #f5f7fa;
}

/* Remove top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Custom button */
.stDownloadButton button {
    background-color: #667eea;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
}

.stDownloadButton button:hover {
    background-color: #764ba2;
    color: white;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 2px dashed #667eea;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background: linear-gradient(90deg,#667eea,#764ba2);
padding:30px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">
<h1 style="color:white;">
🎓 Student Segmentation System
</h1>
<p style="color:white;font-size:18px;">
Upload a dataset and identify student segments using Machine Learning
</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Template Download

template_df = pd.DataFrame(columns = feature_names)
template_csv = template_df.to_csv(index=False)

st.download_button(
    label = "Download Template CSV",
    data = template_csv,
    file_name = "student_template.csv",
    mime = "text/csv"
)

st.info("Download the template, fill student records, save as csv and upload below.")

# Upload file

uploaded_file = st.file_uploader("Upload Student CSV File", type = ['csv'])

if uploaded_file is not None:

    # Read Data
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    # Column Validation
    missing_cols = set(feature_names) - set(data.columns)

    if len(missing_cols) > 0:

        st.error(
            f"Missing Columns: {list(missing_cols)}"
        )

        st.stop()

    # Reorder Columns

    data = data[feature_names]

    # Gender Encoding
    if 'gender' in data.columns:
        data['gender'] = encoder.transform(data['gender'])

    # Scaling

    scaled_data = scaler.transform(data)

    #Convert scaled array to DataFrame

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=feature_names
    )

    #Predict Clusters

    clusters = model.predict(scaled_df)

    # Add Cluster Column

    data['Cluster'] = clusters

    # Optional Segment Names

    cluster_mapping = {
        0: "General Students",
        1: "Highly Active Students"
    }

    data['segment'] = data['Cluster'].map(cluster_mapping)
 
   # Adding Metrix
    col1, col2, col3 = st.columns(3)

    col1.metric(
    "👨‍🎓 Total Students",
    len(data)
  )

    col2.metric(
    "📊 Segments Found",
    data['Cluster'].nunique()
  )

    col3.metric(
    "🏆 Largest Segment",
    data['Cluster'].value_counts().max()
   )

  # Results

    st.subheader("Predicted Segments")
    st.dataframe(data)

    

# Distribution Chart

    st.subheader("Cluster Distribution")

    cluster_count = data['Cluster'].value_counts()

    fig,ax = plt.subplots(figsize=(3,2))
    ax.bar(
        cluster_count.index.astype(str),
        cluster_count.values
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Students")
    ax.set_title("Cluster Distribution")

    st.pyplot(fig, use_container_width=False)

    # Download Results

    result_csv = data.to_csv(index = False)

    st.download_button(
        label = "Downlaod Segmented Results",
        data = result_csv,
        file_name = "student_segments.csv",
        mime = "text/csv"
    )
