# https://kyash03-breastcancerwebapp-webapp-17376n.streamlit.app

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import streamlit as st

st.sidebar.header('About')
st.sidebar.write(
    'This web application allows you to predict whether a patient has cancer. You can input the feature values of a breast lump, and the app will predict whether the breast lump is benign or malignant using an ML model (Support Vector Machine).')
st.sidebar.write(
    'This app utilizes [this](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) dataset, which contains information about breast cancer tumours diagnosed as either benign or malignant.')

st.sidebar.header('Input Parameters')


def get_input():
    radius_mean = st.sidebar.number_input('Mean of Radius', step=1e-6,
                                          format="%.6f")
    texture_mean = st.sidebar.number_input('Mean of Texture', step=1e-6,
                                           format="%.6f")
    perimeter_mean = st.sidebar.number_input(
        'Mean of Perimeter', step=1e-6,
        format="%.6f")
    area_mean = st.sidebar.number_input('Mean of Area', step=1e-6,
                                        format="%.6f")
    smoothness_mean = st.sidebar.number_input(
        'Mean of Smoothness', step=1e-6,
        format="%.6f")
    compactness_mean = st.sidebar.number_input(
        'Mean of Compactness', step=1e-6,
        format="%.6f")
    concavity_mean = st.sidebar.number_input(
        'Mean of Concavity', step=1e-6,
        format="%.6f")
    concave_points_mean = st.sidebar.number_input(
        'Mean of Concave Points', step=1e-6,
        format="%.6f")
    symmetry_mean = st.sidebar.number_input('Mean of Symmetry', step=1e-6,
                                            format="%.6f")
    fractal_dimension_mean = st.sidebar.number_input(
        'Mean of Fractal Dimension', step=1e-6,
        format="%.6f")
    radius_se = st.sidebar.number_input(
        'Standard Error of Radius', step=1e-6,
        format="%.6f")
    texture_se = st.sidebar.number_input(
        'Standard Error of Texture', step=1e-6,
        format="%.6f")
    perimeter_se = st.sidebar.number_input(
        'Standard Error of Perimeter', step=1e-6,
        format="%.6f")
    area_se = st.sidebar.number_input('Standard Error of Area', step=1e-6,
                                      format="%.6f")
    smoothness_se = st.sidebar.number_input(
        'Standard Error of Smoothness', step=1e-6,
        format="%.6f")
    compactness_se = st.sidebar.number_input(
        'Standard Error of Compactness', step=1e-6,
        format="%.6f")
    concavity_se = st.sidebar.number_input(
        'Standard Error of Concavity', step=1e-6,
        format="%.6f")
    concave_points_se = st.sidebar.number_input(
        'Standard Error of Concave Points', step=1e-6,
        format="%.6f")
    symmetry_se = st.sidebar.number_input(
        'Standard Error of Symmetry', step=1e-6,
        format="%.6f")
    fractal_dimension_se = st.sidebar.number_input(
        'Standard Error of Fractal Dimension', step=1e-6,
        format="%.6f")
    radius_worst = st.sidebar.number_input('Worst of Radius', step=1e-6,
                                           format="%.6f")
    texture_worst = st.sidebar.number_input('Worst of Texture', step=1e-6,
                                            format="%.6f")
    perimeter_worst = st.sidebar.number_input(
        'Worst of Perimeter', step=1e-6,
        format="%.6f")
    area_worst = st.sidebar.number_input('Worst of Area', step=1e-6,
                                         format="%.6f")
    smoothness_worst = st.sidebar.number_input(
        'Worst of Smoothness', step=1e-6,
        format="%.6f")
    compactness_worst = st.sidebar.number_input(
        'Worst of Compactness', step=1e-6,
        format="%.6f")
    concavity_worst = st.sidebar.number_input(
        'Worst of Concavity', step=1e-6,
        format="%.6f")
    concave_points_worst = st.sidebar.number_input(
        'Worst of Concave Points', step=1e-6,
        format="%.6f")
    symmetry_worst = st.sidebar.number_input(
        'Worst of Symmetry', step=1e-6,
        format="%.6f")
    fractal_dimension_worst = st.sidebar.number_input(
        'Worst of Fractal Dimension', step=1e-6,
        format="%.6f")

    data = {'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'concave points_mean': concave_points_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'perimeter_se': perimeter_se,
            'area_se': area_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se': concavity_se,
            'concave points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'radius_worst': radius_worst,
            'texture_worst': texture_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'smoothness_worst': smoothness_worst,
            'compactness_worst': compactness_worst,
            'concavity_worst': concavity_worst,
            'concave points_worst': concave_points_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst}

    return data


data = get_input()
user_input_df = pd.DataFrame(data, index=[0])

st.title('Breast Cancer Prediction')

st.write('Input feature values in the sidebar.')

svc_pipeline = joblib.load('final_model.joblib')

if (sum(data.values()) != 0):
    st.write('Prediction:')
    if svc_pipeline.predict(user_input_df)[0] == 0:
        st.success('The patient does not have cancer.')
    else:
        st.error('The patient has cancer.')

st.markdown("""---""")

st.write('The below heatmap shows the correlation between the features of a breast lump.')
df = pd.read_csv('data.csv')
fig_2 = plt.figure(figsize=(14, 12))
sns.heatmap(data=df.iloc[:, 2:-1].corr(), cmap='mako')
st.pyplot(fig_2)
