import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

st.title("K NEAREST NEIGHBORS FOR WINE AND IRIS DATASET")
st.write(""" 
## Explore KNN for different datasets and identifyinh]g which K works best? """)


st.markdown("""<style>body{background-color: Blue;}</style>""",unsafe_allow_html=True)


dataset_name = st.sidebar.selectbox("Select Dataset",("Wine","Iris"))

st.write(f"### {dataset_name} Dataset")

data = None
if dataset_name == "Wine":
    data =  datasets.load_wine()
else:
    data = datasets.load_iris()

X = data.data
y = data.target


st.write("Features:",data["feature_names"])
st.write("Shape of the dataset:",X.shape)
st.write("Number of classes:",len(np.unique(y)))

K = st.sidebar.slider("K Values",1,15)
clf  = KNeighborsClassifier(K)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

st.write(f"Accuracy = {accuracy_score(y_test,y_pred)*100}%")

do_plot = st.button("Plot")

if do_plot:
    pca = PCA(2)
    X_projected = pca.fit_transform(X_test)

    x1=X_projected[:,0]
    x2 = X_projected[:,1]

    fig = plt.figure()
    plt.scatter(x1,x2,c=y_pred,alpha=0.8,cmap="viridis")

    plt.colorbar()
    st.pyplot(fig)

