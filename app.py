import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

st.title("Classification App")

name = st.sidebar.selectbox("Select Dataset", ["Iris", "Breast Cancer", "Wine"])
st.write("Dataset name : ", name)

classifier_name = st.sidebar.selectbox("Select Classifier", ["KNN", "SVM"])

def get_dataset(dataset_name):
	selector = {"Iris" : "datasets.load_iris()",
				"Breast Cancer" : "datasets.load_breast_cancer()",
				"Wine" : "datasets.load_wine()"}
	data = eval(selector[dataset_name])
	x = data.data
	y = data.target
	return x, y

x, y = get_dataset(name)
st.write("Size :", len(x))
st.write("Labels :", len(set(y)))


def get_values(classifier_name):
	selector = {
		"KNN" : "st.sidebar.slider('K', 1, 15)",
		"SVM" : "st.sidebar.slider('C', 0.01, 10.0)"
	}
	out = "K" if classifier_name == "KNN" else "C"
	tmp = eval(selector[classifier_name])
	return {out : tmp}

def create_and_clasify(name, x, y, para):
	print(para)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	if name == "KNN":
		clf = KNeighborsClassifier(n_neighbors=para["K"])
		clf.fit(x_train, y_train)
	else:
		clf = svm.SVC(C=para["C"])
		clf.fit(x_train, y_train)
	y_model = clf.predict(x_test)
	acc = accuracy_score(y_test, y_model)
	return acc 


out = get_values(classifier_name)
acc = create_and_clasify(classifier_name, x, y, out)
st.write("Accuracy :", acc)

pca = PCA(2)

print(len(x))
x_proj = pca.fit_transform(x)
x1 = x_proj[:, 0]
x2 = x_proj[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()
st.pyplot(fig)