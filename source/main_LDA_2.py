from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import datasets
import matplotlib.pyplot as plt


# Loading Wine Dataset 
wine = datasets.load_wine()

X = wine.data
y = wine.target
target_names = wine.target_names

# fitting the LDA model
lda = LDA(n_components=2)
lda_X = lda.fit(X,y).transform(X)

plt.scatter(lda_X[y == 0, 0], lda_X[y == 0, 1], s =80, c = 'blue', label = 'Class 1')
plt.scatter(lda_X[y == 1, 0], lda_X[y == 1, 1], s =80,  c = 'orange', label = 'Class 2')
plt.scatter(lda_X[y == 2, 0], lda_X[y == 2, 1], s =80,  c = 'green', label = 'Class 3')
plt.title('LDA plot for Wine Dataset')
plt.legend()
plt.show()