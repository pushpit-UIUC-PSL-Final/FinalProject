import data as dm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def main(x_train, x_test, y_train, y_test, n_neighbors):

	knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
	knn.fit(x_train, y_train)
	predictions = knn.predict(x_test)

	score = accuracy_score(y_test, predictions)
	print(score)
	return score


if __name__ == "__main__":
	x, y = dm.load_data()
	pca = PCA(n_components=100)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=1)
	for i in [1, 3, 5, 7, 9, 11, 13, 15]:
		print(i)
		main(x_train, x_test, y_train, y_test, i)
			
	
		