import data as dm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

def main(x_train, x_test, y_train, y_test):

	lr = LinearRegression()
	lr.fit(x_train, y_train)
	predictions = lr.predict(x_test)
	print(predictions)
	predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
	print(predictions)
	score = accuracy_score(y_test, predictions)
	print(score)
	return score


if __name__ == "__main__":
	x, y = dm.load_data()
	pca = PCA(n_components=300)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=1)

	main(x_train, x_test, y_train, y_test)
			
	
		