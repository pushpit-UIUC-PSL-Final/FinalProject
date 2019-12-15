
import image_processor as ip
import numpy


def load_data():
	
	benign_data = ip.read_image('./benign_resized')
	malignant_data = ip.read_image('./malignant_resized')

	benign_label = numpy.zeros((benign_data.shape[0]), dtype=int)
	malignant_label = numpy.ones((malignant_data.shape[0]), dtype=int)

	x = numpy.concatenate([benign_data, malignant_data], axis=0)
	y = numpy.concatenate([benign_label, malignant_label])
	return x, y
	
	
if __name__ == "__main__":
	#data = load_data_from_files("./benign_resized")
	#print(data.shape)
	x, y = load_data()
	print(x)
	print(y)