import os
import sys
if sys.argv.count("-nogpu") % 2 != 0:
	print("GPU was turned off")
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import load_model
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import time

batch_size = 32
image_size = (128, 128)
glove_embeddings = {}

def load_dataset(path, truncate=False):
	dataset = tf.keras.preprocessing.image_dataset_from_directory(
		path,
		seed=123,
		image_size=image_size,
		batch_size=batch_size,
		label_mode='int'
	)
	
	images, labels = [], []
	for img, label in dataset:
		images.append(img.numpy())
		labels.append(label.numpy())
	
	images = np.concatenate(images)
	labels = np.concatenate(labels)
	
	images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
	
	if truncate:
		class_0_indices = np.where(labels == 0)[0]
		class_1_indices = np.where(labels == 1)[0]
		
		min_samples = min(len(class_0_indices), len(class_1_indices))
		class_0_indices = np.random.choice(class_0_indices, min_samples, replace=False)
		class_1_indices = np.random.choice(class_1_indices, min_samples, replace=False)
		
		selected_indices = np.concatenate([class_0_indices, class_1_indices])
		np.random.shuffle(selected_indices)
		
		images = images[selected_indices]
		labels = labels[selected_indices]
	
	return images, labels

def create_cnn():
	base_model = tf.keras.applications.MobileNetV2(input_shape=(image_size[0], image_size[1], 3), include_top=False, weights='imagenet')
	base_model.trainable = False
	model = models.Sequential([
		base_model,
		layers.GlobalAveragePooling2D(),
		layers.Dense(128, activation='relu'),
		layers.Dense(1, activation='sigmoid')
	])
	
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	
	return model
	

def create_devise_cnn(embedding_dim=100):
	base_model = tf.keras.applications.MobileNetV2(input_shape=(image_size[0], image_size[1], 3), include_top=False, weights='imagenet')
	base_model.trainable = False
	
	model = models.Sequential([
		base_model,
		layers.GlobalAveragePooling2D(),
		layers.Dense(embedding_dim, activation='linear')
	])
	
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='cosine_similarity', metrics=['accuracy'])
	
	return model

def plot_training_history(path, history):
	# Create a figure with subplots
	plt.figure(figsize=(12, 4))

	# Plot training & validation accuracy values
	plt.subplot(1, 2, 1)
	plt.plot(history.history['accuracy'], label='Train')
	plt.plot(history.history['val_accuracy'], label='Validation')
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(loc='upper left')

	# Plot training & validation loss values
	plt.subplot(1, 2, 2)
	plt.plot(history.history['loss'], label='Train')
	plt.plot(history.history['val_loss'], label='Validation')
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(loc='upper left')

	# Adjust layout and show the plots
	plt.tight_layout()
	plt.savefig(path, dpi=300)


def load_single_image(path):
	img = tf.keras.preprocessing.image.load_img(path, target_size=image_size)
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)
	img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
	
	return img_array

def cnn_predict_class(image, model):
	predictions = cnn.predict(image)
	return 1 if predictions[0] > 0.5 else 0

class TimeHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs=None):
		self.epoch_times = []

	def on_epoch_begin(self, epoch, logs=None):
		self.epoch_start_time = time.time()

	def on_epoch_end(self, epoch, logs=None):
		epoch_time = time.time() - self.epoch_start_time
		self.epoch_times.append(epoch_time)

if __name__ == "__main__":
	if len(sys.argv) == 1:
		print(f"""
Usage: {sys.argv[0]} <parameters>
Possible parameters:
	-path = specify path to dataset
		{sys.argv[0]} -path path/to/my/data
	-o = output path(must be a directory)
	-tr = run NNT in neural network model training mode
	-te = run NNT in model testing mode(loads model from -path)
	-vl = evaluate and score model
	-trunc = truncate input dataset, if the classes are unbalanced
	-epochs = set epoch count(default is 5)
	-nogpu = don't use GPU for training/testing, does nothing on systems with no GPU
Commands in model testing mode:
	exit = exits NNT
	batch = loads a batch of samples and makes predictions
		batch /my/batch/dir
	batchlf = loads a batch of samples, makes predictions and saves predictions to a log file
		batchlf /my/batch/dir my.log
	any other input is treated as a path to a single file for prediction
		""")
		exit()
	
	else:
		train_cnn = False
		save_cnn = False
		test_cnn = False
		validate_cnn = False
		truncate_dataset = False
		cnn_path = ""
		output_dir = ""
		data_path = ""
		epoch_count = 5
		
		i = 1
		while i < len(sys.argv):
			if sys.argv[i] == "-path":
				data_path = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-o":
				output_dir = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-tr":
				train_cnn = not train_cnn
				test_cnn = False if train_cnn else test_cnn
			elif sys.argv[i] == "-s":
				save_cnn = not save_cnn
			elif sys.argv[i] == "-te":
				test_cnn = not test_cnn
				train_cnn = False if test_cnn else train_cnn
			elif sys.argv[i] == "-vl":
				validate_cnn = not validate_cnn
			elif sys.argv[i] == "-trunc":
				truncate_dataset = not truncate_dataset
			elif sys.argv[i] == "-epochs":
				epoch_count = int(sys.argv[i + 1])
				i += 1
			elif sys.argv[i] == "-nogpu":
				i += 1
				continue
			i += 1
		
		if not data_path:
			print("Path to dataset was not specified...")
			exit()
		
		if not output_dir and not test_cnn:
			print("Output path was not specified...")
			exit()
		
		if not os.path.exists(output_dir) and not test_cnn:
			os.makedirs(output_dir, exist_ok=True)
		
		if train_cnn:
			cnn = create_cnn()
			images, labels = load_dataset(data_path, truncate_dataset)
			X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
			
			time_callback = TimeHistory()
			history = cnn.fit(
				X_train, y_train,
				epochs=epoch_count,
				validation_data=(X_test, y_test),
				callbacks=[time_callback]
			)
			
			print(f"Trained MobileNetV2 model in {sum(time_callback.epoch_times)} seconds")
			plot_training_history(f"{output_dir}{os.sep}cnn_train_history.png", history)
			if save_cnn:
				cnn.save(f'{output_dir}{os.sep}cnn_model.keras')
				
			if validate_cnn:
				predictions = cnn.predict(X_test)
				predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
				predictions = np.array(predictions)
				
				score_accuracy = accuracy_score(y_test, predictions)
				score_f1 = f1_score(y_test, predictions)
				score_recall = recall_score(y_test, predictions)
				score_precision = precision_score(y_test, predictions)
				conf_mat = confusion_matrix(y_test, predictions)
				conf_mat = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]
				print(f'Accuracy = {score_accuracy}\nF1 = {score_f1}\nPrecision = {score_precision}\nRecall = {score_recall}\nConfusion matrix = [\n{conf_mat}\n]')
				with open(f'{output_dir}{os.sep}scores.txt', 'w') as f:
					f.write(f'Trained for {sum(time_callback.epoch_times):.2f} seconds\n')
					f.write(f'Accuracy: {score_accuracy:.4f}\n')
					f.write(f'F1 Score: {score_f1:.4f}\n')
					f.write(f'Precision: {score_precision:.4f}\n')
					f.write(f'Recall: {score_recall:.4f}\n')
					f.write(f'Confusion Matrix:\n{conf_mat}\n')
			
			
		if test_cnn:
			cnn = load_model(data_path)
			while True:
				print(">>>", end=' ')
				input_str = input()
				
				if input_str.lower() == 'exit':
					break
				
				else:
					split_com = input_str.split(' ')
					if split_com[0].lower() == "batch":
						files = os.listdir(split_com[1])
						print(f'File count: {len(files)}')
						for file in files:
							fullpath = os.path.join(split_com[1], file)
							if os.path.isfile(fullpath):
								sample = load_single_image(fullpath)
								
								log.write(f"\"{file}\" predicted class: {cnn_predict_class(sample, cnn)}\n")
						continue
					if split_com[0].lower() == "batchlf":
						files = os.listdir(split_com[1])
						print(f'File count: {len(files)}')
						if len(split_com) < 3:
							print("Path to a file with results was not given")
							continue
						with open(split_com[2], 'w', encoding='utf-8') as log:
							for file in files:
								fullpath = os.path.join(split_com[1], file)
								if os.path.isfile(fullpath):
									sample = load_single_image(fullpath)
									
									log.write(f"\"{file}\" predicted class: {cnn_predict_class(sample, cnn)}\n")
						continue
					try:
						sample = load_single_image(input_str)
						print(f"Predicted class: {cnn_predict_class(sample, cnn)}")
					except Exception as e:
						print(f"{e}")

