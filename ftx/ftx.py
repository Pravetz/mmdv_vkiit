import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from scipy.special import softmax
import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import joblib
import lime
import lime.lime_tabular
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
import pickle
import time

global denoise_threshold
denoise_threshold = 0.9

def calculate_snr(data):
	signal_power = np.mean(np.square(data))
	noise_power = np.var(data)
	return signal_power / noise_power

def dynamic_threshold_snr(data, snr_weight=0.90):
	snr = calculate_snr(data)
	if snr_weight == 0:
		return -np.inf
	
	threshold_factor = snr * (1.00 - snr_weight)
	if threshold_factor == 0:
		threshold_factor = 1
	
	high_percentile = np.percentile(np.abs(data), snr_weight * 100)
	
	if high_percentile == 0:
		return np.min(np.abs(data[data != 0]))
	
	return high_percentile / threshold_factor

def denoise(data):
	global denoise_threshold
	fft_values = np.fft.fft(data)
	threshold = dynamic_threshold_snr(np.abs(fft_values), denoise_threshold)
	if threshold == 0:
		return data
	
	fft_values[np.abs(fft_values) < threshold] = 0
	return np.fft.ifft(fft_values).real

def load_dataset(datapath, keyfile, do_denoise=False):
	print("Loading dataset...")
	sensor_values_list = []
	labels = []
	with open(datapath + os.sep + keyfile, 'r', encoding='utf-8') as f:
		i = 1
		for fline in f:
			with open(datapath + os.sep + f"T{i:04}.txt", 'r', encoding='utf-8') as sensor_data:
				sensor_values = []
				for line in sensor_data:
					sensor_values.append(float(line))
				if do_denoise:
					sensor_values = denoise(sensor_values)
				sensor_values_list.append(sensor_values[:])
				labels.append(int(float(fline)))
			
			i += 1
	
	print("Done")
	return np.array(sensor_values_list), np.array(labels)

def create_random_forest(X_train, y_train):
	start_time = time.time()
	classifier = RandomForestClassifier(verbose=1, n_jobs=5)
	classifier.fit(X_train, y_train)
	end_time = time.time()
	
	print(f"Random Forest trained in {end_time - start_time} seconds")
	return classifier

def create_naive_bayes(X_train, y_train):
	start_time = time.time()
	classifier = GaussianNB()
	classifier.fit(X_train, y_train)
	end_time = time.time()
	
	print(f"Naive Bayes trained in {end_time - start_time} seconds")
	return classifier

def create_svm(X_train, y_train):
	print('Навчання SVM...')
	start_time = time.time()
	classifier = SVC(kernel='linear', probability=True)
	classifier.fit(X_train, y_train)
	end_time = time.time()
	print('Done!')
	print(f"SVM trained in {end_time - start_time} seconds")
	
	return classifier

def create_gboost(X_train, y_train):
	start_time = time.time()
	classifier = GradientBoostingClassifier(verbose=1)
	classifier.fit(X_train, y_train)
	end_time = time.time()
	
	print(f"Gradient Boosting trained in {end_time - start_time} seconds")
	return classifier

def create_knn(X_train, y_train):
	start_time = time.time()
	classifier = KNeighborsClassifier()
	classifier.fit(X_train, y_train)
	end_time = time.time()
	
	print(f"KNN trained in {end_time - start_time} seconds")
	return classifier


def create_lime_explainer(X_train):
	explainer = lime.lime_tabular.LimeTabularExplainer(
		X_train, 
		mode="classification", 
		feature_names=[f"Feature {i}" for i in range(X_train.shape[1])],
		class_names=["0", "1"], 
		discretize_continuous=False
	)
	
	return explainer

def save_sets(path, sets, labels):
	i = 1
	np.savetxt(path + f"{os.sep}key.txt", labels, delimiter=',', fmt='%d')
	for data in sets:
		np.savetxt(f"{path}{os.sep}T{i:04}.txt", data, delimiter=',')
		i += 1

def extract_feature_importances(classifier, explainer, X_test, y_test):
	idx_class_0 = (y_test == 0).nonzero()[0]
	idx_class_1 = (y_test == 1).nonzero()[0]
	explanation_0 = explainer.explain_instance(X_test[idx_class_0[0]], get_predict_proba_function(classifier), num_features=X_test.shape[1])
	explanation_1 = explainer.explain_instance(X_test[idx_class_1[0]], get_predict_proba_function(classifier), num_features=X_test.shape[1])
	
	return explanation_0.as_list(), explanation_1.as_list()

def extract_important_feature_indices(feature_importances):
	important_indices = []
	
	mean_importance = np.mean([abs(importance) for _, importance in feature_importances])
	
	feature_pattern = re.compile(r"Feature (\d+)")
	
	for feature, importance in feature_importances:
		if abs(importance) >= mean_importance:
			idx_match = feature_pattern.search(feature)
			important_indices.append(int(idx_match.group(1)))
	
	return np.array(important_indices)

def feature_indices_intersect(fil, fir):
	return np.intersect1d(fil, fir)

def feature_indices_comb(fil, fir):
	return np.unique(np.concatenate((fil, fir)))

def get_predict_proba_function(classifier):
	if 'tf' in globals():
		if isinstance(classifier, tf.keras.Model):
			def predict_proba_tf(X):
				X = np.array(X)
				predictions = classifier.predict(X)
				
				if predictions.shape[1] > 1:
					return softmax(predictions, axis=1)
				else:
					return np.hstack([1 - predictions, predictions])
			
			return predict_proba_tf
	
	if 'torch' in globals():
		if isinstance(classifier, torch.nn.Module):
			def predict_proba_torch(X):
				classifier.eval()
				X_tensor = torch.tensor(X).float().unsqueeze(0)
				
				with torch.no_grad():
					logits = classifier(X_tensor)
				
				if logits.size(1) > 1:
					probs = torch.softmax(logits, dim=1).numpy()
				else:
					probs = torch.sigmoid(logits).numpy()
					probs = np.hstack([1 - probs, probs])
				
				return probs
			
			return predict_proba_torch
	
	if 'sklearn' in globals():
		if hasattr(classifier, 'predict_proba'):
			return classifier.predict_proba
		
		elif hasattr(classifier, 'decision_function'):
			def decision_function_as_proba(X):
				decision_scores = classifier.decision_function(X)
				
				if len(decision_scores.shape) == 1:
					decision_scores = np.vstack([-decision_scores, decision_scores]).T
				
				return softmax(decision_scores, axis=1)
			
			return decision_function_as_proba
	
	raise ValueError(f"Model of type {type(classifier).__name__} is either not supported, or corresponding library was not imported.")

def output_feature_importances(path, importances, flt=None):
	feature_pattern = re.compile(r"Feature (\d+)")
	
	with open(path, 'w', encoding='utf-8') as out:
		for feature, importance in importances:
			idx_match = feature_pattern.search(feature)
			idx = int(idx_match.group(1))
			if flt is not None and idx in flt or flt is None:
				out.write(f"{importance}\n")

def filter_tuples(tuples, flt=None):
	if flt is None:
		return zip(*tuples)
	feature_names = []
	feature_importances = []
	feature_pattern = re.compile(r"Feature (\d+)")
	
	for feature, importance in tuples:
		idx_match = feature_pattern.search(feature)
		idx = int(idx_match.group(1))
		if flt is not None and idx in flt or flt is None:
			feature_names.append(idx_match.group(1))
			feature_importances.append(importance)
			
	return feature_names, feature_importances

def plot_feature_importances(path, importances, classname, flt=None):
	features, importances = filter_tuples(importances, flt)
	
	features = np.array(features)
	importances = np.array(importances)
	
	sorted_idx = np.argsort(np.abs(importances))[::-1]
	features = features[sorted_idx]
	importances = importances[sorted_idx]
	
	colors = ['blue' if i > 0 else 'orange' for i in importances]
	
	fig, ax = plt.subplots(figsize=(10, 8))
	ax.barh(features, importances, color=colors)
	ax.axvline(x=0, color='gray', linestyle='--')
	ax.set_xlabel('Weight')
	ax.set_ylabel('Feature')
	ax.set_title(f'Feature Importance, Class {classname}')
	ax.invert_yaxis()
	ax.tick_params(axis='y', labelsize=1)
	plt.gca().set_yticklabels([])
	
	legend_elements = [
		Line2D([0], [0], color='blue', lw=4, label='Supports'),
		Line2D([0], [0], color='orange', lw=4, label='Contradicts')
	]
	ax.legend(handles=legend_elements, loc='lower right')
	
	plt.tight_layout()
	plt.savefig(path, dpi=300)

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

def load_classifiers(path):
	classifiers = {}
	if not os.path.exists(path):
		return classifiers
	
	for folder_name in os.listdir(path):
		folder_path = os.path.join(path, folder_name)
		classifier_file = os.path.join(folder_path, 'classifier.pkl')
		
		if os.path.isfile(classifier_file):
			print(f'Loading {folder_name}...')
			classifiers[folder_name] = joblib.load(classifier_file)
	
	return classifiers

def load_classifier_feature_indices(path):
	classifier_features = {}
	classifier_feature_imps = {}
	if not os.path.exists(path):
		return classifier_features
	
	for folder_name in os.listdir(path):
		folder_path = os.path.join(path, folder_name)
		features_file = os.path.join(folder_path, 'feature_importances_info_indices.txt')
		features_imps_file = os.path.join(folder_path, 'feature_importances.txt')
		classifier_file = os.path.join(folder_path, 'classifier.pkl')
		
		if os.path.isfile(classifier_file):
			if os.path.isfile(features_file):
				print(f'Loading important features for {folder_name}...')
				classifier_features[folder_name] = []
				with open(features_file, 'r', encoding='utf-8') as ff:
					for line in ff:
						classifier_features[folder_name].append(int(line))
			if os.path.isfile(features_imps_file):
				print(f'Loading feature weights for {folder_name}...')
				classifier_feature_imps[folder_name] = ([], [])
				with open(features_imps_file, 'r', encoding='utf-8') as ff:
					for line in ff:
						classifier_feature_imps[folder_name][0].append((f"Feature {i}", float(line)))
						classifier_feature_imps[folder_name][1].append((f"Feature {i}", float(line)))
						
	
	return classifier_features, classifier_feature_imps

def pick_max_nonzero(tuples):
	def count_nonzero(lst):
		return sum(1 for _, value in lst if value != 0)
	
	max_list = max(tuples, key=count_nonzero)
	return max_list

if __name__ == "__main__":
	if len(sys.argv) == 1:
		print(f"""
Usage: {sys.argv[0]} <parameters>
Possible parameters:
	-path = specify path to dataset
	-key = specify key file name(in -path)
		{sys.argv[0]} -path path/to/my/data -key \"key.txt\"
	-C<gb, rf, nb, svm, knn> = add Gradient Boosting/Random Forest/Naive Bayes/SVM/KNN model architecture to a set of classifiers for training
	-cfu = enables union operation on classifier important feature lists
	-cfi = enables intersection operation on classifier important feature lists
	-o = output path(must be a directory)
	-sc = save classifiers
	-sds = save data split(training and testing sets)
	-sps = save reduced datasets for each intersection/unification of classifiers
	-noexpl = do not save explanation plot
	-noft = do not extract features
	-nords = do not reduce input dataset
	-denoise <T> = enable denoise with threshold T, if T is 0, denoise is disabled
	-lc <path> = load classifiers in given path
	-lds = additionally load data split from path, given to -lc
	-lcif = additionally load feature lists from path, given to -lc
	-topn <N> = classifier top places count, by default, N equals to length of the set of classifiers
	-topby <accuracy, precision, recall, f1> = create top of classifiers by a specified score, accuracy by default
		""")
		exit()
	else:
		datapath = ""
		keyfile = ""
		cload_path = ""
		output_dir = ""
		possible_metrics = [ 'accuracy', 'precision', 'recall', 'f1' ]
		top_metric = 'accuracy'
		make_rnn = False
		output_reduced_set = True
		save_pair_reduced_sets = False
		unite_classifier_ftindices = False
		intersect_classifier_ftindices = False
		load_classifier_important_fts = False
		save_classifiers = False
		dont_explain = False
		dont_extract_features = False
		do_denoise = False
		top_n = None
		load_data_split = False
		save_data_split = False
		
		used_classifiers = {}
		i = 1
		while i < len(sys.argv):
			if sys.argv[i] == "-path":
				datapath = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-key":
				keyfile = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-nords":
				output_reduced_set = not output_reduced_set
			elif sys.argv[i] == "-o":
				output_dir = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-Cnb":
				used_classifiers["Naive Bayes"] = create_naive_bayes
			elif sys.argv[i] == "-Crf":
				used_classifiers["Random Forest"] = create_random_forest
			elif sys.argv[i] == "-Csvm":
				used_classifiers["SVM"] = create_svm
			elif sys.argv[i] == "-Cgb":
				used_classifiers["Gradient Boosting"] = create_gboost
			elif sys.argv[i] == "-Cknn":
				used_classifiers["K-Nearest Neighbors"] = create_knn
			elif sys.argv[i] == "-cfu":
				unite_classifier_ftindices = not unite_classifier_ftindices
			elif sys.argv[i] == "-cfi":
				intersect_classifier_ftindices = not intersect_classifier_ftindices
			elif sys.argv[i] == "-sc":
				save_classifiers = not save_classifiers
			elif sys.argv[i] == "-sds":
				save_data_split = not save_data_split
			elif sys.argv[i] == "-lc":
				if len(sys.argv) > i + 1 and sys.argv[i + 1][0] != '-':
					cload_path = sys.argv[i + 1]
					i += 1
			elif sys.argv[i] == "-lds":
				load_data_split = not load_data_split
			elif sys.argv[i] == "-lcif":
				load_classifier_important_fts = not load_classifier_important_fts
			elif sys.argv[i] == "-noexpl":
				dont_explain = not dont_explain
			elif sys.argv[i] == "-noft":
				dont_extract_features = not dont_extract_features
			elif sys.argv[i] == "-denoise":
				do_denoise = not do_denoise
				denoise_threshold = 0.9
				if len(sys.argv) > i + 1 and sys.argv[i + 1][0] != '-':
					denoise_threshold = float(sys.argv[i + 1])
					i += 1
				if denoise_threshold == 0:
					do_denoise = False
			elif sys.argv[i] == "-topn":
				if len(sys.argv) > i + 1 and sys.argv[i + 1][0] != '-':
					top_n = int(sys.argv[i + 1])
					i += 1
			elif sys.argv[i] == "-topby":
				if len(sys.argv) > i + 1 and sys.argv[i + 1][0] != '-' and sys.argv[i + 1].lower() in possible_metrics:
					top_metric = sys.argv[i + 1].lower()
					i += 1
			elif sys.argv[i] == "-sps":
				save_pair_reduced_sets = not save_pair_reduced_sets 
			i += 1
		
		if not datapath or not keyfile:
			print("Path to dataset or key file was not specified...")
			exit()
		
		if not output_dir:
			print("Output path was not specified...")
			exit()
		
		if not os.path.exists(output_dir):
			os.makedirs(output_dir, exist_ok=True)
		
		original_sensor_data, labels = load_dataset(datapath, keyfile, do_denoise)
		if not load_data_split:
			X_train, X_test, y_train, y_test = train_test_split(original_sensor_data, labels, test_size=0.2, random_state=42)
			smote = SMOTE(random_state=42)
			X_train, y_train = smote.fit_resample(X_train, y_train)
		else:
			if cload_path:
				try:
					with open(os.path.join(cload_path, 'X_train.pkl'), 'rb') as f:
						X_test = pickle.load(f)
					with open(os.path.join(cload_path, 'y_train.pkl'), 'rb') as f:
						y_test = pickle.load(f)
					with open(os.path.join(cload_path, 'X_test.pkl'), 'rb') as f:
						X_test = pickle.load(f)
					with open(os.path.join(cload_path, 'y_test.pkl'), 'rb') as f:
						y_test = pickle.load(f)
				except Exception as e:
					print(e)
					X_train, X_test, y_train, y_test = train_test_split(original_sensor_data, labels, test_size=0.2, random_state=42)
					smote = SMOTE(random_state=42)
					X_train, y_train = smote.fit_resample(X_train, y_train)
			else:
				print('No path was given to FTX result directory')
				X_train, X_test, y_train, y_test = train_test_split(original_sensor_data, labels, test_size=0.2, random_state=42)
				smote = SMOTE(random_state=42)
				X_train, y_train = smote.fit_resample(X_train, y_train)
		
		if save_data_split:
			with open(os.path.join(output_dir, 'X_train.pkl'), 'wb') as f:
				pickle.dump(X_train, f)
			with open(os.path.join(output_dir, 'y_train.pkl'), 'wb') as f:
				pickle.dump(y_train, f)
			with open(os.path.join(output_dir, 'X_test.pkl'), 'wb') as f:
				pickle.dump(X_test, f)
			with open(os.path.join(output_dir, 'y_test.pkl'), 'wb') as f:
				pickle.dump(y_test, f)
			
		
		if len(used_classifiers) == 0 and not cload_path:
			exit()
		
		classifier_performance = []
		loaded_classifiers = set()
		classifiers = {}
		classifiers_all_feature_importances = {}
		classifiers_imp_feature_indices = {}
		if cload_path:
			classifiers = load_classifiers(cload_path)
			loaded_classifiers = { C for C, _ in classifiers.items() }
			if load_classifier_important_fts:
				classifiers_imp_feature_indices, classifiers_all_feature_importances = load_classifier_feature_indices(cload_path)
			for C, _ in classifiers.items():
				y_pred = classifiers[C].predict(X_test)
				accuracy = accuracy_score(y_test, y_pred)
				f1 = f1_score(y_test, y_pred, average='weighted')
				recall = recall_score(y_test, y_pred, average='weighted')
				precision = precision_score(y_test, y_pred, average='weighted')
				print(f"{C} Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}")
				classifier_performance.append({
					'classifier': C,
					'accuracy': accuracy,
					'precision': precision,
					'recall': recall,
					'f1': f1
				})
				
				
				if not dont_extract_features and not load_classifier_important_fts:
					explainer = create_lime_explainer(X_train)
					classifiers_all_feature_importances[C] = extract_feature_importances(classifiers[C], explainer, X_test, y_test)
				
					if not dont_explain:
						plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_0.png", classifiers_all_feature_importances[C][0], '0')
						plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_1.png", classifiers_all_feature_importances[C][1], '1')
					max_nonzero_imps = pick_max_nonzero(classifiers_all_feature_importances[C])
					classifiers_imp_feature_indices[C] = extract_important_feature_indices(max_nonzero_imps)
					
					print(f"Important features count for {C}: {len(classifiers_imp_feature_indices[C])}")
					if not dont_explain:
						plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_filtered_0.png", classifiers_all_feature_importances[C][0], '0', classifiers_imp_feature_indices[C])
						plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_filtered_1.png", classifiers_all_feature_importances[C][1], '1', classifiers_imp_feature_indices[C])
					output_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances.txt", classifiers_all_feature_importances[C])
					output_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_filtered.txt", max_nonzero_imps, classifiers_imp_feature_indices[C])
					np.savetxt(f"{output_dir + os.sep + C}{os.sep}feature_importances_info_indices.txt", classifiers_imp_feature_indices[C].astype(int), delimiter=',', fmt='%d')
				
					if output_reduced_set:
						sensor_data = original_sensor_data[:, classifiers_imp_feature_indices[C]]
						os.makedirs(output_dir + os.sep + C + f'{os.sep}reduced_dataset', exist_ok=True)
						save_sets(output_dir + os.sep + C + f'{os.sep}reduced_dataset', sensor_data, labels)
		
		for C, constructor in used_classifiers.items():
			if C in loaded_classifiers:
				print(f'{C} was already loaded, skipping training...')
				continue
			
			os.makedirs(output_dir + os.sep + C, exist_ok=True)
			classifiers[C] = constructor(X_train, y_train)
			y_pred = classifiers[C].predict(X_test)
			accuracy = accuracy_score(y_test, y_pred)
			f1 = f1_score(y_test, y_pred, average='weighted')
			recall = recall_score(y_test, y_pred, average='weighted')
			precision = precision_score(y_test, y_pred, average='weighted')
			print(f"{C} Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}")
			classifier_performance.append({
				'classifier': C,
				'accuracy': accuracy,
				'precision': precision,
				'recall': recall,
				'f1': f1
			})
			
			
			if not dont_extract_features:
				explainer = create_lime_explainer(X_train)
				classifiers_all_feature_importances[C] = extract_feature_importances(classifiers[C], explainer, X_test, y_test)
			
				if not dont_explain:
					plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_0.png", classifiers_all_feature_importances[C][0], '0')
					plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_1.png", classifiers_all_feature_importances[C][1], '1')
				max_nonzero_imps = pick_max_nonzero(classifiers_all_feature_importances[C])
				classifiers_imp_feature_indices[C] = extract_important_feature_indices(max_nonzero_imps)
				
				print(f"Important features count for {C}: {len(classifiers_imp_feature_indices[C])}")
				if not dont_explain:
					plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_filtered_0.png", classifiers_all_feature_importances[C][0], '0', classifiers_imp_feature_indices[C])
					plot_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_filtered_1.png", classifiers_all_feature_importances[C][1], '1', classifiers_imp_feature_indices[C])
				output_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances.txt", max_nonzero_imps)
				output_feature_importances(f"{output_dir + os.sep + C}{os.sep}feature_importances_filtered.txt", max_nonzero_imps, classifiers_imp_feature_indices[C])
				np.savetxt(f"{output_dir + os.sep + C}{os.sep}feature_importances_info_indices.txt", classifiers_imp_feature_indices[C].astype(int), delimiter=',', fmt='%d')
			
				if output_reduced_set:
					sensor_data = original_sensor_data[:, classifiers_imp_feature_indices[C]]
					os.makedirs(output_dir + os.sep + C + f'{os.sep}reduced_dataset', exist_ok=True)
					save_sets(output_dir + os.sep + C + f'{os.sep}reduced_dataset', sensor_data, labels)
			if save_classifiers:
				with open(f"{output_dir + os.sep + C}{os.sep}classifier.pkl", 'wb') as f:
					pickle.dump(classifiers[C], f)
		
		top_n = len(classifiers) if top_n is None else top_n
		classifier_performance.sort(key=lambda x: x[top_metric], reverse=True)
		classifier_performance = classifier_performance[: top_n]
		top_classifiers = {perf['classifier'] for perf in classifier_performance}
		print(f"Top-{top_n} classifiers(by {top_metric}):")
		for i, perf in enumerate(classifier_performance):
			print(f"{i + 1}. {perf['classifier']}")
		
		
		if unite_classifier_ftindices or intersect_classifier_ftindices and len(classifiers) > 1 and not dont_extract_features:
			walked_pairs = set()
			for clname, cl in classifiers.items():
				if clname not in top_classifiers:
					print(f"{clname} is not in the top of classifiers, skipped")
					continue
				for crname, cr in classifiers.items():
					if crname not in top_classifiers:
						print(f"{crname} is not in the top of classifiers, skipped")
						continue
					
					if clname != crname and f'{clname} + {crname}' not in walked_pairs and f'{crname} + {clname}' not in walked_pairs:
						os.makedirs(output_dir + os.sep + clname + ' + ' + crname, exist_ok=True)
						walked_pairs.add(f'{clname} + {crname}')
						cl_max_nonzero_imps = pick_max_nonzero(classifiers_all_feature_importances[clname])
						cr_max_nonzero_imps = pick_max_nonzero(classifiers_all_feature_importances[crname])
						
						if unite_classifier_ftindices:
							print(f'Union of \'{clname}\' and \'{crname}\'')
							combined_indices = feature_indices_comb(classifiers_imp_feature_indices[clname], classifiers_imp_feature_indices[crname])
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_c.txt", cl_max_nonzero_imps)
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_filtered_c.txt", cl_max_nonzero_imps, combined_indices)
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_c.txt", cr_max_nonzero_imps)
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_filtered_c.txt", cr_max_nonzero_imps, combined_indices)
							np.savetxt(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}combined_feature_importances_info_indices.txt", combined_indices.astype(int), delimiter=',', fmt='%d')
							if not dont_explain:
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_combined_0.png", classifiers_all_feature_importances[clname][0], 'Nominal', combined_indices)
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_combined_0.png", classifiers_all_feature_importances[crname][0], 'Nominal', combined_indices)
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_combined_1.png", classifiers_all_feature_importances[clname][1], 'Faulted', combined_indices)
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_combined_1.png", classifiers_all_feature_importances[crname][1], 'Faulted', combined_indices)
							if output_reduced_set and save_pair_reduced_sets:
								sensor_data = original_sensor_data[:, combined_indices]
								os.makedirs(output_dir + os.sep + clname + ' + ' + crname + f'{os.sep}union_reduced_dataset', exist_ok=True)
								save_sets(output_dir + os.sep + clname + ' + ' + crname + f'{os.sep}union_reduced_dataset', sensor_data, labels)
						if intersect_classifier_ftindices:
							print(f'Intersection of \'{clname}\' and \'{crname}\'')
							intersected_indices = feature_indices_intersect(classifiers_imp_feature_indices[clname], classifiers_imp_feature_indices[crname])
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_i.txt", cl_max_nonzero_imps)
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_filtered_i.txt", cl_max_nonzero_imps, intersected_indices)
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_i.txt", cr_max_nonzero_imps)
							output_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_filtered_i.txt", cr_max_nonzero_imps, intersected_indices)
							np.savetxt(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}intersected_feature_importances_info_indices.txt", intersected_indices.astype(int), delimiter=',', fmt='%d')
							if not dont_explain:
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_intersected_0.png", classifiers_all_feature_importances[clname][0], 'Nominal', intersected_indices)
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_intersected_0.png", classifiers_all_feature_importances[crname][0], 'Nominal', intersected_indices)
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{clname}_feature_importances_intersected_1.png", classifiers_all_feature_importances[clname][1], 'Faulted', intersected_indices)
								plot_feature_importances(f"{output_dir + os.sep + clname + ' + ' + crname}{os.sep}{crname}_feature_importances_intersected_1.png", classifiers_all_feature_importances[crname][1], 'Faulted', intersected_indices)
							if output_reduced_set and save_pair_reduced_sets:
								sensor_data = original_sensor_data[:, intersected_indices]
								os.makedirs(output_dir + os.sep + clname + ' + ' + crname + f'{os.sep}intersection_reduced_dataset', exist_ok=True)
								save_sets(output_dir + os.sep + clname + ' + ' + crname + f'{os.sep}intersection_reduced_dataset', sensor_data, labels)
				
			final_i_featureset = np.array([])
			final_c_featureset = np.array([])
			final_dir_name = ""
			i = 0
			while i < top_n:
				if i == 0 and len(classifiers_imp_feature_indices) >= 2:
					final_i_featureset = feature_indices_intersect(classifiers_imp_feature_indices[classifier_performance[0]['classifier']], classifiers_imp_feature_indices[classifier_performance[1]['classifier']])
					final_c_featureset = feature_indices_comb(classifiers_imp_feature_indices[classifier_performance[0]['classifier']], classifiers_imp_feature_indices[classifier_performance[1]['classifier']])
					final_dir_name = classifier_performance[0]['classifier'] + " + " + classifier_performance[1]['classifier']
					i += 2
					continue
				
				final_i_featureset = feature_indices_intersect(final_i_featureset, classifiers_imp_feature_indices[classifier_performance[i]['classifier']])
				final_c_featureset = feature_indices_comb(final_c_featureset, classifiers_imp_feature_indices[classifier_performance[i]['classifier']])
				final_dir_name = final_dir_name + " + " + classifier_performance[i]['classifier']
				
				i += 1
			
			if not os.path.exists(os.path.join(output_dir, final_dir_name)):
				print(f"Saving {final_dir_name}...")
				os.makedirs(os.path.join(output_dir, final_dir_name), exist_ok=True)
				
				np.savetxt(os.path.join(output_dir, final_dir_name, "intersected_feature_indices.txt"), final_i_featureset.astype(int), delimiter=',', fmt='%d')
				np.savetxt(os.path.join(output_dir, final_dir_name, "united_feature_indices.txt"), final_c_featureset.astype(int), delimiter=',', fmt='%d')
				if output_reduced_set:
					sensor_data = original_sensor_data[:, final_i_featureset]
					os.makedirs(os.path.join(output_dir, final_dir_name, "intersection_reduced_set"), exist_ok=True)
					save_sets(os.path.join(output_dir, final_dir_name, "intersection_reduced_set"), sensor_data, labels)
					sensor_data = original_sensor_data[:, final_c_featureset]
					os.makedirs(os.path.join(output_dir, final_dir_name, "union_reduced_set"), exist_ok=True)
					save_sets(os.path.join(output_dir, final_dir_name, "union_reduced_set"), sensor_data, labels)
		
	