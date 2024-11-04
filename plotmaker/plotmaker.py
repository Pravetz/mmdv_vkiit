import os
import sys
import numpy as np
import subprocess
import math
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

global gnuplot_path
gnuplot_path = 'gnuplot'

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
	with open(datapath + "/" + keyfile, 'r', encoding='utf-8') as f:
		i = 1
		for fline in f:
			with open(datapath + "/" + f"T{i:04}.txt", 'r', encoding='utf-8') as sensor_data:
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

def load_dataset_labels(datapath, keyfile):
	print("Loading dataset...")
	labels = []
	files = []
	with open(datapath + "/" + keyfile, 'r', encoding='utf-8') as f:
		i = 1
		for fline in f:
			labels.append(int(float(fline)))
			files.append(f"T{i:04}")
			
			i += 1
	
	print("Done")
	return np.array(labels), np.array(files)

def plot_to_file(datafile, output_file, terminal='png', do_denoise=False):
	if do_denoise:
		sensor_values = []
		with open(datafile, 'r', encoding='utf-8') as sensor_data:
			for line in sensor_data:
				sensor_values.append(float(line))
		sensor_values = denoise(sensor_values)
		with open("pm_tmp.txt", 'w', encoding='utf-8') as pm_temp:
			for feature in sensor_values:
				pm_temp.write(f'{feature}\n')
		datafile = "pm_tmp.txt"
	
	gnuplot_commands = f"""
	set terminal {terminal}
	set output '{output_file}'
	unset key
	set format x ""
	set format y ""
	plot '{datafile}' using 0:1 with linespoints
	"""
	
	process = subprocess.Popen([gnuplot_path], stdin=subprocess.PIPE)
	process.communicate(gnuplot_commands.encode())
	

def label_counts_empty(label_counters):
	for k, v in label_counters.items():
		if v > 0:
			return False
	
	return True

def save_balanced_dataset(path, keyfile, X, Y):
	with open(path + "/" + keyfile, 'w', encoding='utf-8') as fkey:
		for y in Y:
			fkey.write(f'{y}\n')
	
	for i, x in enumerate(X):
		with open(path + "/" + f"T{i + 1:04}.txt", 'w', encoding='utf-8') as T:
			for feature in x:
				T.write(f'{feature}\n')

if __name__ == "__main__":
	if len(sys.argv) == 1:
		print(f"""
Usage: {sys.argv[0]} <parameters>
Possible parameters:
	-path = specify path to dataset
	-key = specify key file name(in -path)
		{sys.argv[0]} -path path/to/my/data -key \"key.txt\"
	-o = output path(must be a directory)
	-b = perform SMOTE-balancing on input dataset
	-denoise <T> = enable denoise with threshold T, if T is 0, denoise is disabled
	-gppath = set path to gnuplot(\'gnuplot\' by default)
		""")
		exit()
	
	else:
		datapath = ""
		keyfile = ""
		output_dir = ""
		balanced = False
		do_denoise = False
		
		i = 1
		while i < len(sys.argv):
			if sys.argv[i] == "-path":
				datapath = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-key":
				keyfile = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-o":
				output_dir = sys.argv[i + 1]
				i += 1
			elif sys.argv[i] == "-b":
				balanced = not balanced
			elif sys.argv[i] == "-denoise":
				do_denoise = not do_denoise
				denoise_threshold = 0.9
				if len(sys.argv) > i + 1 and sys.argv[i + 1][0] != '-':
					denoise_threshold = float(sys.argv[i + 1])
					i += 1
				if denoise_threshold == 0:
					do_denoise = False
			elif sys.argv[i] == "-gppath":
				if len(sys.argv) > i + 1 and sys.argv[i + 1][0] != '-':
					gnuplot_path = sys.argv[i + 1]
					i += 1
			i += 1
		
		if not datapath or not keyfile:
			print("Path to dataset or key file was not specified...")
			exit()
		
		if not output_dir:
			print("Output path was not specified...")
			exit()
		
		if not os.path.exists(output_dir):
			os.makedirs(output_dir, exist_ok=True)
		
		if balanced:
			original_sensor_data, labels = load_dataset(datapath, keyfile, do_denoise)
			smote = SMOTE(random_state=42)
			X, y = smote.fit_resample(original_sensor_data, labels)
			os.makedirs(output_dir + '/balanced_set', exist_ok=True)
			save_balanced_dataset(output_dir + '/balanced_set', keyfile, X, y)
			datapath = output_dir + '/balanced_set'
			output_dir += '/plot_classes'
		
		labels, files = load_dataset_labels(datapath, keyfile)
		unique_labels = np.unique(labels)
		
		for label in unique_labels:
			os.makedirs(output_dir + f'/{label}', exist_ok=True)
		
		for i, file in enumerate(files):
			plot_to_file(f"{datapath}/{file}.txt", output_dir + f'/{labels[i]}/{file}.png', do_denoise=do_denoise)
		