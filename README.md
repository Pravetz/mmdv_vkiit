# mmdv_vkiit
This repository contains a software complex, developed for All-Ukrainian competition-hackathon of scientific works of young scientists in the field of intellectual information technologies in 2024(Всеукраїнський конкурс-хакатон наукових робіт молодих учених в галузі інтелектуальних інформаційних технологій 2024 року).
It consists of three CLI applications(FTX, NNT and Plotmaker)
and a Flask web-app, which serves as a "graphical demo" for aforementioned CLI apps, allowing user to classify images using MobileNetV2 models or generate LIME-explanations with important feature filtering for ML classifiers like Random Forest, Naive Bayes etc.

# FTX
Or "FeaTure eXtractor", is used for feature extraction based on LIME.

It is configured using following console parameters:

```
-path = specify path to dataset
-key = specify key file name(in -path)
	ftx.py -path path/to/my/data -key "key.txt"
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
```
# NNT
Or "Neural Network Trainer" is used for training of MobileNetV2 model on given graphical data.

It is configured using following console parameters:

```
-path = specify path to dataset
	nnt.py -path path/to/my/data
-o = output path(must be a directory)
-tr = run NNT in neural network model training mode
-te = run NNT in model testing mode(loads model from -path)
-vl = evaluate and score model
-trunc = truncate input dataset, if the classes are unbalanced
-epochs = set epoch count(default is 5)
-nogpu = don't use GPU for training/testing, does nothing on systems with no GPU
```
Commands in model testing mode:
```
exit = exits NNT
batch = loads a batch of samples and makes predictions
	batch /my/batch/dir
batchlf = loads a batch of samples, makes predictions and saves predictions to a log file
	batchlf /my/batch/dir my.log
any other input is treated as a path to a single file for prediction
```

# Plotmaker
This one is used for conversion of text dataset into graphical format using Gnuplot, make sure you have installed it(you can get it here: https://gnuplot.sourceforge.net/) before using plotmaker.

It is configured using following console parameters:

```
-path = specify path to dataset
-key = specify key file name(in -path)
	plotmaker.py -path path/to/my/data -key "key.txt"
-o = output path(must be a directory)
-b = perform SMOTE-balancing on input dataset
-denoise <T> = enable denoise with threshold T, if T is 0, denoise is disabled
-gppath = set path to gnuplot('gnuplot' by default)
```
