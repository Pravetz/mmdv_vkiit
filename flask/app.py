import os
import re
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from werkzeug.utils import secure_filename
import sklearn
import joblib
import lime
import lime.lime_tabular
import plotly.graph_objects as go
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'keras', 'pkl', 'txt'}
classes = [
	"Nominal",
	"Faulted"
]

models = {
	'model1': load_model('models/mobilenet_bfull.keras'),
	'model2': load_model('models/mobilenet_bunited.keras'),
	'model3': load_model('models/mobilenet_bint.keras')
}

model_display_names = {
	'model1': 'MobileNetV2, full dataset',
	'model2': 'MobileNetV2, union reduced dataset',
	'model3': 'MobileNetV2, intersection reduced dataset'
}

lime_models = {
	'svm': joblib.load('models/svmclass.pkl'),
	'random_forest': joblib.load('models/rfclass.pkl'),
	'naive_bayes': joblib.load('models/nbclass.pkl'),
	'knn': joblib.load('models/knnclass.pkl'),
	'gradient_boosting': joblib.load('models/gbclass.pkl')
}

lime_model_display_names = {
	'svm': 'SVM',
	'random_forest': 'Random Forest',
	'naive_bayes': 'Naive Bayes',
	'knn': 'K-Nearest Neighbors',
	'gradient_boosting': 'Gradient Boosting'
}

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_single_file(file):
	sensor_values = []
	for line in file:
		sensor_values.append(float(line))
	
	return np.array(sensor_values)

def preprocess_image(image_path):
	img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)
	return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

@app.route('/', methods=['GET', 'POST'])
def index():
	lime_explanation_html = None
	predicted_class = None
	display_model_name = None
	uploaded_image = None
	
	if request.method == 'POST':
		if 'classify' in request.form:
			if 'file' in request.files:
				file = request.files['file']
				if file and allowed_file(file.filename):
					filename = secure_filename(file.filename)
					filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
					file.save(filepath)

					model_name = request.form.get('model')
					model = models.get(model_name)

					img_array = preprocess_image(filepath)
					prediction = model.predict(img_array)
					predicted_class = 1 if prediction[0] > 0.5 else 0
					predicted_class = classes[predicted_class]

					display_model_name = model_display_names.get(model_name)
					uploaded_image = filepath
		
		elif 'explain' in request.form:
			model_name = request.form.get('lime_model')
			lime_model = lime_models.get(model_name)

			file1 = request.files['file1']
			file2 = request.files['file2']

			if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
				samples1 = load_single_file(file1)
				samples2 = load_single_file(file2)

				if samples1.shape[0] != samples2.shape[0]:
					return "Error: The two sample files must have the same number of features."

				combined_samples = np.vstack((samples1, samples2))

				explainer = lime.lime_tabular.LimeTabularExplainer(
					training_data=combined_samples,
					mode="classification",
					feature_names=[f"Feature {i}" for i in range(combined_samples.shape[1])],
					class_names=["0", "1"],
					discretize_continuous=False
				)

				idx = 0
				try:
					exp = explainer.explain_instance(combined_samples[idx], get_predict_proba_function(lime_model), num_features=combined_samples.shape[1])

					feature_importances = exp.as_list()
					important_indices_features = extract_important_feature_indices(feature_importances)
					classname = classes[np.argmax(exp.predict_proba)]

					if 'filter' in request.form and request.form['filter'] == 'yes':
						lime_explanation_html = plot_feature_importances(feature_importances, classname, flt=important_indices_features)
					else:
						lime_explanation_html = plot_feature_importances(feature_importances, classname, flt=None)
				except Exception as e:
					return f'{e}'
	
	return render_template('index.html', uploaded_image=uploaded_image, prediction=predicted_class, model_name=display_model_name, lime_explanation=lime_explanation_html)

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

def extract_important_feature_indices(feature_importances):
	important_indices = []
	
	mean_importance = np.mean([abs(importance) for _, importance in feature_importances])
	
	feature_pattern = re.compile(r"Feature (\d+)")
	
	for feature, importance in feature_importances:
		if abs(importance) >= mean_importance:
			idx_match = feature_pattern.search(feature)
			important_indices.append(int(idx_match.group(1)))
	
	return np.array(important_indices)

def plot_feature_importances(importances, classname, flt=None):
	features, importances = filter_tuples(importances, flt)
	features = np.array(features)
	importances = np.array(importances)
	
	sorted_idx = np.argsort(np.abs(importances))[::-1]
	features = features[sorted_idx]
	importances = importances[sorted_idx]
	
	colors = ['blue' if i > 0 else 'orange' for i in importances]
	
	fig = go.Figure(go.Bar(
		x=importances,
		y=features,
		orientation='h',
		marker_color=colors,
		text=[f'Feature {ft} importance: {imp}' for ft, imp in zip(features, importances)],
		hoverinfo='text',
		showlegend=False
	))
	
	fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='blue'), name='Supports'))
	fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='orange'), name='Contradicts'))
	
	fig.update_layout(
		title=f'Feature Importance, Class {classname}',
		xaxis_title='Weight',
		yaxis_title='Feature',
		yaxis=dict(autorange="reversed"),
		shapes=[dict(type="line", yref="paper", x0=0, x1=0, y0=0, y1=1, line=dict(color="gray", width=2, dash="dash"))],
		height=800
	)
	
	return fig.to_html(full_html=False)


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
	
	raise ValueError(f"Model of type {type(classifier).__name__} is either not supported or corresponding libraries were not imported.")


@app.route('/lime_explanation', methods=['POST'])
def lime_explanation():
	if 'file1' not in request.files or 'file2' not in request.files:
		return redirect(request.url)

	model_name = request.form.get('lime_model')
	lime_model = lime_models.get(model_name)

	file1 = request.files['file1']
	file2 = request.files['file2']

	if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
		samples1 = load_single_file(file1)
		samples2 = load_single_file(file2)
		
		if samples1.shape[0] != samples2.shape[0]:
			return "Error: The two sample files must have the same number of features."
		
		combined_samples = np.vstack((samples1, samples2))
		
		explainer = lime.lime_tabular.LimeTabularExplainer(
			training_data=combined_samples,
			mode="classification",
			feature_names=[f"Feature {i}" for i in range(combined_samples.shape[1])],
			class_names=["0", "1"],
			discretize_continuous=False
		)
		
		idx = 0
		try:
			exp = explainer.explain_instance(combined_samples[idx], get_predict_proba_function(lime_model), num_features=combined_samples.shape[1])
			
			feature_importances = exp.as_list()
			important_indices_features = extract_important_feature_indices(feature_importances)
			classname = classes[np.argmax(exp.predict_proba)]
			
			if 'filter' in request.form and request.form['filter'] == 'yes':
				lime_explanation_html = plot_feature_importances(feature_importances, classname, flt=important_indices_features)
			else:
				lime_explanation_html = plot_feature_importances(feature_importances, classname, flt=None)
		except Exception as e:
			return f'{e}';
		
		return render_template('index.html', lime_explanation=lime_explanation_html)

if __name__ == '__main__':
	app.run(debug=True)
