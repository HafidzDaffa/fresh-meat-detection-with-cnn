from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
from load import *
from matplotlib.pyplot import imread
import cv2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = os.path.basename('upload')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
H = 150
W = 150

model, graph = load_graph_weights()
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index_page():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	file = request.files['fileupload']
	filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
	file.save(filename)
	img = imread(filename, mode='L')
	img = cv2.resize(img, (150, 150))
	img = img.reshape(3, 150, 150, 3)
	with graph.as_default():
		out = model.predict(img)
		print(out)
		print(np.argmax(out,axis=1))
		response = np.array_str(np.argmax(out,axis=1))
		return response		
		#return render_template('predicted.html')
if __name__ == "__main__":
	app.run()
