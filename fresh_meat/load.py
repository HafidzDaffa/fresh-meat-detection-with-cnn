import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf


def load_graph_weights(): 
	json_file = open('model/meat.json','r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights('model/meat_model.h5')
	print("Loaded Model from disk")

	model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
	graph = tf.get_default_graph()

	return model, graph
