#!/ericthestein/bin/python3.5

from flask import Flask, send_from_directory, flash, request, redirect, url_for
from flask_restful import reqparse, abort, Api, Resource
from werkzeug.utils import secure_filename
import werkzeug
import os

import numpy as np
import tensorflow as tf
from theano import *
from keras.models import model_from_yaml
from keras.utils import to_categorical
import librosa

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('recording', required= True, type=werkzeug.datastructures.FileStorage, location='files')

# PREVENT CRASH
'''
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
'''
'''
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, device_count={'CPU': 1}, gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
'''

# LOAD MODEL
yaml_file = open("/home/ericthestein/mysite/model.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)


# LOAD WEIGHTS
model.load_weights("/home/ericthestein/mysite/model.h5")
print("Loaded model from disk")


# GET LABELS FUNCTION
DATA_PATH = "/home/ericthestein/mysite/recordings/"
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# CONVERT .WAV FILE TO MFCC FUNCTION
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

# PREDICT FUNCTION
feature_dim_1 = 20
feature_dim_2 = 11
channel = 1

def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    rawModelPrediction = model.predict(sample_reshaped)
    alteredSet = [rawModelPrediction[0][0]-0.62, rawModelPrediction[0][1]]
    print(alteredSet)
    return get_labels()[0][
            np.argmax(alteredSet)
    ]

class onInvoke(Resource):
    def post(self):
        # UPLOAD FILE
        args = parser.parse_args()
        recording = args.recording
        recording.save(secure_filename(recording.name))
        filepath = recording.name
        # RETURN CLASSIFICATION
        classification = predict(filepath, model)
        print("classification: " + classification)
        os.remove(filepath)
        return classification

api.add_resource(onInvoke, '/')
