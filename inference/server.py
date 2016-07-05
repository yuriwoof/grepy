"""
classify.py is an out-of-the-box image classifer callable from the command line.
By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import io
import sys
import time
import ConfigParser

from flask import Flask, request
from flask import jsonify

import caffe

app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            # input file
            inputs = [caffe.io.load_image(image)]

            print("Classifying %d inputs." % len(inputs))

            # Classify.
            start = time.time()
            predictions = classifier.predict(inputs, not center_only)
            print("Done in %.2f s." % (time.time() - start))

            prediction = zip(predictions[0].tolist(),categories)
            prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)   

        else:
          return jsonify(error="Invalid request")

    return jsonify(resutl=prediction[:num_of_result])

if __name__ == '__main__':
    caffe_dir = os.environ.get("CAFFE_ROOT")

    # read config.ini file    
    with open("config.ini") as f:
      file = f.read()
    config = ConfigParser.SafeConfigParser()
    config.readfp(io.BytesIO(file))

    # Set parameter
    model_def = os.path.join(caffe_dir,config.get('option','model_def'))
    pretrained_model = os.path.join(caffe_dir,config.get('option','pretrained_model'))
    mean_file = os.path.join(caffe_dir,config.get('option','mean_file'))
    categories = np.loadtxt(os.path.join(caffe_dir,
                "data/ilsvrc12/synset_words.txt"),str,delimiter="\t")
    # center_only = config.get('option','center_only')
    center_only = True
    #input_scale = config.get('option','input_scale') 
    input_scale = None
    # num_of_result = config.get('option','num_of_result')
    num_of_result = 3

    images_dim = config.get('option','images_dim')
    image_dims = [int(s) for s in images_dim.split(',')]

    # raw_scale = config.get('option','raw_scale')
    raw_scale = 255.0
    channel_swap = config.get('option','channel_swap')

    mean, channel_swap = None, None
    if mean_file:
        mean = np.load(mean_file)
    if channel_swap:
        channel_swap = [int(s) for s in channel_swap.split(',')]

    caffe.set_mode_cpu()
    print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(model_def, pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)

    app.debug = True
    app.run(host='0.0.0.0')

