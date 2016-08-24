"""
server.py

This is the inference server via Caffe.
"""
import numpy as np
import os
import io
import time
import ConfigParser

from flask import g
from flask import Flask, request
from flask import jsonify

import caffe

app = Flask(__name__)


def initialize(app):
    print("Initialize...")

    ctx = app.app_context()
    ctx.push()

    caffe_dir = os.environ.get("CAFFE_ROOT")

    # read config.ini file
    with open("config.ini") as f:
        file = f.read()
        config = ConfigParser.SafeConfigParser()
        config.readfp(io.BytesIO(file))

    # Set parameter
    model_def = os.path.join(caffe_dir, config.get('option', 'model_def'))
    pretrained_model = os.path.join(caffe_dir, config.get('option',
                                    'pretrained_model'))
    mean_file = os.path.join(caffe_dir, config.get('option', 'mean_file'))
    g.categories = np.loadtxt(os.path.join(caffe_dir,
                              "data/ilsvrc12/synset_words.txt"),
                              str, delimiter="\t")
    # center_only = config.get('option','center_only')
    g.center_only = True
    # input_scale = config.get('option','input_scale')
    input_scale = None
    # num_of_result = config.get('option','num_of_result')
    g.num_of_result = 3

    images_dim = config.get('option', 'images_dim')
    image_dims = [int(s) for s in images_dim.split(',')]

    # raw_scale = config.get('option','raw_scale')
    raw_scale = 255.0
    channel_swap = config.get('option', 'channel_swap')

    mean, channel_swap = None, None
    if mean_file:
        mean = np.load(mean_file)
    if channel_swap:
        channel_swap = [int(s) for s in channel_swap.split(',')]

    caffe.set_mode_cpu()

    # Make classifier.
    g.classifier = caffe.Classifier(model_def, pretrained_model,
                                    image_dims=image_dims, mean=mean,
                                    input_scale=input_scale,
                                    raw_scale=raw_scale,
                                    channel_swap=channel_swap)


def array2dict(x, keys):
    """Convert list to dictionary.
    """
    x = np.asarray(x)

    dict = {}
    for i in range(len(keys)):
        dict[keys[i]] = x[i]

    return dict


@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':

        classifier = getattr(g, 'classifier', None)
        categories = getattr(g, 'categories', None)
        center_only = getattr(g, 'center_only', None)
        num_of_result = getattr(g, 'num_of_result', None)

        image = request.files['image']
        if image:
            # input file
            inputs = [caffe.io.load_image(image)]

            print("Classifying %d inputs." % len(inputs))

            # Classify.
            start = time.time()
            predictions = classifier.predict(inputs, not center_only)
            print("Done in %.2f s." % (time.time() - start))

            # Merge with label array
            prediction = zip(predictions[0].tolist(), categories)
            prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

            dl = []
            keys = ['score', 'name']
            for i in range(len(prediction[:num_of_result])):
                    dl.append(array2dict(prediction[:num_of_result][i], keys))

        else:
            return jsonify(error="Invalid request")

    return jsonify(result=dl)


# initialize
initialize(app)


if __name__ == '__main__':

    app.debug = True
    app.run(host='0.0.0.0')
