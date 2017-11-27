import configparser

from flask import Flask, render_template, flash, request

import requests
URL = 'https://cube-6deda82f-1a50-41c8-9df0-95a9674ce18b.api.sherlockml.io/predict'
#URL = 'https://cube-8027cc21-be15-4ce9-9930-7586b70c39b5.api.sherlockml.io/predict'
API_KEY_PATH = '/project/training/Deployment/secret/api_keys.ini'


# App config.
DEBUG = False
app = Flask(__name__)
app.config.from_object(__name__)

# This is needed ... leave it as is
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/", methods=['GET', 'POST'])
def index():

    config = configparser.ConfigParser()
    config.read(API_KEY_PATH)
    api_key = config['model-api']['key']

    if request.method == 'POST':

        r = requests.get(
            URL,
            headers={'SherlockML-UserAPI-Key': api_key},
            params=dict(request.form)
        )

        is_bankrupt = r.json()['will_go_bankrupt']

        flash(is_bankrupt)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=8888, debug=DEBUG)
