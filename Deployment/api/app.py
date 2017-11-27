#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import flask

import bankruptcy


DEBUG = False

# This is where you host API's on SherlockML
PORT = 8000

# This is where your trained model sits
MODEL_PATH = '/project/training/Deployment/data/model.pkl'


app = flask.Flask(__name__)


def abort(code, message):
    """JSON errors because Flask defaults to HTML."""
    message = 'error {}: {}'.format(code, message)
    flask.abort(flask.make_response(flask.jsonify(message=message), code))


@app.route('/', methods=['GET'])
def index():
    return flask.jsonify({'help': 'one endpoint: /predict'})


@app.route('/predict', methods=['GET'])
def predict():
    """Predict whether or not a company will go bankrupt.

    Risk factors are passed in as query parameters. They all default to 'A'
    or 'average risk' if not specified. The other allowed values are 'P' and
    'N' for positive and negative risk respectively.
    
    The risk factors are: competitiveness, credibility, financial_flexibility,
    industrial_risk, management_risk, operating_risk .
    """

    risk_factors = [
        'competitiveness',
        'credibility',
        'financial_flexibility',
        'industrial_risk',
        'management_risk',
        'operating_risk'
    ]

    predict_kwargs = {}
    for risk_factor in risk_factors:
        value = flask.request.args.get(risk_factor)
        if value is not None:
            predict_kwargs[risk_factor] = value
        else:
            predict_kwargs[risk_factor] = 'A'

    # Prepare output
    output = {
        'status': None,
        'risk_factors': None,
        'will_go_bankrupt': None
    }
    
    try:
        will_go_bankrupt = bankruptcy.predict(MODEL_PATH, **predict_kwargs)
        return flask.jsonify({
            'will_go_bankrupt': will_go_bankrupt,
            'risk_factors': predict_kwargs
        })
    except ValueError as e:
        abort(400, '{}'.format(e))


if __name__ == '__main__':
    app.run(port=PORT, debug=DEBUG)
