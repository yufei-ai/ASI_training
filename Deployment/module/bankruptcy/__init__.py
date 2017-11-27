import collections
import configparser
import os

import numpy
import pandas
import requests

from sklearn.externals import joblib


PATH_TO_CSV = '/project/training/Deployment/data/companies.csv'
URL = 'https://cube-14d6bb1c-b959-4233-af85-84aef24af6e6.api.sherlockml.io'


class ApiSession(object):
    """Bankruptcy data API session."""

    def __init__(self, credentials_path):
        config = configparser.ConfigParser()
        config.read(credentials_path)
        self.headers = {'SherlockML-UserAPI-Key': config['data-api']['key']}

    def _get_companies(self, page):
        response = requests.get(
            os.path.join(URL, 'companies'),
            headers=self.headers,
            params={'page': page}
        )
        response.raise_for_status()
        return response.json()

    def get_companies(self):
        # TODO Pagination
        # Rewrite this function so that you get the data for all the companies
        all_items = []
        k = 0
        maxi_number = self._get_companies(page=k)['maximum_items_per_page']
        items = self._get_companies(page=k)['items']
        all_items.extend(items)
        
        while len(items) == maxi_number :
            k+=1
            items = self._get_companies(page=k)['items']
            all_items.extend(items)
            if k>1000000:
                break
        #companies = self._get_companies(page=1)['items']
        #all_items.extend(companies)
        # ...
        # Tip: self._get_companies(page)['maximum_items_per_page'] returns
        # the maximum number of items per page and this is independent of page
        return all_items


def predict(
    model_path,
    competitiveness='A',
    credibility='A',
    financial_flexibility='A',
    industrial_risk='A',
    management_risk='A',
    operating_risk='A'
):
    """Predict bankruptcy given model path and features.

    Allowed risk factor values are str 'A' (default), 'P' or 'N'.
    """

    parameters = locals()
    parameters.pop('model_path')

    # Check risk factor inputs
    for k, v in parameters.items():
        if v not in ['A', 'N', 'P']:
            raise ValueError('invalid risk factor value')

    # Read risk factor columns from the data file so you get the ordering right
    df = pandas.read_csv(PATH_TO_CSV, nrows=0)
    risk_factor_columns = list(df.columns)
    risk_factor_columns.remove('company_id')
    risk_factor_columns.remove('is_bankrupt')

    # Prepare input for model
    risk_factors = collections.OrderedDict()
    for column in risk_factor_columns:
        for suffix in ['A', 'N', 'P']:  # order matters
            risk_factors[column + '_' + suffix] = 0
    for parameter in parameters:
        risk_factors[parameter + '_' + parameters[parameter]] = 1
    risk_factors = [v for k, v in risk_factors.items()]

    # Load model
    clf = joblib.load(model_path)

    # Correct format for input
    risk_factors = numpy.array(risk_factors).reshape(1, -1)
    
    # Predict
    prediction = bool(clf.predict(risk_factors)[0])

    return prediction
